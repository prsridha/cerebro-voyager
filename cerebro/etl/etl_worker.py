# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import gc
import sys
import time
import json
import uuid
import traceback
import pandas as pd
import multiprocessing
from pathlib import Path
from functools import partial
from invoke import run as local
from kubernetes import client, config
from pathos.multiprocessing import ProcessPool

from cerebro.util.params import Params
import cerebro.kvs.constants as kvs_constants
from cerebro.util.voyager_io import VoyagerIO
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.util.cerebro_logger import CerebroLogger


class EtlProcess:
    def __init__(self, worker_id, mode, shard_multiplicity, params, etl_spec, is_feature_download, progress_dict):
        logging = CerebroLogger("worker-{}".format(worker_id))
        self.logger = logging.create_logger("etl-worker-process")

        self.mode = mode
        self.params = params
        self.etl_spec = etl_spec
        self.worker_id = worker_id
        self.progress_dict = progress_dict
        self.shard_multiplicity = shard_multiplicity
        self.is_feature_download = is_feature_download
        self.output_path = self.params.etl[mode]["output_path"]

        self.logger.info("Initialized ETL Process object")

    def download_file(self, filepath, file_io):
        mode = self.mode
        params = self.params

        if params.etl["download_type"] == "server":
            from_path = os.path.join(params.etl[mode]["multimedia_url"], filepath)
            to_path = os.path.join(params.etl[mode]["multimedia_download_path"], filepath)
            to_path_dir = "/".join(to_path.split("/")[:-1])

            # download file
            Path(to_path_dir).mkdir(parents=True, exist_ok=True)
            local("cp -R -u -p {} {}".format(from_path, to_path))

        elif params.etl["download_type"] == "url":
            exclude_prefix = params.etl[mode]["multimedia_url"]
            from_path = os.path.join(params.etl[mode]["multimedia_url"], filepath)
            to_path = params.etl[mode]["multimedia_download_path"]

            # download file
            Path(os.path.dirname(to_path)).mkdir(parents=True, exist_ok=True)
            file_io.download(to_path, from_path, exclude_prefix)
        else:
            # error
            pass

    def process_data(self, shard):
        # pathos 'self' issue - move this function out of the class and
        # pass an object of EtlProcess instead of 'self'.
        row_count = 0
        process_id = str(uuid.uuid4())
        features = list(shard.columns)
        num_shard_rows = shard.shape[0]
        m_factor = int(num_shard_rows / self.shard_multiplicity)

        # create a KVS handle
        kvs = KeyValueStore()

        file_io = None
        if self.params.etl["download_type"] == "url":
            file_io = VoyagerIO()

        res_partition = []
        for _, row in shard.iterrows():
            for i, feature_name in enumerate(features):
                if self.is_feature_download[i]:
                    try:
                        self.download_file(row[feature_name], file_io)
                    except Exception as e:
                        gc.collect()
                        err_msg = str(e) + "\n" + traceback.format_exc()
                        kvs.set_error(str(err_msg))
                        return

            # run through user's row prep function
            to_path = os.path.join(self.params.etl[self.mode]["multimedia_download_path"])
            row_id = row["id"]
            try:
                if self.mode == "predict":
                    input_tensor, _ = self.etl_spec.row_prep(row, self.mode, to_path)
                    res_partition.append([row_id, input_tensor])
                else:
                    input_tensor, output_tensor = self.etl_spec.row_prep(row, self.mode, to_path)
                    res_partition.append([row_id, input_tensor, output_tensor])
            except Exception as e:
                err_msg = str(e) + "\n" + traceback.format_exc()
                kvs.set_error(str(err_msg))
                return

            # compute conditional values
            is_multiple = (row_count > 0) and (row_count % m_factor == 0)
            is_last_row = row_count == num_shard_rows - 1
            is_last_sub_shard = row_count / m_factor == self.shard_multiplicity

            # push progress to dict at every 100th row
            if row_count % 100 == 0 or is_last_row:
                progress = (row_count + 1) / num_shard_rows
                self.progress_dict[process_id] = progress

            # update row count
            row_count += 1

            # save partition to Pickle file at intervals of "multiplicity"; extra rows go to last sub shard
            if (is_multiple and not is_last_sub_shard) or is_last_row:
                if self.mode == "predict":
                    result = pd.DataFrame(res_partition, columns=["id", "input_tensor"])
                else:
                    result = pd.DataFrame(res_partition, columns=["id", "input_tensor", "output_tensor"])
                m_str = "_" + str(int(row_count % self.shard_multiplicity))
                path = os.path.join(self.output_path, "data_" + str(process_id) + m_str + ".pkl")
                self.logger.info("Saving to path:" + str(path))
                print("Saving to path:" + str(path))
                result.to_pickle(path)

                # clear memory and recreate variables
                del result
                del res_partition
                gc.collect()
                res_partition = []


class ETLWorker:
    def __init__(self):
        # obtain worker_id from env variable
        worker_name = os.environ.get("ORDINAL_ID")
        worker_id = int(worker_name.split("-")[-1])

        logging = CerebroLogger("worker-{}".format(worker_id))
        self.logger = logging.create_logger("etl-worker")

        self.params = None
        self.shards = None
        self.etl_spec = None
        self.metadata_df = None
        self.progress_dict = None
        self.worker_id = worker_id
        self.is_feature_download = None

        # load values from cerebro-info configmap
        namespace = os.environ['NAMESPACE']
        username = os.environ['USERNAME']
        config.load_kube_config()
        v1 = client.CoreV1Api()
        cm = v1.read_namespaced_config_map(name='{}-cerebro-info'.format(username), namespace=namespace)
        cm_data = json.loads(cm.data["data"])
        self.user_ids = (cm_data["uid"], cm_data["gid"])
        user_code_path = cm_data["user_code_path"]
        self.shard_multiplicity = cm_data["shard_multiplicity"]

        # get node info
        cm = v1.read_namespaced_config_map(name='cerebro-node-hardware-info', namespace=namespace)
        node_info = json.loads(cm.data["data"])
        self.num_gpus = node_info["num_gpus"]

        # add user repo dir to sys path for library discovery
        sys.path.insert(0, user_code_path)

        # get number of processes
        # self.num_process = 48
        cores = int(os.cpu_count())
        self.num_process = cores if cores >= 48 else 48

        # initialize necessary handlers
        self.kvs = KeyValueStore()
        self.p = ProcessPool(self.num_process)

        self.logger.info("Starting Cerebro worker {}".format(worker_id))
        print("Starting Cerebro worker {}".format(worker_id))

    def initialize_worker(self):
        # initialize params and get features to download
        self.params = Params()

        if self.kvs.etl_get_spec():
            self.etl_spec = self.kvs.etl_get_spec()
            self.is_feature_download = self.etl_spec.set_features()

            # run user-defined worker dependency routine
            self.etl_spec.initialize_worker()

            self.logger.info("Installed worker dependencies and loaded params on worker{}".format(self.worker_id))
            print("Installed worker dependencies and loaded params on worker", self.worker_id)

        return

    def shard_metadata(self, mode):
        # read csv from shared storage
        csv_path = os.path.join(self.params.etl[mode]["partition_path"], "worker" + str(self.worker_id) + ".csv")
        self.metadata_df = pd.read_csv(csv_path)
        self.logger.info("Received metadata of size {} on worker {}".format(
            len(self.metadata_df.index), self.worker_id))
        print("Received metadata of size", len(self.metadata_df.index), "on worker ", self.worker_id)

        n = self.num_process
        self.shards = []
        nrows = len(self.metadata_df.index)
        shard_size = nrows // n
        for i in range(n):
            if i == (n - 1):
                shard = self.metadata_df.iloc[i * shard_size:, :]
            else:
                shard = self.metadata_df.iloc[i * shard_size: (i + 1) * shard_size, :]
            self.shards.append(shard)

        self.logger.info("Sharding metadata completed on worker{}".format(self.worker_id))
        print("Sharding metadata completed on worker{}".format(self.worker_id))

    def run_data_parallel(self, mode):
        self.logger.info("Started data parallel row processing on worker{}".format(self.worker_id))
        print("Started data parallel row processing on worker", self.worker_id)

        # create ETL output directories
        output_path = self.params.etl[mode]["output_path"]
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # read and shard metadata
        self.shard_metadata(mode)

        # initialize processes
        manager = multiprocessing.Manager()
        self.progress_dict = manager.dict()
        proc = EtlProcess(self.worker_id, mode, self.shard_multiplicity, self.params,
                          self.etl_spec, self.is_feature_download, self.progress_dict)

        self.p.restart()
        self.p.imap(proc.process_data, self.shards)
        self.logger.info("Started row processing on all {} cores".format(self.num_process))

        # track progress and update to KVS
        done = 0
        while done < self.num_process:
            err = self.kvs.get_error()
            if err:
                # exit with an error code
                self.logger.error("Caught error in Worker {}".format(self.worker_id))
                self.logger.error(str(err))

            # count number of completed processes
            done = sum(value == 1 for value in self.progress_dict.values())

            # update KVS every half second
            percentage = (sum(self.progress_dict.values()) / self.num_process) * 100
            self.kvs.etl_set_worker_progress(self.worker_id, percentage)
            # time.sleep(0.2)

        self.p.close()
        self.p.join()

        # combine all Pickle files into a single file for train, test and predict modes
        if mode != "val":
            self.logger.info("Coalescing {} dataset shards to a single file".format(mode))
            all_files = [os.path.abspath(os.path.join(output_path, f))
                         for f in os.listdir(output_path)]
            pickle_files = [file for file in all_files if file.endswith('.pkl')]
            all_df = []
            for f in pickle_files:
                df = pd.read_pickle(f)
                all_df.append(df)
            combined_df = pd.concat(all_df)
            combined_df.to_pickle(os.path.join(output_path, "{}_data{}.pkl".format(mode, self.worker_id)))

            # delete all individual Pickle files
            for file in pickle_files:
                try:
                    os.remove(file)
                except OSError as e:
                    print(f"Error deleting {file}: {e}")

        print("Completed process partition on worker {}".format(self.worker_id))
        self.logger.info("Completed process partition on worker {}".format(self.worker_id))

    def upload_processed_data(self, mode):
        # create I/O object for reads and writes
        update_progress_fn = partial(self.kvs.etl_set_worker_progress, self.worker_id)
        file_io = VoyagerIO(update_progress_fn)

        self.logger.info("Beginning upload of processed ETL data")
        prefix = os.path.join(self.params.etl["etl_dir"], mode)
        output_path = self.params.etl[mode]["output_path"]
        exclude_prefix = output_path
        file_io.upload(output_path, prefix, exclude_prefix)
        self.logger.info("Completed upload of {} data to destination from worker {}".format(mode, self.worker_id))

        # change ownership of output dir
        uid, gid = self.user_ids
        os.chown(output_path, uid, gid)
        # Iterate through all directories and files within the current directory
        for root, dirs, files in os.walk(output_path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                os.chown(dir_path, uid, gid)

            for file_name in files:
                file_path = os.path.join(root, file_name)
                os.chown(file_path, uid, gid)

    def download_processed_data(self, mode):
        # create I/O object for reads and writes
        update_progress_fn = partial(self.kvs.etl_set_worker_progress, self.worker_id)
        file_io = VoyagerIO(update_progress_fn)

        self.logger.info("Beginning download of processed ETL data")
        exclude_prefix = os.path.join(self.params.etl["etl_dir"], mode)
        prefix = os.path.join(self.params.etl["etl_dir"], mode, "{}_data{}.pkl".format(mode, self.worker_id))
        Path(self.params.etl[mode]["output_path"]).mkdir(parents=True, exist_ok=True)
        output_path = self.params.etl[mode]["output_path"]
        file_io.download(output_path, prefix, exclude_prefix)
        self.logger.info("Completed download of {} data from destination on worker {}".format(mode, self.worker_id))

    def serve_forever(self):
        self.logger.info("Started ETL worker server on worker{}".format(self.worker_id))

        done = False
        prev_task_mode = (None, None)
        task, mode = self.kvs.etl_get_task()
        if task and task != kvs_constants.ETL_TASK_INITIALIZE:
            # recovered
            self.logger.info("Worker{} restart detected in ETL".format(self.worker_id))
            self.kvs.set_restarts(self.worker_id)

            self.initialize_worker()
            prev_task_mode = (kvs_constants.ETL_TASK_INITIALIZE, None)
            self.logger.info("Recovery ETL_TASK_INITIALIZE completed on worker{}.".format(self.worker_id))

        while True:
            if prev_task_mode == (task, mode):
                # no new tasks
                time.sleep(0.5)
            else:
                if task == kvs_constants.ETL_TASK_INITIALIZE:
                    self.logger.info("Beginning task ETL_TASK_INITIALIZE on worker{}".format(self.worker_id))
                    self.initialize_worker()
                    self.logger.info("ETL_TASK_INITIALIZE completed on worker{}.".format(self.worker_id))
                elif task == kvs_constants.ETL_TASK_PREPROCESS:
                    self.run_data_parallel(mode)
                    self.logger.info("ETL_TASK_PREPROCESS for {} data completed on worker{}.".format(
                        mode, self.worker_id))
                elif task == kvs_constants.ETL_TASK_LOAD_PROCESSED:
                    if not self.params:
                        self.initialize_worker()
                    self.download_processed_data(mode)
                    self.logger.info("ETL_TASK_LOAD_PROCESSED for {} data completed on worker {}.".format(
                        mode, self.worker_id))
                elif task == kvs_constants.ETL_TASK_SAVE_PROCESSED:
                    self.upload_processed_data(mode)
                    self.logger.info("ETL_TASK_SAVE_PROCESSED for {} data completed on worker{}.".format(
                        mode, self.worker_id))
                elif task == kvs_constants.PROGRESS_COMPLETE:
                    done = True
                    self.kvs.etl_set_worker_progress(self.worker_id, 0)
                    self.logger.info("ETL tasks complete on worker{}".format(self.worker_id))

                # mark task as complete in worker
                self.kvs.etl_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)

            # check for errors:
            err = self.kvs.get_error()
            if err:
                # mark as done, wait for restart
                done = True
                self.logger.error("Caught error in ETL Worker {}, waiting for restart".format(self.worker_id))
                self.logger.error(str(err))

            # update prev task and poll KVS for next tasks
            prev_task_mode = (task, mode)
            if done:
                # wait for ETL Controller to close workers
                time.sleep(1)
            else:
                task, mode = self.kvs.etl_get_task()


def main():
    worker = ETLWorker()
    worker.serve_forever()


if __name__ == '__main__':
    main()
