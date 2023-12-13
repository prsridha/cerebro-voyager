import os
import gc
import sys
import time
import json
import uuid
import argparse
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
    def __init__(self, worker_id, mode, shard_multiplicity, params, etl_spec, is_feature_download, progress_queue):
        logging = CerebroLogger("worker-{}".format(worker_id))
        self.logger = logging.create_logger("etl-worker-process")

        self.mode = mode
        self.params = params
        self.etl_spec = etl_spec
        self.worker_id = worker_id
        self.queue = progress_queue
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
                        kvs.set_error(str(e))
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
                kvs.set_error(str(e))
                return

            # compute conditional values
            is_multiple = (row_count > 0) and (row_count % m_factor == 0)
            is_last_row = row_count == num_shard_rows - 1
            is_last_sub_shard = row_count / m_factor == self.shard_multiplicity

            # push progress to queue at every 10th row
            if row_count % 10 == 0 or is_last_row:
                progress = (row_count + 1) / num_shard_rows
                progress_data = {"process_id": process_id, "progress": progress}
                self.queue.put(progress_data)

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
    def __init__(self, worker_id):
        logging = CerebroLogger("worker-{}".format(worker_id))
        self.logger = logging.create_logger("etl-worker")

        self.file_io = None
        self.params = None
        self.shards = None
        self.etl_spec = None
        self.metadata_df = None
        self.progress_queue = None
        self.worker_id = worker_id
        self.is_feature_download = None

        # load values from cerebro-info configmap
        namespace = os.environ['NAMESPACE']
        username = os.environ['USERNAME']
        config.load_kube_config()
        v1 = client.CoreV1Api()
        cm = v1.read_namespaced_config_map(name='{}-cerebro-info'.format(username), namespace=namespace)
        cm_data = json.loads(cm.data["data"])
        user_code_path = cm_data["user_code_path"]
        self.shard_multiplicity = cm_data["shard_multiplicity"]

        # get node info
        cm = v1.read_namespaced_config_map(name='cerebro-node-hardware-info', namespace=namespace)
        node_info = json.loads(cm.data["data"])
        self.num_gpus = node_info["node" + str(worker_id)]["num_gpus"]

        # add user repo dir to sys path for library discovery
        sys.path.insert(0, user_code_path)

        # boost low vcpu nodes
        cores = int(os.cpu_count())
        # self.num_process = cores
        self.num_process = cores if cores >= 48 else 48

        # initialize necessary handlers
        self.kvs = KeyValueStore(init_tables=True)
        self.p = ProcessPool(self.num_process)

        self.logger.info("Starting Cerebro worker {}".format(worker_id))
        print("Starting Cerebro worker {}".format(worker_id))

    def initialize_worker(self):
        # initialize params and get features to download
        self.params = Params()

        # create S3 I/O object for S3 reads and writes
        update_progress_fn = partial(self.kvs.etl_set_worker_progress, self.worker_id)

        self.file_io = VoyagerIO(update_progress_fn)

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
        progress_data = {}
        manager = multiprocessing.Manager()
        self.progress_queue = manager.Queue()
        proc = EtlProcess(self.worker_id, mode, self.shard_multiplicity, self.params,
                          self.etl_spec, self.is_feature_download, self.progress_queue)

        self.p.restart()
        self.p.imap(proc.process_data, self.shards)
        self.logger.info("Started row processing on all {} cores".format(self.num_process))

        # track progress and update to KVS
        done = 0
        while done < self.num_process:
            err = self.kvs.get_error()
            if err:
                # exit with an error code
                self.logger.error("Caught error in Worker, exiting")
                self.logger.error(str(err))

            progress = self.progress_queue.get()
            process_id = progress['process_id']
            progress_value = progress['progress']
            if process_id not in progress_data:
                progress_data[process_id] = 0
            progress_data[process_id] = progress_value
            percentage = (sum(progress_data.values()) / self.num_process) * 100
            self.kvs.etl_set_worker_progress(self.worker_id, percentage)
            if progress_value == 1.0:
                done += 1

        self.p.close()
        self.p.join()

        # combine all Pickle files into a single file for train, test and predict modes
        if mode != "val":
            self.logger.info("Coalescing {} dataset shards to a single file...".format(mode))
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
        self.logger.info("Beginning upload of processed ETL data")
        prefix = os.path.join(self.params.etl["etl_dir"], mode)
        output_path = self.params.etl[mode]["output_path"]
        self.file_io.upload(output_path, prefix)
        self.logger.info("Completed upload of {} data to destination from worker {}".format(mode, self.worker_id))

    def download_processed_data(self, mode):
        self.logger.info("Beginning download of processed ETL data")
        exclude_prefix = os.path.join(self.params.etl["etl_dir"], mode)
        prefix = os.path.join(self.params.etl["etl_dir"], mode, "{}_data{}.pkl".format(mode, self.worker_id))
        Path(self.params.etl[mode]["output_path"]).mkdir(parents=True, exist_ok=True)
        output_path = self.params.etl[mode]["output_path"]
        self.file_io.download(output_path, prefix, exclude_prefix)
        self.logger.info("Completed download of {} data from destination on worker {}".format(mode, self.worker_id))

    def serve_forever(self):
        self.logger.info("Started serving forever...")
        print("Started serving forever...")

        done = False
        worker_idling = False
        prev_task_mode = (None, None)
        task, mode = self.kvs.etl_get_task()
        if task:
            # recovered
            self.logger.info("Worker{} restart detected in ETL".format(self.worker_id))
            self.kvs.set_restarts(self.worker_id)

            # ETL is idling after restart
            if task == kvs_constants.PROGRESS_COMPLETE:
                worker_idling = True

            self.initialize_worker()
            prev_task_mode = (kvs_constants.ETL_TASK_INITIALIZE, None)
            self.kvs.etl_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)
            self.logger.info("Recovery ETL_TASK_INITIALIZE completed on worker{}.".format(self.worker_id))

        while not done:
            if prev_task_mode == (task, mode):
                # no new tasks
                time.sleep(0.5)
            else:
                if task == kvs_constants.ETL_TASK_INITIALIZE:
                    worker_idling = False
                    self.logger.info("Beginning task ETL_TASK_INITIALIZE on worker{}".format(self.worker_id))
                    self.initialize_worker()
                    self.logger.info("ETL_TASK_INITIALIZE completed on worker{}.".format(self.worker_id))
                elif task == kvs_constants.ETL_TASK_PREPROCESS:
                    self.run_data_parallel(mode)
                    self.logger.info("ETL_TASK_PREPROCESS for {} data completed on worker{}.".format(
                        mode, self.worker_id))
                elif task == kvs_constants.ETL_TASK_LOAD_PROCESSED:
                    if not self.params or self.file_io:
                        self.initialize_worker()
                    self.download_processed_data(mode)
                    self.logger.info("ETL_TASK_LOAD_PROCESSED for {} data completed on worker {}.".format(
                        mode, self.worker_id))
                elif task == kvs_constants.ETL_TASK_SAVE_PROCESSED:
                    self.upload_processed_data(mode)
                    self.logger.info("ETL_TASK_SAVE_PROCESSED for {} data completed on worker{}.".format(
                        mode, self.worker_id))
                elif task == kvs_constants.PROGRESS_COMPLETE and not worker_idling:
                    done = True
                    self.kvs.etl_set_worker_progress(self.worker_id, 0)
                    self.logger.info("ETL Complete. Restarting worker{} for future ETL tasks.".format(self.worker_id))

                # mark task as complete in worker
                self.kvs.etl_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)

            # poll KVS for task
            if not done:
                prev_task_mode = (task, mode)
                task, mode = self.kvs.etl_get_task()

            # check for errors:
            err = self.kvs.get_error()
            if err:
                # exit with an error code
                self.logger.error("Caught error in Worker, exiting")
                self.logger.error(str(err))


def main():
    parser = argparse.ArgumentParser(
        description='Argument parser for generating model predictions.')
    parser.add_argument('--id', help='Worker ID', default="0", type=str)
    args = parser.parse_args()
    worker_id = int(args.id.split("-")[-1])

    worker = ETLWorker(worker_id)
    worker.serve_forever()


if __name__ == '__main__':
    main()
