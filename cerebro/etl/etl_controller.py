import os
import gc
import sys
import json
import time
import pandas as pd
from pathlib import Path
from kubernetes import client, config
from tqdm.notebook import tqdm_notebook

from cerebro.util.params import Params
from cerebro.util.alerts import html_alert
import cerebro.kvs.constants as kvs_constants
from cerebro.util.voyager_io import VoyagerIO
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.util.cerebro_logger import CerebroLogger

class ETLController:
    logging = CerebroLogger("controller")
    logger = logging.create_logger("etl")

    def __init__(self):
        self.metadata_df = None
        self.etl_spec = None
        self.fraction = None

        self.task_descriptions = {
            kvs_constants.ETL_TASK_PREPROCESS: "Preprocess Data",
            kvs_constants.ETL_TASK_LOAD_PROCESSED: "Download Processed Data",
            kvs_constants.ETL_TASK_SAVE_PROCESSED: "Upload Processed Data"
        }

        # initialize Params
        self.params = Params()

        # create Key Value Store handle and in initialize tables
        self.kvs = KeyValueStore()
        self.logger.info("Created key value store handle")

        # reset all errors
        self.kvs.set_error("")

        # load values from cerebro-info configmap
        self.namespace = os.environ['NAMESPACE']
        self.username = os.environ['USERNAME']
        config.load_kube_config()
        v1 = client.CoreV1Api()
        cm = v1.read_namespaced_config_map(name='{}-cerebro-info'.format(self.username), namespace=self.namespace)
        cerebro_info = json.loads(cm.data["data"])
        self.num_workers = cerebro_info["num_workers"]
        user_code_path = cerebro_info["user_code_path"]
        self.user_ids = (cerebro_info["uid"], cerebro_info["gid"])

        # add user repo dir to sys path for library discovery
        sys.path.insert(0, user_code_path)

        # get node info
        cm = v1.read_namespaced_config_map(name='cerebro-node-hardware-info', namespace=self.namespace)
        self.node_info = json.loads(cm.data["data"])

        # initialize node info
        self.gpu_counts = [0 for _ in range(self.num_workers)]
        self.total_gpus = 0
        for node_id in range(self.num_workers):
            self.gpu_counts[node_id] = self.node_info["num_gpus"]
            self.total_gpus += self.gpu_counts[node_id]

    def initialize_controller(self, etl_spec, fraction):
        self.etl_spec = etl_spec
        self.fraction = fraction

        if self.etl_spec:
            # save functions on KVS
            self.kvs.etl_set_spec(self.etl_spec)

            # initialize etl workers
            self.initialize_etl_workers()

            # process misc. files
            self.etl_spec.read_misc(self.params.miscellaneous["output_path"])
        else:
            self.kvs.etl_set_spec("")

    def initialize_etl_workers(self):
        # Initialize ETL Workers
        for worker_id in range(self.num_workers):
            self.kvs.etl_set_worker_status(worker_id, kvs_constants.IN_PROGRESS)

        # scale up ETL workers
        self.scale_workers(self.num_workers)

        # mark as initializing
        self.kvs.etl_set_task(kvs_constants.ETL_TASK_INITIALIZE, "")
        self.logger.info("Marked ETL state as initializing workers")

        # wait for initialization to complete on all nodes
        done = False
        while not done:
            completed = [self.kvs.etl_get_worker_status(w) == kvs_constants.PROGRESS_COMPLETE for w in range(self.num_workers)]
            if all(completed):
                done = True
                break
            else:
                time.sleep(1)
        print("Initialized all ETL workers")
        self.logger.info("Initialized all ETL workers")

    def scale_workers(self, num_workers):
        config.load_kube_config()
        app_name = "cerebro-etl-worker"

        def scale_complete():
            if num_workers == 0:
                v1 = client.AppsV1Api()
                current_replicas = v1.read_namespaced_stateful_set(name="f{self.username}-{app_name}",
                                                                   namespace=self.namespace).spec.replicas
                return current_replicas == num_workers
            else:
                # check if pods are in "Running" state
                v1 = client.CoreV1Api()
                pods = v1.list_namespaced_pod(namespace=self.namespace,
                                              label_selector=f"app={app_name},user={self.username}").items
                ready_count = 0
                for pod in pods:
                    if pod.status.phase == "Running":
                        ready_count += 1
                return ready_count == num_workers


        # scale up ETL workers
        print("Scaling ETL workers to {}".format(num_workers))
        v1 = client.AppsV1Api()
        statefulset = v1.read_namespaced_stateful_set(name=f"{self.username}-{app_name}", namespace=self.namespace)
        statefulset.spec.replicas = num_workers
        v1.replace_namespaced_stateful_set(name=f"{self.username}-{app_name}", namespace=self.namespace, body=statefulset)

        # wait for desired number of workers
        wait_time = 0
        while not scale_complete():
            time.sleep(0.5)
            wait_time += 0.5
            if wait_time >= 100:
                raise Exception("Unable to schedule ETL Workers on Voyager")

    def download_metadata(self, mode=None):
        if not mode:
            remote_urls = [
                self.params.etl["train"]["metadata_url"],
                self.params.etl["val"]["metadata_url"],
                self.params.etl["test"]["metadata_url"],
                self.params.etl["predict"]["metadata_url"],
            ]
        else:
            remote_urls = [self.params.etl[mode]["metadata_url"]]

        for remote_url in remote_urls:
            # download the metadata files from remote
            if self.params.etl["download_type"] == "url":
                file_io = VoyagerIO()
                exclude_prefix = remote_url
                prefix = remote_url
                output_path = self.params.etl[mode]["metadata_download_path"]
                Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
                file_io.download(output_path, prefix, exclude_prefix)
                self.logger.info("Downloaded {} metadata".format(mode if mode else "all"))

    def download_processed_val_data(self):
        self.logger.info("Beginning download of processed ETL val data")
        str_task = self.task_descriptions[kvs_constants.ETL_TASK_LOAD_PROCESSED]
        desc = "{} Progress".format(str_task + " " + str.capitalize("val"))
        val_progress = tqdm_notebook(total=100, desc=desc, position=0, leave=True)
        file_io = VoyagerIO()

        # download val data from Ceph
        exclude_prefix = os.path.join(self.params.etl["etl_dir"], "val")
        prefix = os.path.join(exclude_prefix, "val_data.pkl")
        Path(self.params.etl["val"]["output_path"]).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(self.params.etl["val"]["output_path"], "val_data.pkl")
        file_io.download(output_path, prefix, exclude_prefix)
        val_progress.update(100)
        self.logger.info("Completed download of val data from Ceph on controller")

    def upload_processed_val_data(self):
        self.logger.info("Beginning upload of processed ETL val data")
        str_task = self.task_descriptions[kvs_constants.ETL_TASK_SAVE_PROCESSED]
        desc = "{} Progress".format(str_task + " " + str.capitalize("val"))
        val_progress = tqdm_notebook(total=100, desc=desc, position=0, leave=True)
        file_io = VoyagerIO()

        remote_path = os.path.join(self.params.etl["etl_dir"], "val")
        local_path = self.params.etl["val"]["output_path"]
        exclude_prefix = local_path
        file_io.upload(local_path, remote_path, exclude_prefix)
        val_progress.update(100)
        self.logger.info("Completed upload of val data to destination from Controller")

    def shard_data(self, mode):
        # load seed value
        seed = self.kvs.get_seed()

        # load metadata
        complete_metadata_df = pd.read_csv(self.params.etl[mode]["metadata_download_path"])

        # shuffle data for train and validation modes only
        if mode in ["train", "val"]:
            self.metadata_df = complete_metadata_df.sample(frac=self.fraction, random_state=seed)
        else:
            num_rows = int(complete_metadata_df.shape[0] * self.fraction)
            self.metadata_df = complete_metadata_df.head(num_rows)

        self.logger.info("Loaded {} metadata".format(mode))

        # partition size is proportional to num_gpus on each node
        n = self.num_workers
        partition_size_ratios = [0.0 for _ in range(n)]
        for nid in range(n):
            partition_size_ratios[nid] = self.gpu_counts[nid] / self.total_gpus

        cumulative_ratio = 0.0
        nrows = len(self.metadata_df.index)
        en_idx = 0
        for nid in range(n):
            st_idx = en_idx
            if nid == (n - 1):
                partition = self.metadata_df.iloc[st_idx:, :]
            else:
                cumulative_ratio += partition_size_ratios[nid]
                en_idx = int(cumulative_ratio * nrows)
                partition = self.metadata_df.iloc[st_idx: en_idx, :]

            partition_path = self.params.etl[mode]["partition_path"]
            Path(partition_path).mkdir(parents=True, exist_ok=True)
            partition.to_csv(os.path.join(partition_path, "worker" + str(nid) + ".csv"), index=False)

            self.logger.info("Saved {} data partition shard of node {}".format(mode, nid))

        self.logger.info("Sharding of {} complete with {} shards".format(mode, n))

    def process_task(self, task, mode):
        str_task = self.task_descriptions[task]

        # mark all ETL worker status as in-progress and set progress to 0
        for worker_id in range(self.num_workers):
            self.kvs.etl_set_worker_progress(worker_id, 0)
            self.kvs.etl_set_worker_status(worker_id, kvs_constants.IN_PROGRESS)

        # mark kvs as beginning task
        self.kvs.etl_set_task(task, mode)
        self.logger.info("Beginning ETL task {} in mode {}".format(str_task.lower(), mode))

        desc = "{} Progress".format(str_task + " " + str.capitalize(mode))
        progress = tqdm_notebook(total=100, desc=desc, position=0, leave=True)
        while True:
            # check for errors and raise alert
            err = self.kvs.get_error()
            if err:
                self.logger.error("Notified of error in Controller, exiting")
                self.logger.error(str(err))

                # scale down ETL workers to 0
                self.scale_workers(0)

                html_alert(err)
                return

            total = 0.0
            completed = []
            for worker_id in range(self.num_workers):
                total += self.kvs.etl_get_worker_progress(worker_id)
                completed.append(self.kvs.etl_get_worker_status(worker_id) == kvs_constants.PROGRESS_COMPLETE)
            percentage = round(total / self.num_workers, 2)

            if all(completed):
                progress.update(100 - progress.n)
                progress.close()
                break
            else:
                progress.update(percentage - progress.n)

        self.logger.info("ETL task {} in mode {} complete".format(str_task.lower(), mode))

    def combine_val_pkl(self):
        # combine all Pickle files into a single file
        mode = "val"
        output_path = self.params.etl[mode]["output_path"]
        self.logger.info("Coalescing {} dataset shards to a single file...".format(mode))
        all_files = [os.path.abspath(os.path.join(output_path, f))
                     for f in os.listdir(output_path)]
        pickle_files = [file for file in all_files if file.endswith('.pkl')]
        all_df = []
        for f in pickle_files:
            df = pd.read_pickle(f)
            all_df.append(df)
        combined_df = pd.concat(all_df)
        combined_df.to_pickle(os.path.join(output_path, "{}_data.pkl".format(mode)))

        # delete all individual Pickle files
        for file in pickle_files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting {file}: {e}")

    def run_etl(self):
        # check which tasks are given in the dataset locators
        etl_dir_present = self.params.etl["etl_dir"] is not None
        etl_mode_present = {
            "train": self.params.etl["train"]["metadata_url"] and self.params.etl["train"]["multimedia_url"],
            "val": self.params.etl["val"]["metadata_url"] and self.params.etl["val"]["multimedia_url"],
            "test": self.params.etl["test"]["metadata_url"] and self.params.etl["test"]["multimedia_url"],
            "predict": self.params.etl["predict"]["metadata_url"] and self.params.etl["predict"]["multimedia_url"]
        }

        # download, shard and process data for modes based on their mode_present value
        for mode, mode_present in etl_mode_present.items():
            if mode_present:
                self.download_metadata(mode=mode)
                self.shard_data(mode=mode)
                self.process_task(kvs_constants.ETL_TASK_PREPROCESS, mode)

                # combine all pickle files only for validation dataset
                if mode == "val":
                    # combining happens in workers for other modes
                    self.combine_val_pkl()

        # if only etl_dir is given then download processed data
        # if both etl_dir and train/val/test/predict tasks are
        # given - upload processed data, change ownership of dir
        if etl_dir_present:
            if not etl_mode_present["train"]:
                self.process_task(kvs_constants.ETL_TASK_LOAD_PROCESSED, "train")
            else:
                self.process_task(kvs_constants.ETL_TASK_SAVE_PROCESSED, "train")
            # handle val data separately
            if not etl_mode_present["val"]:
                self.download_processed_val_data()
            else:
                self.upload_processed_val_data()
            if not etl_mode_present["test"]:
                self.process_task(kvs_constants.ETL_TASK_LOAD_PROCESSED, "test")
            else:
                self.process_task(kvs_constants.ETL_TASK_SAVE_PROCESSED, "test")
            if not etl_mode_present["predict"]:
                self.process_task(kvs_constants.ETL_TASK_LOAD_PROCESSED, "predict")
            else:
                self.process_task(kvs_constants.ETL_TASK_SAVE_PROCESSED, "predict")

            # update ownership of etl_dir from root to user
            if etl_dir_present:
                uid, gid = self.user_ids
                ownership_cmd = "chown -R {uid}:{gid} {dir_path}".format(
                    uid=uid, gid=gid, dir_path=self.params.etl["etl_dir"]
                )
                os.system(ownership_cmd)

    def exit_etl(self):
        # idle all ETL Workers
        self.kvs.etl_set_task(kvs_constants.PROGRESS_COMPLETE, "")

        # scale down workers
        self.scale_workers(0)

        # clean up
        del self.metadata_df
        gc.collect()
        self.logger.info("Cleaned up ETL controller and worker memory")
        self.logger.info("Closed ETL")
