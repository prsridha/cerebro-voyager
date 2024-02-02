import os
import time
import json
import pprint
import random
import hashlib
import itertools
import pandas as pd
from pathlib import Path
from copy import deepcopy
from pprint import pprint
from datetime import datetime
from kubernetes import client, config
from tqdm.notebook import tqdm_notebook
from IPython.display import display, FileLink

from cerebro.util.params import Params
import cerebro.kvs.constants as kvs_constants
from cerebro.util.voyager_io import VoyagerIO
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.util.cerebro_logger import CerebroLogger
from cerebro.parallelisms.parallelism_init import PARALLELISMS_LIST

import torch

class MOPController:
    logging = CerebroLogger("controller")
    logger = logging.create_logger("mop")

    def __init__(self):
        self.num_models = 0
        self.params = Params()
        self.num_epochs = None
        self.param_grid = None
        self.minibatch_spec = None

        # init mop states
        self.msts = []
        self.mw_pair = []
        self.model_on_worker = {}
        self.worker_running_model = {}
        self.model_nworkers_trained = {}

        # load values from cerebro-info configmap
        self.namespace = os.environ['NAMESPACE']
        self.username = os.environ['USERNAME']
        config.load_kube_config()
        v1 = client.CoreV1Api()
        cm1 = v1.read_namespaced_config_map(name='{}-cerebro-info'.format(self.username), namespace=self.namespace)
        cerebro_info = json.loads(cm1.data["data"])
        self.num_workers = cerebro_info["num_workers"]
        self.user_ids = (cerebro_info["uid"], cerebro_info["gid"])

        # save sub-epoch func in KVS
        self.kvs = KeyValueStore()

        # create Metrics directories
        for mode in ["train", "val"]:
            for directory in self.params.mop["metrics_storage_path"]:
                full_path = os.path.join(directory, mode)
                Path(full_path).mkdir(parents=True, exist_ok=True)

    def initialize_controller(self, minibatch_spec, num_epochs, param_grid):
        self.num_epochs = num_epochs
        self.param_grid = param_grid
        self.minibatch_spec = minibatch_spec

        # process misc. files
        self.minibatch_spec.read_misc(self.params.miscellaneous["output_path"])
        self.logger.info("read_misc function call complete in MOP Controller")

        self.kvs.mop_set_spec(minibatch_spec)
        self.logger.info("Saved MOP spec on KeyValueStore")

        # scale MOP workers
        scale_status = self.scale_workers(self.num_workers)
        id_str = kvs_constants.MOP_TASK_INITIALIZE
        task_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()
        for w in range(self.num_workers):
            self.kvs.mop_set_task(kvs_constants.MOP_TASK_INITIALIZE, w, task_id)
            self.logger.info("Initialized MOP workers")

    def scale_workers(self, num_workers):
        config.load_kube_config()
        app_name = "cerebro-mop-worker"

        def scale_complete():
            if num_workers == 0:
                v1 = client.AppsV1Api()
                current_replicas = v1.read_namespaced_stateful_set(name=f"{self.username}-{app_name}",
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


        # scale up MOP workers
        v1 = client.AppsV1Api()
        statefulset = v1.read_namespaced_stateful_set(name=f"{self.username}-{app_name}", namespace=self.namespace)
        statefulset.spec.replicas = num_workers
        v1.replace_namespaced_stateful_set(name=f"{self.username}-{app_name}", namespace=self.namespace, body=statefulset)

        # wait for desired number of workers
        wait_time = 0
        while not scale_complete():
            time.sleep(0.5)
            wait_time += 0.5
            if wait_time >= 250:
                raise Exception("Unable to schedule MOP Workers on Voyager")
        return True

    def save_artifacts(self):
        dt_version = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        base_dir = os.path.join(self.params.mop["output_dir"], "artifact_" + dt_version)

        # save checkpoints
        progress = tqdm_notebook(total=3, desc="Save Metrics", position=0, leave=True)
        file_io = VoyagerIO()

        chckpt_dir = os.path.join(base_dir, "checkpoints")
        Path(chckpt_dir).mkdir(parents=True, exist_ok=True)
        from_path = self.params.mop["checkpoint_storage_path"]
        file_io.upload(from_path, chckpt_dir, exclude_prefix=from_path)
        progress.update(1)
        print("Saved model checkpoints")

        # save user metrics
        metrics_dir = os.path.join(base_dir, "metrics")
        Path(metrics_dir).mkdir(parents=True, exist_ok=True)
        from_path = os.path.join(self.params.mop["metrics_storage_path"]["tensorboard"])
        exclude_prefix = os.path.dirname(from_path)
        file_io.upload(from_path, metrics_dir, exclude_prefix=exclude_prefix)
        progress.update(1)
        from_path = os.path.join(self.params.mop["metrics_storage_path"]["user_metrics"])
        file_io.upload(from_path, metrics_dir, exclude_prefix=exclude_prefix)
        progress.update(1)
        print("Saved model building metrics")

        # close progress bar
        progress.close()

        # change ownership to user
        uid, gid = self.user_ids
        ownership_cmd = "chown -R {uid}:{gid} {dir_path}".format(
            uid=uid, gid=gid, dir_path=base_dir
        )
        os.system(ownership_cmd)

        # print the saved dir location
        print_dir = os.path.join(os.path.normpath(base_dir), "artifact_" + dt_version)
        print("Saved experiment artifacts at {}".format(print_dir))

    def download_models(self):
        # if models_dir is defined and checkpoint dir is empty - then download models from remote
        checkpoint_path = self.params.mop["checkpoint_storage_path"]
        if self.params.mop["models_dir"] and len(os.listdir(checkpoint_path)) == 0:
            file_io = VoyagerIO()
            files = file_io.list_files(self.params.mop["models_dir"])
            download_progress = tqdm_notebook(total=len(files), desc="Downloading Models", position=0, leave=True)

            for f in files:
                file_io.download(checkpoint_path, f, self.params.mop["models_dir"])
                download_progress.update(1)

            download_progress.close()

    def get_runnable_model(self, worker_id):
        # get seed value
        seed = self.kvs.get_seed()

        runnable_model = -1
        models_list = list(range(self.num_models))
        random.Random(seed).shuffle(models_list)

        for m in models_list:
            if not (self.mw_pair[m][worker_id]):
                if self.model_on_worker[m] == -1:
                    runnable_model = m
                    break
        return runnable_model

    def find_combinations(self):
        param_grid = self.param_grid
        param_keys = list(param_grid.keys())
        params_list = [param_grid[key] for key in param_keys]
        combinations = list(itertools.product(*params_list))

        param_combinations = []
        for comb in combinations:
            d = {}
            for i in range(len(comb)):
                d[param_keys[i]] = comb[i]
            param_combinations.append(d)
        self.num_models = len(param_combinations)

        # save combinations to KVS
        param_combinations_indexed = dict(enumerate(param_combinations))
        self.kvs.mop_set_model_mapping(param_combinations_indexed)
        combinations_filepath = os.path.join(self.params.mop["checkpoint_storage_path"], "model_search_space.json")
        Path(self.params.mop["checkpoint_storage_path"]).mkdir(parents=True, exist_ok=True)
        with open(combinations_filepath, "w") as f:
            json.dump(param_combinations_indexed, f, indent=4, sort_keys=True)

        return param_combinations

    def create_model_components(self):
        for model_id, hyperparams in enumerate(self.msts):
            model_object = self.minibatch_spec.create_model_components(hyperparams)
            model_object_path = os.path.join(self.params.mop["checkpoint_storage_path"], "model_" + str(model_id),
                                             "model_object_{}.pt".format(str(model_id)))
            Path(os.path.dirname(model_object_path)).mkdir(parents=True, exist_ok=True)
            torch.save(model_object, model_object_path)
            self.logger.info("Created model components for model {}".format(model_id))

    def skipped_sampler(self):
        # DDP is the only available parallelism
        best_parallelism = "DDP"
        msts_parallelisms = deepcopy(self.msts)
        for m in range(self.num_models):
            msts_parallelisms[m]["parallelism"] = best_parallelism
            self.kvs.mop_set_parallelism_mapping(m, best_parallelism)

        self.logger.info("Completed Trial Runs")

        # print model selection search space
        self.logger.info("List of models and their hyperparameter configurations:")
        print("List of models and their hyperparameter configurations:")
        config_df = pd.DataFrame(msts_parallelisms).drop(columns="parallelism")
        config_df = config_df.rename_axis('model_id').reset_index()
        display(config_df)
        self.logger.info(str(config_df.to_dict()))

    def sampler(self):
        mpls = set()
        mpl_on_worker = {}
        n_workers = self.num_workers

        # generate model-parallelism combinations
        for p in PARALLELISMS_LIST:
            for m in range(self.num_models):
                mpls.add((m, p))

        mpls_to_schedule = set(deepcopy(mpls))
        remaining_mpls = set(deepcopy(mpls))

        for w in range(n_workers):
            mpl_on_worker[w] = None

        sampling_progress = tqdm_notebook(total=len(remaining_mpls), desc='Trial Runs', position=0, leave=True)
        while remaining_mpls:
            for w in range(n_workers):
                if mpls_to_schedule and not mpl_on_worker[w]:
                    # schedule new trial run
                    mpl = mpls_to_schedule.pop()
                    mpl_on_worker[w] = mpl
                    model_id, parallelism = mpl

                    # mark worker as busy
                    self.kvs.mop_set_worker_status(w, kvs_constants.IN_PROGRESS)
                    self.kvs.mop_set_model_parallelism_on_worker(w, model_id, parallelism)

                    # set worker task
                    id_str = "-".join([str(i) for i in (kvs_constants.MOP_TASK_TRIALS, w, model_id, parallelism)])
                    task_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()
                    self.kvs.mop_set_task(kvs_constants.MOP_TASK_TRIALS, w, task_id)

                    self.logger.info("Scheduled trial run of parallelism {} on worker {}".format(mpl, w))
                else:
                    # check if worker status is "complete"
                    mpl = mpl_on_worker[w]
                    status = self.kvs.mop_get_worker_status(w)
                    if status == kvs_constants.PROGRESS_COMPLETE and mpl:
                        remaining_mpls.remove(mpl)
                        sampling_progress.update(1)
                        mpl_on_worker[w] = None
            time.sleep(1)

        # save the best parallelism for each model
        msts_parallelisms = deepcopy(self.msts)
        for m in range(self.num_models):
            makespans = {p: self.kvs.mop_get_sample_time(m, p) for p in PARALLELISMS_LIST}
            best_parallelism = min(makespans, key=makespans.get)
            msts_parallelisms[m]["parallelism"] = best_parallelism
            self.kvs.mop_set_parallelism_mapping(m, best_parallelism)

        self.logger.info("Completed Trial Runs")

        # print model selection search space
        self.logger.info("List of models and their chosen parallelisms:")
        print("List of models and their chosen parallelisms:")
        config_df = pd.DataFrame(msts_parallelisms)
        config_df = config_df.rename_axis('model_id').reset_index()
        display(config_df)
        self.logger.info(str(config_df.to_dict()))

    def init_epoch(self):
        n_workers = self.num_workers
        msts = deepcopy(self.msts)

        # initialize all data structures
        self.mw_pair = []
        self.model_on_worker = {}
        self.model_nworkers_trained = {}
        self.worker_running_model = [-1] * n_workers

        for m in range(self.num_models):
            self.model_on_worker[m] = -1
            self.model_nworkers_trained[m] = 0

        for _ in range(self.num_models):
            lis = []
            for _ in range(n_workers):
                lis.append(False)
            self.mw_pair.append(lis)

        s = "Model ID: Model msts\n"
        for i in range(self.num_models):
            s += str(i) + " : " + pprint.pformat(msts[i]) + "\n"
        self.logger.info("Initial model configurations:")
        self.logger.info(s)

    def scheduler(self, epoch):
        n_workers = self.num_workers
        models_to_build = set(range(self.num_models))

        model_progresses = {m: tqdm_notebook(total=n_workers, desc='Model ' + str(m), position=0, leave=False) for m in range(len(models_to_build))}

        while models_to_build:
            for worker_id in range(n_workers):
                if self.worker_running_model[worker_id] == -1:
                    model_id = self.get_runnable_model(worker_id)
                    if model_id != -1:
                        is_last_worker = self.model_nworkers_trained[model_id] == n_workers - 1

                        # update KVS
                        self.kvs.mop_set_model_on_worker(worker_id, epoch, model_id, is_last_worker)
                        self.kvs.mop_set_worker_status(worker_id, kvs_constants.IN_PROGRESS)

                        # set worker task
                        id_str = "-".join([str(i) for i in (kvs_constants.MOP_TASK_TRAIN_VAL, worker_id, epoch, model_id, is_last_worker)])
                        task_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()
                        self.kvs.mop_set_task(kvs_constants.MOP_TASK_TRAIN_VAL, worker_id, task_id)

                        self.model_on_worker[model_id] = worker_id
                        self.worker_running_model[worker_id] = model_id

                        self.logger.info("Sent model {} to build on worker {} with config {}".format(
                            str(model_id), str(worker_id), str(self.msts[model_id])))
                else:
                    # poll since this particular worker is busy
                    model_id = self.worker_running_model[worker_id]
                    if model_id != -1:
                        status = self.kvs.mop_get_worker_status(worker_id)
                        completed = status == kvs_constants.PROGRESS_COMPLETE

                        if completed:
                            self.logger.info("Received Model {} built on worker {}".format(str(model_id), str(worker_id)))
                            # models[m].n = status["result"]
                            self.model_on_worker[model_id] = -1
                            self.worker_running_model[worker_id] = -1
                            self.model_nworkers_trained[model_id] += 1
                            self.mw_pair[model_id][worker_id] = True
                            model_done = True
                            for i in range(n_workers):
                                if not self.mw_pair[model_id][i]:
                                    model_done = False
                                    break
                            if model_done:
                                models_to_build.remove(model_id)

                            # update model progress bar
                            update_val = self.model_nworkers_trained[model_id] - model_progresses[model_id].n
                            model_progresses[model_id].update(update_val)
                            self.logger.info("Model:" + str(model_id) + " trained on " + str(self.model_nworkers_trained[model_id]) + "/" + str(n_workers))

        self.logger.info("Ending epoch...")

        # close progress bars
        for m in model_progresses:
            model_progresses[m].close()

    def testing(self, model_tag, batch_size):
        output_dir = self.params.mop["test_output_path"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # update KVS
        self.kvs.mop_set_test_params(model_tag, batch_size)

        # set worker task
        id_str = "-".join([str(i) for i in (kvs_constants.MOP_TASK_TEST, model_tag, batch_size)])
        task_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()

        # run test on all workers
        for worker_id in range(self.num_workers):
            self.kvs.mop_set_worker_status(worker_id, kvs_constants.IN_PROGRESS)
            self.kvs.mop_set_task(kvs_constants.MOP_TASK_TEST, worker_id, task_id)

        # update completion progress
        progress = tqdm_notebook(total=self.num_workers, desc="Testing Progress", position=0, leave=True)
        while True:
            all_complete = []
            for w in range(self.num_workers):
                status = self.kvs.mop_get_worker_status(w)
                completed = status == kvs_constants.PROGRESS_COMPLETE
                all_complete.append(completed)
            progress.update(sum(all_complete) - progress.n)
            time.sleep(1)
            if all(all_complete):
                progress.close()
                break

        # aggregate results
        agg_df = pd.DataFrame()
        for worker_id in range(self.num_workers):
            output_path = os.path.join(self.params.mop["test_output_path"], f"test_output_{Path(model_tag).stem}_{worker_id}.csv")
            df = pd.read_csv(output_path, header=0)
            agg_df = pd.concat([agg_df, df], ignore_index=True)
        reduced_df = agg_df.mean().to_frame()

        # display metrics on the notebook
        display(reduced_df)

        return True

    def prediction(self, model_tag, batch_size):
        output_dir = self.params.mop["predict_output_path"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # update KVS
        self.kvs.mop_set_predict_params(model_tag, batch_size)

        # set worker task
        id_str = "-".join([str(i) for i in (kvs_constants.MOP_TASK_TEST, model_tag, batch_size)])
        task_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()

        # set worker task
        for worker_id in range(self.num_workers):
            self.kvs.mop_set_worker_status(worker_id, kvs_constants.IN_PROGRESS)
            self.kvs.mop_set_task(kvs_constants.MOP_TASK_PREDICT, worker_id, task_id)

        # run test on all workers
        for worker_id in range(self.num_workers):
            self.kvs.mop_set_worker_status(worker_id, kvs_constants.IN_PROGRESS)

        # update completion progress
        progress = tqdm_notebook(total=self.num_workers, desc="Inference Progress", position=0, leave=True)
        while True:
            all_complete = []
            for w in range(self.num_workers):
                status = self.kvs.mop_get_worker_status(w)
                completed = status == kvs_constants.PROGRESS_COMPLETE
                all_complete.append(completed)
            progress.update(sum(all_complete) - progress.n)
            time.sleep(1)
            if all(all_complete):
                progress.close()
                break

        # combine inference output files from all workers
        combined_df = pd.DataFrame()
        model_tag_stem = str(Path(model_tag).stem)
        combined_output_file = f"predict_output_{model_tag_stem}.csv"
        predict_output_path = self.params.mop["predict_output_path"]
        for worker_id in range(self.num_workers):
            predict_output_worker_file = os.path.join(predict_output_path,
                                                     f"predict_output_{model_tag_stem}_{worker_id}.csv")
            df = pd.read_csv(predict_output_worker_file, header=0)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            os.remove(predict_output_worker_file)

        # sort based on row_id
        combined_df = combined_df.sort_values("row_id")
        combined_df.to_csv(combined_output_file, index=False)

        # display download link to combined file
        display(FileLink(combined_output_file))

        return True

    def grid_search(self):
        num_epochs = self.num_epochs
        self.msts = self.find_combinations()

        # create models and optimizers
        print("Creating models")
        self.create_model_components()

        # skipping Sampling as only DDP is enabled
        # print("Running parallelism sampling...")
        self.skipped_sampler()

        print("\n")
        print("You can monitor your models on the Tensorboard Dashboard")
        print("Beginning model scheduling \n")
        self.logger.info("Beginning model scheduling")
        epoch_progress = tqdm_notebook(total=self.num_epochs, desc='Epochs', position=0, leave=True)
        print("\n")

        for epoch in range(1, num_epochs + 1):
            self.logger.info("EPOCH: {}".format(epoch))
            self.init_epoch()
            self.scheduler(epoch)
            epoch_progress.update(1)
