import os
import time
import json
import pprint
import random
import itertools
import pandas as pd
from pathlib import Path
from copy import deepcopy
import xmlrpc.client as xc
from datetime import datetime
from IPython.display import display
from kubernetes import client, config
from tqdm.notebook import tqdm_notebook

from cerebro.util.s3_io import S3IO
from cerebro.util.params import Params
import cerebro.kvs.constants as kvs_constants
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.util.cerebro_logger import CerebroLogger
from cerebro.parallelisms.parallelism_init import PARALLELISMS_LIST

import torch

class MOPController:
    logging = CerebroLogger("controller")
    logger = logging.create_logger("mop")

    def __init__(self):
        self.num_models = 0
        self.worker_names = []
        self.params = Params()
        self.num_epochs = None
        self.param_grid = None
        self.sub_epoch_spec = None

        # init mop states
        self.msts = []
        self.mw_pair = []
        self.model_on_worker = {}
        self.worker_running_model = {}
        self.model_nworkers_trained = {}

        # load values from cerebro-info and cerebro-node-hardware-info configmaps
        # config.load_incluster_config()
        config.load_kube_config()
        v1 = client.CoreV1Api()
        namespace = os.environ['NAMESPACE']
        username = os.environ['USERNAME']
        cm = v1.read_namespaced_config_map(name='{}-cerebro-info'.format(username), namespace=namespace)
        cm1_data = json.loads(cm1.data["data"])
        username = cm1_data["username"]
        rpc_port = cm1_data["worker_rpc_port"]

        cm2 = v1.read_namespaced_config_map(name='cerebro-node-hardware-info', namespace=namespace)
        self.num_nodes = len(json.loads(cm2.data["data"]))

        # save sub-epoch func in KVS
        self.kvs = KeyValueStore(init_tables=True)

        # create Metrics directories
        for mode in ["train", "val"]:
            for directory in self.params.mop["metrics_storage_path"]:
                full_path = os.path.join(directory, mode)
                Path(full_path).mkdir(parents=True, exist_ok=True)

        # get MOP workers
        for i in range(self.num_nodes):
            host_args = {"username": username, "pod_id": str(i), "namespace": namespace}
            host = "http://{username}-cerebro-worker-{pod_id}.{username}-workersvc.{namespace}.svc.cluster.local".format(**host_args)
            self.worker_names.append(host + ":" + str(rpc_port))
        self.workers = {i: xc.ServerProxy(ip) for i, ip in enumerate(self.worker_names)}

    def initialize_controller(self, num_epochs, param_grid, sub_epoch_spec):
        self.num_epochs = num_epochs
        self.param_grid = param_grid
        self.sub_epoch_spec = sub_epoch_spec

        if sub_epoch_spec:
            self.kvs.mop_set_spec(sub_epoch_spec)
            self.logger.info("Saved MOP spec on KeyValueStore")

            # initilize MOP workers
            for w, worker in self.workers.items():
                try:
                    self.kvs.mop_set_task(kvs_constants.MOP_TASK_INITIALIZE)
                    worker.initialize_worker()
                except Exception as e:
                    self.logger.error(
                        "Unable to reach MOP worker {} during worker initialization. Error: {}".format(w, str(e)))
                    print("Unable to reach MOP worker {} during worker initialization. Error: {}".format(w, str(e)))

            self.logger.info("Initialized MOP workers")
        else:
            self.kvs.mop_set_spec("")

    def save_metrics_s3(self):
        dt_version = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        base_dir = os.path.join(self.params.mop["output_dir"], "artifact_" + dt_version)
        print("Created save directory in S3")

        # save checkpoints
        progress = tqdm_notebook(total=100, desc="Save Metrics", position=0, leave=True)
        s3io = S3IO(self.params.bucket_name, progress.update)

        chckpt_dir = os.path.join(base_dir, "checkpoints")
        Path(chckpt_dir).mkdir(parents=True, exist_ok=True)
        from_path = self.params.mop["checkpoint_storage_path"]
        s3io.upload(from_path, chckpt_dir)
        print("Saved model checkpoints")

        # save user metrics
        metrics_dir = os.path.join(base_dir, "metrics")
        Path(metrics_dir).mkdir(parents=True, exist_ok=True)
        from_path = os.path.join(self.params.mop["metrics_storage_path"]["tensorboard"])
        s3io.upload(from_path, metrics_dir)
        from_path = os.path.join(self.params.mop["metrics_storage_path"]["user_metrics"])
        s3io.upload(from_path, metrics_dir)
        print("Saved model building metrics")

        print("Saved metrics to S3 at {}".format(base_dir))

    def download_models(self):
        if self.params.mop["models_dir"]:
            downloaded_models_path = self.params.mop["checkpoint_storage_path"]
            s3io = S3IO(self.params.bucket_name, None)
            files = s3io.list_files(self.params.mop["models_dir"])
            download_progress = tqdm_notebook(total=len(files), desc="Download Models", position=0, leave=True)

            for f in files:
                prefix = os.path.join(self.params.mop["models_dir"], f)
                download_path = os.path.join(downloaded_models_path, f.split(".")[0])
                Path(os.path.dirname(download_path)).mkdir(parents=True, exist_ok=True)
                s3io.download(download_path, prefix, self.params.mop["models_dir"])
                download_progress.update(1)

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
            model_object = self.sub_epoch_spec.create_model_components(hyperparams)
            model_object_path = os.path.join(self.params.mop["checkpoint_storage_path"], "model_" + str(model_id),
                                             "model_object_{}.pt".format(str(model_id)))
            Path(os.path.dirname(model_object_path)).mkdir(parents=True, exist_ok=True)
            torch.save(model_object, model_object_path)
            self.logger.info("Created model components for model {}".format(model_id))

    def sampler(self):
        mpls = set()
        mpl_on_worker = {}
        n_workers = self.num_nodes

        # generate model-parallelism combinations
        for p in PARALLELISMS_LIST:
            for m in range(self.num_models):
                mpls.add((m, p))

        mpls_to_schedule = set(deepcopy(mpls))
        remaining_mpls = set(deepcopy(mpls))
        self.kvs.mop_set_task(kvs_constants.MOP_TASK_TRIALS)

        for w in range(n_workers):
            mpl_on_worker[w] = None

        sampling_progress = tqdm_notebook(total=len(remaining_mpls), desc='Trial Runs', position=0, leave=True)
        while remaining_mpls:
            for w in range(n_workers):
                if mpls_to_schedule and not mpl_on_worker[w]:
                    # schedule new trial run
                    mpl = mpls_to_schedule.pop()
                    mpl_on_worker[w] = mpl
                    self.kvs.mop_set_model_parallelism_on_worker(w, mpl)
                    try:
                        # mark worker as busy
                        self.kvs.mop_set_worker_status(w, kvs_constants.IN_PROGRESS)
                        model_id, parallelism = mpl
                        self.workers[w].sample_parallelism_on_worker(model_id, parallelism)
                        self.logger.info("Scheduled trial run of parallelism {} on worker {}".format(mpl, w))
                        # print("Scheduled trial run of parallelism {} on worker {}".format(mpl, w))
                    except Exception as e:
                        self.logger.error("Failed to schedule trial run of parallelism {} on worker {}. Error {}".format(mpl, w, str(e)))
                        print("Failed to schedule trial run of parallelism {} on worker {}. Error {}".format(mpl, w, str(e)))
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

        self.logger.info("List of models and their chosen parallelisms:")
        print("List of models and their chosen parallelisms:")

        # print model selection search space
        config_df = pd.DataFrame(msts_parallelisms)
        config_df = config_df.rename_axis('model_id').reset_index()
        display(config_df)

        self.logger.info(str(config_df.to_dict()))

    def init_epoch(self):
        n_workers = self.num_nodes
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
        n_workers = self.num_nodes
        models_to_build = set(range(self.num_models))

        self.kvs.mop_set_task(kvs_constants.MOP_TASK_TRAIN_VAL)
        self.logger.info("Beginning model scheduling...")
        self.logger.info("Starting epoch...")

        model_progresses = {m: tqdm_notebook(total=n_workers, desc='Model ' + str(m), position=0, leave=False) for m in range(len(models_to_build))}

        while models_to_build:
            for worker_id in range(n_workers):
                if self.worker_running_model[worker_id] == -1:
                    model_id = self.get_runnable_model(worker_id)
                    if model_id != -1:
                        is_last_worker = self.model_nworkers_trained[model_id] == n_workers - 1

                        # update KVS and launch job
                        self.kvs.mop_set_model_on_worker(worker_id, epoch, model_id, is_last_worker)
                        self.kvs.mop_set_worker_status(worker_id, kvs_constants.IN_PROGRESS)
                        try:
                            self.workers[worker_id].train_model_on_worker(model_id, epoch, is_last_worker)
                        except Exception as e:
                            self.logger.error("Failed to schedule train of model {} on worker {}. Error {}".format(model_id, worker_id, str(e)))
                            print("Failed to schedule train of model {} on worker {}. Error {}".format(model_id, worker_id, str(e)))

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

                            # print("Model:" + str(model_id) + " trained on " + str(self.model_nworkers_trained[model_id]) + "/" + str(n_workers))
                            self.logger.info("Model:" + str(model_id) + " trained on " + str(self.model_nworkers_trained[model_id]) + "/" + str(n_workers))
        self.logger.info("Ending epoch...")

        # close progress bars
        for m in model_progresses:
            model_progresses[m].close()

    def testing(self, model_tag, batch_size, output_filename):
        output_dir = self.params.mop["test_output_path"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # set worker task
        self.kvs.mop_set_task(kvs_constants.MOP_TASK_TEST)

        # run test on all workers
        for worker_id in range(self.num_nodes):
            self.kvs.mop_set_worker_status(worker_id, kvs_constants.IN_PROGRESS)
            try:
                self.workers[worker_id].test_model_on_worker(model_tag, batch_size)
            except Exception as e:
                self.logger.error("Failed to schedule test of model {} on worker {}. Error {}".format(model_tag, worker_id, str(e)))
                print("Failed to schedule test of model {} on worker {}. Error {}".format(model_tag, worker_id, str(e)))

        # update completion progress
        progress = tqdm_notebook(total=100, desc="Testing Progress", position=0, leave=False)
        while True:
            all_complete = []
            for w in range(self.num_nodes):
                status = self.kvs.mop_get_worker_status(w)
                completed = status == kvs_constants.PROGRESS_COMPLETE
                if completed:
                    progress.update(1)
                all_complete.append(completed)
                time.sleep(1)
            if all(all_complete):
                break

        # combine output files and save
        combined_filename = os.path.join(output_dir, output_filename)
        combined_df = pd.concat([pd.read_json(file) for file in os.listdir(output_dir) if file.startswith("test_output_")], ignore_index=True)
        mean_df = combined_df.mean().to_frame().T
        mean_df.to_json(combined_filename, orient='records')

        return True

    def prediction(self, model_tag, batch_size, output_filename):
        output_dir = self.params.mop["prediction_output_path"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # set worker task
        self.kvs.mop_set_task(kvs_constants.MOP_TASK_TEST)

        # run test on all workers
        for worker_id in range(self.num_nodes):
            self.kvs.mop_set_worker_status(worker_id, kvs_constants.IN_PROGRESS)
            try:
                self.workers[worker_id].test_model_on_worker(model_tag, batch_size)
            except Exception as e:
                self.logger.error(
                    "Failed to schedule prediction of model {} on worker {}. Error {}".format(model_tag, worker_id, str(e)))
                print(
                    "Failed to schedule prediction of model {} on worker {}. Error {}".format(model_tag, worker_id, str(e)))

        # update completion progress
        progress = tqdm_notebook(total=100, desc="Inference Progress", position=0, leave=False)
        while True:
            all_complete = []
            for w in range(self.num_nodes):
                status = self.kvs.mop_get_worker_status(w)
                completed = status == kvs_constants.PROGRESS_COMPLETE
                if completed:
                    progress.update(1)
                    self.logger.info("Completed prediction on epoch...")
                all_complete.append(completed)
                time.sleep(1)
            if all(all_complete):
                break

        # combine output files
        combined_filename = os.path.join(output_dir, output_filename)
        combined_data = pd.concat(
            (pd.read_csv(file) for file in sorted(os.listdir(output_dir)) if file.startswith("prediction_output_")),
            ignore_index=True)
        combined_data.to_csv(combined_filename, index=False)

        return True

    def grid_search(self):
        num_epochs = self.num_epochs
        self.msts = self.find_combinations()

        # create models and optimizers
        print("Creating models...")
        self.create_model_components()

        # run makespan sampling
        print("Running parallelism sampling...")
        self.sampler()

        print("\n")
        epoch_progress = tqdm_notebook(total=self.num_epochs, desc='Epochs', position=0, leave=True)
        print("Beginning model scheduling...")
        print("You can monitor your models on the Tensorboard Dashboard")
        for i in range(num_epochs):
            print("EPOCH: {}".format(i + 1))
            self.logger.info("EPOCH: {}".format(i + 1))
            self.init_epoch()
            self.scheduler(i)
            epoch_progress.update(1)
