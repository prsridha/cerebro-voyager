import os
import gc
import dill
import json
import time
import base64
import random
import threading
from pathlib import Path

import pandas as pd
from kubernetes import client, config

from cerebro.util.params import Params
import cerebro.kvs.constants as kvs_constants
from cerebro.util.save_metrics import SaveMetrics
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.util.cerebro_logger import CerebroLogger
from cerebro.util.coalesce_dataset import CoalesceDataset
from cerebro.parallelisms.parallelism_init import get_parallelism_executor

from torch.utils.data import Subset

class CerebroWorker:
    def __init__(self):
        # obtain worker_id from env variable
        worker_name = os.environ.get("ORDINAL_ID")
        worker_id = int(worker_name.split("-")[-1])

        logging = CerebroLogger("worker-{}".format(worker_id))
        self.logger = logging.create_logger("mop-worker")
        self.logger.info("Starting MOP worker {}".format(worker_id))

        self.params = None
        self.sub_epoch_spec = None
        self.worker_id = worker_id
        self.kvs = KeyValueStore()

        # load values from cerebro-info configmap
        namespace = os.environ['NAMESPACE']
        username = os.environ['USERNAME']
        config.load_kube_config()
        v1 = client.CoreV1Api()
        cm = v1.read_namespaced_config_map(name='{}-cerebro-info'.format(username), namespace=namespace)
        cm_data = json.loads(cm.data["data"])
        self.sample_size = cm_data["sample_size"]

        # add user repo dir to sys path for library discovery
        # sys.path.insert(0, user_code_path)

        self.logger.info('Starting Cerebro worker {}'.format(worker_id))
        print('Starting Cerebro worker {}'.format(worker_id))

    def initialize_worker(self):
        self.params = Params()

        # create necessary paths
        for mode in ["train", "val"]:
            metrics_dir = os.path.join(self.params.mop["metrics_storage_path"]["user_metrics"], mode)
            Path(metrics_dir).mkdir(parents=True, exist_ok=True)

        # read subepoch spec and call initialize worker
        self.logger.info("Initializing worker packages")
        self.sub_epoch_spec = self.kvs.mop_get_spec()
        self.sub_epoch_spec.initialize_worker()
        self.logger.info("Subepoch init worker called")

    def sample_parallelism(self, ParallelismExecutor, model_id, model_config, parallelism_name):
        # obtain seed value
        seed = self.kvs.get_seed()

        # obtain paths
        sample_data_path = os.path.join(self.params.etl["train"]["output_path"], "train_data{}.pkl".format(self.worker_id))

        # sample 'sample_size' percent of examples from the dataset
        dataset = CoalesceDataset(sample_data_path)
        num_samples = int(len(dataset) * self.sample_size)
        sampled_indices = random.Random(seed).sample(range(len(dataset)), num_samples)
        sampled_dataset = Subset(dataset, sampled_indices)

        # create parallelism object of given Parallelism class
        model_checkpoint_path = os.path.join(self.params.mop["checkpoint_storage_path"], "model_" + str(model_id),
                                             "model_object_{}.pt".format(model_id))
        parallelism = ParallelismExecutor(self.worker_id, model_config, model_checkpoint_path)

        self.logger.info("Sampling model {} with parallelism {} on worker {}".format(model_id, parallelism_name, self.worker_id))

        # call the train function
        start = time.time()
        parallelism.execute_sample(self.sub_epoch_spec.train, sampled_dataset)
        end = time.time()

        time_elapsed = end - start
        self.kvs.mop_set_sample_time(model_id, parallelism_name, time_elapsed)
        self.logger.info("Completed parallelism sampling of model {} with parallelism {} on worker {}".format(model_id, parallelism_name, self.worker_id))

        # set worker status as complete
        self.kvs.mop_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)

    def train_model(self, ParallelismExecutor, epoch, model_id, model_config, is_last_worker):
        print("training model {} on worker {}".format(model_id, self.worker_id))
        self.logger.info("training model {} on worker {}".format(model_id, self.worker_id))

        # create dataset object
        train_data_path = os.path.join(self.params.etl["train"]["output_path"], "train_data{}.pkl".format(self.worker_id))
        dataset = CoalesceDataset(train_data_path)

        # define necessary paths and create parallelism object
        model_checkpoint_path = os.path.join(self.params.mop["checkpoint_storage_path"], "model_" + str(model_id),
                                             "model_object_{}.pt".format(model_id))
        parallelism = ParallelismExecutor(self.worker_id, model_config, model_checkpoint_path)

        # call the train function
        parallelism.execute_train(dataset, model_id)

        if is_last_worker:
            # aggregate and plot train epoch metrics
            user_metrics_func = dill.loads(base64.b64decode(self.sub_epoch_spec.metrics_agg))
            csv_path = os.path.join(self.params.mop["metrics_storage_path"]["user_metrics"], "train",
                                    "{}.csv".format(model_id))
            metrics_df = pd.read_csv(csv_path)
            reduced_df = user_metrics_func("train", model_config, metrics_df)
            SaveMetrics.save_to_tensorboard(reduced_df, "train_epoch", model_id, epoch)

            # run validation
            self.validate_model(parallelism, model_id, epoch)
            self.logger.info("completed validation of model {} on worker {}".format(model_id, self.worker_id))
            print("completed validation of model {} on worker {}".format(model_id, self.worker_id))

        # set worker status as complete
        self.kvs.mop_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)

    def validate_model(self, parallelism, model_id, epoch):
        # create dataset object
        val_data_path = os.path.join(self.params.etl["val"]["output_path"], "val_data.pkl")
        dataset = CoalesceDataset(val_data_path)

        # run validation via parallelism
        parallelism.execute_val(self.sub_epoch_spec.val_test, self.sub_epoch_spec.metrics_agg, dataset, model_id, epoch)

    def test_model(self, ParallelismExecutor, model_tag, batch_size):
        # create dataset object
        test_data_path = os.path.join(self.params.etl["test"]["output_path"], "test_data{}.pkl".format(self.worker_id))
        dataset = CoalesceDataset(test_data_path)

        # get model checkpoint path
        if model_tag.isdigit():
            model_checkpoint_path = os.path.join(self.params.mop["checkpoint_storage_path"], "model_" + str(model_tag),
                                                 "model_object_{}.pt".format(model_tag))
        else:
            model_tag_dir = model_tag.split(".")[0]
            model_checkpoint_path = os.path.join(self.params.mop["checkpoint_storage_path"], model_tag_dir, model_tag)

        # create parallelism object
        model_config = {"batch_size": batch_size}
        parallelism = ParallelismExecutor(self.worker_id, model_config, model_checkpoint_path)

        # run test via parallelism
        output_path = os.path.join(self.params.mop["test_output_path"])
        Path(os.path.dirname(output_path)).mkdir(exist_ok=True)
        parallelism.execute_test(self.sub_epoch_spec.val_test, dataset, output_path)

        # set worker status as complete
        self.kvs.mop_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)

    def sample_parallelism_on_worker(self, model_id, parallelism_name):
        # attempting garbage collection of previous threads
        gc.collect()

        # get the best parallelism class for this model
        ParallelismExecutor = get_parallelism_executor(parallelism_name)

        # get model hyperparameters
        model_config = self.kvs.mop_get_model_mapping(model_id)
        thread = threading.Thread(target=self.sample_parallelism, args=(ParallelismExecutor, model_id, model_config, parallelism_name))
        thread.start()

        self.logger.info("Thread started in sample_parallelism_on_worker")
        print("Thread started in sample_parallelism_on_worker")

        print("trial run of model {} on worker {} for parallelism {}".format(model_id, self.worker_id, parallelism_name))
        self.logger.info("trial run of model {} on worker {} for parallelism {}".format(model_id, self.worker_id, parallelism_name))

    def train_model_on_worker(self, model_id, epoch, is_last_worker):
        # attempting garbage collection of previous threads
        gc.collect()

        # get the best parallelism class for this model
        p_name = self.kvs.mop_get_parallelism_mapping(model_id)
        ParallelismExecutor = get_parallelism_executor(p_name)

        # get model hyperparameters
        model_config = self.kvs.mop_get_model_mapping(model_id)
        thread = threading.Thread(target=self.train_model, args=(ParallelismExecutor, epoch, model_id, model_config, is_last_worker))
        thread.start()

        self.logger.info("Thread started in train_model_on_worker")
        print("Thread started in train_model_on_worker")

    def test_model_on_worker(self, model_tag, batch_size):
        # attempting garbage collection of previous threads
        gc.collect()

        # default parallelism to FSDP
        p_name = "FSDP"
        ParallelismExecutor = get_parallelism_executor(p_name)

        # begin test
        thread = threading.Thread(target=self.test_model, args=(ParallelismExecutor, model_tag, batch_size))
        thread.start()

        self.logger.info("Thread started in test_model_on_worker")
        print("Thread started in test_model_on_worker")

    def server_forever(self):
        done = False
        prev_task, prev_task_id = None, None
        task, task_id = self.kvs.mop_get_task(self.worker_id)
        if task and task != kvs_constants.MOP_TASK_INITIALIZE:
            # recovered
            self.logger.info("Worker{} recovery detected in MOP".format(self.worker_id))
            self.kvs.set_restarts(self.worker_id)

            self.initialize_worker()
            prev_task = kvs_constants.MOP_TASK_INITIALIZE
            self.logger.info("Recovery MOP_TASK_INITIALIZE completed on worker{}.".format(self.worker_id))

        while True:
            if (prev_task, prev_task_id) == (task, task_id):
                # no new tasks
                time.sleep(0.5)
            else:
                if task == kvs_constants.MOP_TASK_TRIALS:
                    model_id, parallelism_name = self.kvs.mop_get_model_parallelism_on_worker(self.worker_id)
                    self.sample_parallelism_on_worker(model_id, parallelism_name)
                elif task == kvs_constants.MOP_TASK_TRAIN_VAL:
                    d = self.kvs.mop_get_models_on_worker(self.worker_id)
                    self.train_model_on_worker(d["model_id"], d["epoch"], d["is_last_worker"])
                elif task == kvs_constants.MOP_TASK_TEST:
                    pass
                elif task == kvs_constants.MOP_TASK_PREDICT:
                    pass
                elif task == kvs_constants.PROGRESS_COMPLETE:
                    done = True
                    self.logger.info("MOP tasks complete on worker{}".format(self.worker_id))

                # mark task as complete in worker
                self.kvs.mop_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)

            # check for errors:
            err = self.kvs.get_error()
            if err:
                # mark as done, wait for restart
                done = True
                self.logger.error("Caught error in MOP Worker {}, waiting for restart".format(self.worker_id))
                self.logger.error(str(err))

            # update prev task and poll KVS for next tasks
            prev_task, prev_task_id = task, task_id
            if done:
                # wait for MOP Controller to close workers
                time.sleep(1)
            else:
                task, task_id = self.kvs.mop_get_task(self.worker_id)

def main():
    worker = CerebroWorker()
    worker.server_forever()


if __name__ == '__main__':
    main()
