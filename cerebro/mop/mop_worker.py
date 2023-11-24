import os
import gc
import sys
import json
import time
import random
import argparse
import threading
from pathlib import Path
from kubernetes import client, config
from xmlrpc.server import SimpleXMLRPCServer

from cerebro.util.params import Params
import cerebro.kvs.constants as kvs_constants
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.util.cerebro_logger import CerebroLogger
from cerebro.util.coalesce_dataset import CoalesceDataset
from cerebro.parallelisms.parallelism_init import get_parallelism_executor

from torch.utils.data import Subset

class CerebroWorker:
    def __init__(self, worker_id):
        logging = CerebroLogger("worker-{}".format(worker_id))
        self.logger = logging.create_logger("mop-worker")
        self.user_logger = logging.create_logger("mop-user")

        self.kvs = KeyValueStore()
        self.params = None
        self.sub_epoch_spec = None
        self.worker_id = worker_id

        # load values from cerebro-info configmap
        # config.load_incluster_config()
        config.load_kube_config()
        v1 = client.CoreV1Api()
        namespace = os.environ['NAMESPACE']
        cm = v1.read_namespaced_config_map(name='cerebro-info', namespace=namespace)
        cm_data = json.loads(cm.data["data"])
        hostname = "0.0.0.0"
        port = cm_data["worker_rpc_port"]
        user_repo_path = cm_data["user_repo_path"]
        self.sample_size = cm_data["sample_size"]
        self.metrics_cycle_size = cm_data["metrics_cycle_size"]

        self.server = SimpleXMLRPCServer((hostname, port), allow_none=True, logRequests=False)
        self.server.register_function(self.initialize_worker)
        self.server.register_function(self.test_model_on_worker)
        self.server.register_function(self.train_model_on_worker)
        self.server.register_function(self.sample_parallelism_on_worker)

        # add user repo dir to sys path for library discovery
        sys.path.insert(0, user_repo_path)

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

    def server_forever(self):
        state = self.kvs.mop_get_task()
        if state != "" and self.kvs.mop_get_worker_status(self.worker_id) == kvs_constants.IN_PROGRESS:
            # restarted
            self.logger.info("Worker{} recovery detected in MOP".format(self.worker_id))
            self.kvs.set_restarts(self.worker_id)

            self.initialize_worker()

        if state == kvs_constants.MOP_TASK_TRIALS:
            model_id, parallelism_name = self.kvs.mop_get_model_parallelism_on_worker(self.worker_id)
            self.sample_parallelism_on_worker(model_id, parallelism_name)
        elif state == kvs_constants.MOP_TASK_TRAIN_VAL:
            d = self.kvs.mop_get_models_on_worker(self.worker_id)
            self.train_model_on_worker(d["model_id"], d["epoch"], d["is_last_worker"])
        elif state == kvs_constants.MOP_TASK_TEST:
            pass

        self.logger.info("Started serving forever...")
        print("Started serving forever...")
        self.server.serve_forever()

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

        # call the train function
        start = time.time()
        parallelism.execute_sample(self.sub_epoch_spec.train, sampled_dataset)
        end = time.time()

        time_elapsed = end - start
        self.kvs.mop_set_sample_time(model_id, parallelism_name, time_elapsed)
        self.logger.info("completed parallelism sampling of model {} on worker {}".format(model_id, self.worker_id))
        print("completed parallelism sampling of model {} on worker {}".format(model_id, self.worker_id))

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
        parallelism.execute_train(self.sub_epoch_spec.train, self.sub_epoch_spec.metrics_agg, dataset, self.metrics_cycle_size, model_id)

        if is_last_worker:
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


def main():
    parser = argparse.ArgumentParser(
        description='Argument parser for generating model predictions.')
    parser.add_argument('--id', help='Worker ID', default="0", type=str)
    args = parser.parse_args()
    worker_id = int(args.id.split("-")[-1])

    worker = CerebroWorker(worker_id)
    worker.server_forever()


if __name__ == '__main__':
    main()
