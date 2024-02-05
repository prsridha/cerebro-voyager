import os
import gc
import time
import dill
import base64
import random
import warnings
import pandas as pd
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from cerebro.util.coalesce_dataset import CoalesceDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import habana_frameworks.torch.hpu as hthpu
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.distributed.hccl

from cerebro.util.params import Params
from cerebro.util.save_metrics import SaveMetrics
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.util.cerebro_logger import CerebroLogger
from cerebro.parallelisms.parallelism_spec import Parallelism

# initialize device to Habana HPU
device = torch.device('hpu')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)
    warnings.filterwarnings("ignore")

    # initialize the process group
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

def clean_up():
    dist.destroy_process_group()

kvs = KeyValueStore()

class DDPExecutor(Parallelism):
    def __init__(self, worker_id, model_config, model_checkpoint_path, epoch, seed, sample_size=None):
        super().__init__(worker_id, model_config, model_checkpoint_path, epoch, seed, sample_size)
        self.name = "DDPExecutor"
        logging = CerebroLogger("worker-{}".format(worker_id))
        self.logger = logging.create_logger("ddp-worker")

        self.mode = None
        self.epoch = epoch
        self.model_id = None
        self.model_tag = None
        self.seed = kvs.get_seed()
        self.worker_id = worker_id
        self.sample_size = sample_size
        self.hyperparams = model_config
        self.world_size = hthpu.device_count()
        self.model_path = model_checkpoint_path

        # get output path
        params = Params()
        self.etl_output_path = {
            "sample": os.path.join(params.etl["train"]["output_path"], "train_data{}.pkl".format(self.worker_id)),
            "train": os.path.join(params.etl["train"]["output_path"], "train_data{}.pkl".format(self.worker_id)),
            "val": os.path.join(params.etl["val"]["output_path"], "val_data.pkl"),
            "test": os.path.join(params.etl["test"]["output_path"], "test_data{}.pkl".format(self.worker_id)),
            "predict": os.path.join(params.etl["predict"]["output_path"], "predict_data{}.pkl".format(self.worker_id))
        }
        self.predict_output_path = params.mop["predict_output_path"]

        # create state dict path
        self.state_dict_path = os.path.join(os.path.dirname(self.model_path), "state_dicts")
        Path(self.state_dict_path).mkdir(parents=True, exist_ok=True)
        self.logger.info("Initialized DDP Executor with model ")

    def save_local_metrics(self, rank, metrics_list, user_metrics_func):
        # group-by on key
        df = pd.DataFrame(metrics_list)
        grouped_metrics = df.to_dict(orient='list')

        # convert to tensors
        for key in grouped_metrics:
            grouped_metrics[key] = torch.tensor(grouped_metrics[key]).to(device)

        # all-reduce
        for key, tensor in grouped_metrics.items():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # get average
        for key in grouped_metrics:
            grouped_metrics[key] /= self.world_size

        if rank == 0:
            # plot only train and val metrics on tensorboard
            if self.mode == "train":
                reduced_metrics = {key: tensor.item() for key, tensor in grouped_metrics.items()}
                SaveMetrics.save_to_tensorboard(reduced_metrics, self.mode, self.model_id, self.epoch)
                SaveMetrics.save_to_file(reduced_metrics, self.mode, "{}.csv".format(self.model_id))
                self.logger.info(f"Saved model {self.model_id}'s train metrics to file and tensorboard")
            elif self.mode == "val":
                reduced_metrics = {key: tensor.tolist() for key, tensor in grouped_metrics.items()}
                result = user_metrics_func(self.mode, self.hyperparams, reduced_metrics)
                SaveMetrics.save_to_tensorboard(result, self.mode, self.model_id, self.epoch)
                SaveMetrics.save_to_file(result, self.mode, "{}.csv".format(self.model_id))
                self.logger.info(f"Saved model {self.model_id}'s val metrics to tensorboard")
            elif self.mode == "test":
                reduced_metrics = {key: tensor.tolist() for key, tensor in grouped_metrics.items()}
                result = user_metrics_func(self.mode, self.hyperparams, reduced_metrics)
                output_filename = f"test_output_{self.model_tag}_{self.worker_id}.csv"
                SaveMetrics.save_to_file(result, self.mode, output_filename)
                self.logger.info(f"Saved model {self.model_id}'s test metrics to file")

    def load_checkpoint(self, model_object):
        # check if checkpoints exist
        if len(os.listdir(self.state_dict_path)) == 0:
            return model_object

        new_model_object = {}
        for obj_name, obj in model_object.items():
            state_dict_file_path = os.path.join(self.state_dict_path, obj_name + ".pt")
            model_checkpoint = torch.load(state_dict_file_path)
            if obj_name != "optimizer":
                obj.to(device)
            obj.load_state_dict(model_checkpoint)
            new_model_object[obj_name] = obj

        return new_model_object

    def save_checkpoint(self, model_object):
        # use a barrier to make sure task is done on all ranks
        dist.barrier()
        for obj_name, obj in model_object.items():
            if dist.get_rank() == 0:
                state_dict_file_path = os.path.join(self.state_dict_path, obj_name + ".pt")
                if obj_name == "optimizer":
                    torch.save(obj.state_dict(), state_dict_file_path)
                else:
                    torch.save(obj.module.state_dict(), state_dict_file_path)

    def _execute_inner(self, rank, user_func_str, user_metrics_func_str=None):
        # initialize process with this rank
        setup(rank, self.world_size)

        # load user_func from serialized str
        user_func = dill.loads(base64.b64decode(user_func_str))
        user_metrics_func = dill.loads(base64.b64decode(user_metrics_func_str)) if user_metrics_func_str else None

        # get dataset
        data_path = self.etl_output_path[self.mode]
        dataset = CoalesceDataset(data_path)
        if self.mode == "sample":
            num_samples = int(len(dataset) * self.sample_size)
            sampled_indices = random.Random(self.seed).sample(range(len(dataset)), num_samples)
            dataset = Subset(dataset, sampled_indices)

        # create data sampler and loader
        sampler = DistributedSampler(dataset, rank=rank, num_replicas=self.world_size, shuffle=False, seed=self.seed)
        dataloader = DataLoader(dataset, batch_size=self.hyperparams["batch_size"], sampler=sampler, shuffle=False)

        # load models and their state dicts
        model_object = torch.load(self.model_path)
        updated_obj = self.load_checkpoint(model_object)

        # parallelize models from model object file
        for obj_name, obj in updated_obj.items():
            if obj_name != "optimizer":
                obj.to(device)
                p_model = DDP(obj)
                updated_obj[obj_name] = p_model

        # call user's ML function
        if self.mode == "sample":
            for k, minibatch in enumerate(dataloader):
                user_func(updated_obj, minibatch, self.hyperparams, device)

        elif self.mode == "train":
            num_iterations = len(dataloader)
            if rank == 0:
                self.logger.info("Running user's train function with DDP on a minibatch")
            for k, minibatch in enumerate(dataloader):
                if rank == 0:
                    print(f"Iteration {k+1}/{num_iterations}")
                updated_obj, metrics = user_func(updated_obj, minibatch, self.hyperparams, device)
                self.save_local_metrics(rank, [metrics], user_metrics_func)

            # save checkpoint of uploaded model object
            self.save_checkpoint(updated_obj)
            if rank == 0:
                self.logger.info("Completed user's train function with DDP on a minibatch. Metrics and checkpoint saved")

        elif self.mode == "val":
            minibatch_metrics = []
            for k, minibatch in enumerate(dataloader):
                metrics = user_func(updated_obj, minibatch, self.hyperparams, device)
                minibatch_metrics.append(metrics)

            self.save_local_metrics(rank, minibatch_metrics, user_metrics_func)

        elif self.mode == "test":
            test_outputs = []
            subepoch_size = len(dataloader)
            for k, minibatch in enumerate(dataloader):
                output = user_func(updated_obj, minibatch, self.hyperparams, device)
                test_outputs.append(output)

                # compute completed percentage and update progress on rank 0
                if rank == 0:
                    percentage = k / subepoch_size * 100
                    print(f"COMPLETED {k}, {percentage}")
                    kvs.mop_set_worker_progress(self.worker_id, percentage)

            self.save_local_metrics(rank, test_outputs, user_metrics_func)

        elif self.mode == "predict":
            predict_outputs = []
            subepoch_size = len(dataloader)
            for k, minibatch in enumerate(dataloader):
                outputs, probabilities = user_func(updated_obj, minibatch, self.hyperparams, device)
                row_ids = minibatch[1].tolist()
                mapped_classes = [(row_ids[i], outputs[i], probabilities[i]) for i in range(len(row_ids))]
                predict_outputs.extend(mapped_classes)

                # compute completed percentage and update progress on rank 0
                if rank == 0:
                    percentage = k / subepoch_size * 100
                    kvs.mop_set_worker_progress(self.worker_id, percentage)

            # save inference outputs of each rank
            output_filename = os.path.join(self.predict_output_path, f"predict_output_{self.model_tag}_{self.worker_id}_{rank}.csv")
            output_df = pd.DataFrame(predict_outputs, columns=["row_id", "output_class", "confidence"])
            output_df.to_csv(output_filename, index=False)

        print("Completed DDP event")

        # clean up resources
        del sampler
        del dataloader
        del updated_obj
        del model_object
        gc.collect()
        clean_up()

    def execute_sample(self, minibatch_spec):
        # set values
        self.mode = "sample"
        user_train_func_str = base64.b64encode(dill.dumps(minibatch_spec))

        # spawn DDP workers
        time.sleep(5)
        mp.spawn(self._execute_inner, args=(user_train_func_str,), nprocs=self.world_size, join=True)

    def execute_train(self, minibatch_spec, model_id):
        # get train and metrics_agg functions
        self.mode = "train"
        self.model_id = model_id
        user_train_func, user_metrics_func = minibatch_spec.train, minibatch_spec.metrics_agg
        user_train_func_str = base64.b64encode(dill.dumps(user_train_func))
        user_metrics_func_str = base64.b64encode(dill.dumps(user_metrics_func))

        # spawn DDP workers
        self.logger.info(f"Executing DDP train for model {self.model_id} on worker {self.worker_id}")
        time.sleep(5)
        mp.spawn(self._execute_inner, args=(user_train_func_str, user_metrics_func_str), nprocs=self.world_size, join=True)
        self.logger.info("Completed DDP train function")

    def execute_val(self, minibatch_spec, model_id):
        # get val and metrics_agg functions
        self.mode = "val"
        self.model_id = model_id
        user_val_func, user_metrics_func = minibatch_spec.val_test, minibatch_spec.metrics_agg
        user_val_func_str = base64.b64encode(dill.dumps(user_val_func))
        user_metrics_func_str = base64.b64encode(dill.dumps(user_metrics_func))

        # spawn DDP workers
        self.logger.info(f"Executing DDP val for model {self.model_id} on worker {self.worker_id}")
        time.sleep(5)
        mp.spawn(self._execute_inner, args=(user_val_func_str, user_metrics_func_str), nprocs=self.world_size, join=True)
        self.logger.info("Completed DDP val function")

    def execute_test(self, minibatch_spec, model_tag):
        # get val and metrics_agg functions
        self.mode = "test"
        self.model_tag = model_tag
        user_test_func, user_metrics_func = minibatch_spec.val_test, minibatch_spec.metrics_agg
        user_test_func_str = base64.b64encode(dill.dumps(user_test_func))
        user_metrics_func_str = base64.b64encode(dill.dumps(user_metrics_func))

        # spawn DDP workers
        self.logger.info(f"Executing DDP test for model {self.model_id} on worker {self.worker_id}")
        time.sleep(5)
        mp.spawn(self._execute_inner, args=(user_test_func_str, user_metrics_func_str), nprocs=self.world_size, join=True)
        self.logger.info("Completed DDP test function")

    def execute_predict(self, minibatch_spec, model_tag):
        # get predict function
        self.mode = "predict"
        self.model_tag = model_tag
        user_pred_func = minibatch_spec.predict
        user_pred_func_str = base64.b64encode(dill.dumps(user_pred_func))

        self.logger.info(f"Executing DDP predict for model {model_tag} on worker {self.worker_id}")
        # spawn DDP workers
        time.sleep(5)
        mp.spawn(self._execute_inner, args=(user_pred_func_str,), nprocs=self.world_size, join=True)