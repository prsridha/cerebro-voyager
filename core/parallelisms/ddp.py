import os
import gc
import dill
import base64
import warnings
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from cerebro.util.save_metrics import SaveMetrics
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.parallelisms.parallelism_spec import Parallelism

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    warnings.filterwarnings("ignore")

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def clean_up():
    dist.destroy_process_group()


class DDPExecutor(Parallelism):
    def __init__(self, worker_id, model_config, model_checkpoint_path):
        super().__init__(worker_id, model_config, model_checkpoint_path)
        self.name = "DDPExecutor"
        self.mode = None
        self.sm_handle = None
        self.output_path = None
        self.model_object = None
        self.metrics_cycle_size = None

        kvs = KeyValueStore()
        self.seed = kvs.get_seed()
        self.worker_id = worker_id
        self.hyperparams = model_config
        self.model_path = model_checkpoint_path
        self.world_size = torch.cuda.device_count()

        # create state dict path
        self.state_dict_path = os.path.join(os.path.dirname(self.model_path), "state_dicts")
        Path(self.state_dict_path).mkdir(parents=True, exist_ok=True)

    def save_metrics(self, rank, metrics, user_metrics_func):
        grouped_metrics = defaultdict(list)

        # group by on key
        for item in metrics:
            for k, v in item.items():
                grouped_metrics[k].append(v)

        # convert to tensors and move to appropriate rank
        for k, v in grouped_metrics.items():
            grouped_metrics[k] = torch.Tensor(grouped_metrics[k]).to(rank)

        grouped_metrics_list = torch.stack(list(grouped_metrics.values())).to(rank)

        # reduce across ranks for train
        if self.mode == "train":
            dist.all_reduce(grouped_metrics_list, op=dist.ReduceOp.SUM)

        if rank == 0:
            # remove extra list layer and repack into dict
            reduced_metrics = dict(zip(grouped_metrics.keys(), map(lambda x: x.tolist(), grouped_metrics_list)))

            # run through user's metrics aggregator
            result = user_metrics_func(self.mode, self.hyperparams, reduced_metrics)

            # plot only train and val metrics on tensorboard
            if self.mode == "train" or self.mode == "val":
                self.sm_handle.save_to_tensorboard(result)
            elif self.mode == "test":
                output_filename = os.path.join(self.output_path, "test_output_{}.json".format(self.worker_id))
                self.sm_handle.save_to_file(result, output_filename)

    def load_checkpoint(self, model_object):
        # check if checkpoints exist
        if len(os.listdir(self.state_dict_path)) == 0:
            return model_object

        new_model_object = {}
        rank = dist.get_rank()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

        for obj_name, obj in model_object.items():
            state_dict_file_path = os.path.join(self.state_dict_path, obj_name + ".pt")
            model_checkpoint = torch.load(state_dict_file_path, map_location=map_location)
            if obj_name != "optimizer":
                obj.to(rank)
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

    def _execute_inner(self, rank, user_func_str, dataset, user_metrics_func_str=None):
        # empty GPU cache
        torch.cuda.empty_cache()

        # load user_func from serialized str
        user_func = dill.loads(base64.b64decode(user_func_str))
        user_metrics_func = dill.loads(base64.b64decode(user_metrics_func_str)) if user_metrics_func_str else None

        # initialize process with this rank
        setup(rank, self.world_size)
        torch.cuda.set_device(rank)
        current_device = torch.cuda.current_device()

        # create data sampler and dataloader
        sampler = DistributedSampler(dataset, rank=rank, num_replicas=self.world_size, shuffle=False, seed=self.seed)
        dataloader = DataLoader(dataset, batch_size=self.hyperparams["batch_size"], sampler=sampler, pin_memory=True,
                                num_workers=2, shuffle=False)

        # load models and their state dicts
        model_object = torch.load(self.model_path)
        updated_obj = self.load_checkpoint(model_object)

        # parallelize models from model object file
        for obj_name, obj in updated_obj.items():
            if obj_name != "optimizer":
                obj.to(rank)
                p_model = DDP(obj, device_ids=[rank])
                updated_obj[obj_name] = p_model

        # call user function
        if self.mode == "sample":
            for k, minibatch in enumerate(dataloader):
                user_func(updated_obj, minibatch, self.hyperparams, current_device)
        elif self.mode == "train":
            train_metrics = []
            for k, minibatch in enumerate(dataloader):
                updated_obj, minibatch_metrics = user_func(updated_obj, minibatch, self.hyperparams, current_device)
                train_metrics.append(minibatch_metrics)

                # this has to reach the user's logs/notebook
                if rank == 0:
                    print("Minibatch number :", k)

                # At every "metrics_cycle_size" - reduce metrics across all ranks
                if k % self.metrics_cycle_size == 0:
                    self.save_metrics(rank, train_metrics, user_metrics_func)
                    train_metrics = []

            # save checkpoint of uploaded model object
            self.save_checkpoint(updated_obj)

        elif self.mode == "val":
            val_metrics = []
            for k, minibatch in dataloader:
                minibatch_metrics = user_func(updated_obj, minibatch, self.hyperparams, current_device)
                val_metrics.append(minibatch_metrics)

            # At every epoch - save metrics
            if rank == 0:
                self.save_metrics(rank, val_metrics, user_metrics_func)

        elif self.mode == "test":
            test_outputs = []
            for k, minibatch in dataloader:
                output = user_func(updated_obj, minibatch, self.hyperparams, current_device)
                test_outputs.append(output)

            if rank == 0:
                self.save_metrics(rank, test_outputs, user_metrics_func)

        elif self.mode == "predict":
            predict_outputs = []
            for k, minibatch in dataloader:
                output = user_func(updated_obj, minibatch, self.hyperparams, current_device)
                output["row_id"] = minibatch[3]
                predict_outputs.append(output)

            if rank == 0:
                output_df = pd.DataFrame(predict_outputs)
                output_df.to_csv(self.output_path)

        print("Completed DDP event")

        # clean up resources
        del sampler
        del dataloader
        del updated_obj
        del model_object
        gc.collect()
        clean_up()

    def execute_sample(self, user_train_func, dataset):
        # set values
        self.mode = "sample"
        user_train_func_str = base64.b64encode(dill.dumps(user_train_func))

        # spawn DDP workers
        mp.spawn(self._execute_inner, args=(user_train_func_str, dataset), nprocs=self.world_size, join=True)

    def execute_train(self, user_train_func, user_metrics_func, dataset, metrics_cycle_size, model_id):
        # set values, create SaveMetrics handle and necessary paths
        self.mode = "train"
        self.metrics_cycle_size = metrics_cycle_size
        self.sm_handle = SaveMetrics(self.mode, model_id)
        user_train_func_str = base64.b64encode(dill.dumps(user_train_func))
        user_metrics_func_str = base64.b64encode(dill.dumps(user_metrics_func))

        # spawn DDP workers
        mp.spawn(self._execute_inner, args=(user_train_func_str, dataset, user_metrics_func_str), nprocs=self.world_size, join=True)

    def execute_val(self, user_val_func, user_metrics_func, dataset, model_id, epoch):
        # set values, create SaveMetrics handle and necessary paths
        self.mode = "val"
        self.sm_handle = SaveMetrics(self.mode, model_id, epoch)
        user_val_func_str = base64.b64encode(dill.dumps(user_val_func))
        user_metrics_func_str = base64.b64encode(dill.dumps(user_metrics_func))

        # spawn DDP workers
        mp.spawn(self._execute_inner, args=(user_val_func_str, dataset, user_metrics_func_str), nprocs=self.world_size, join=True)

    def execute_test(self, user_test_func, dataset, output_path):
        # set values and create necessary paths
        self.mode = "test"
        self.output_path = output_path
        user_test_func_str = base64.b64encode(dill.dumps(user_test_func))

        # spawn DDP workers
        mp.spawn(self._execute_inner, args=(user_test_func_str, dataset), nprocs=self.world_size, join=True)

    def execute_predict(self, user_pred_func, dataset, output_path):
        # set values and create necessary paths
        self.mode = "predict"
        self.output_path = output_path
        user_pred_func_str = base64.b64encode(dill.dumps(user_pred_func))

        # spawn FSDP workers
        mp.spawn(self._execute_inner, args=(user_pred_func_str, dataset), nprocs=self.world_size, join=True)