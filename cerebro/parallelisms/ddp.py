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

from cerebro.util.params import Params
from cerebro.util.save_metrics import SaveMetrics
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.util.cerebro_logger import CerebroLogger
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
    def __init__(self, worker_id, model_config, model_checkpoint_path, epoch):
        super().__init__(worker_id, model_config, model_checkpoint_path, epoch)
        self.name = "DDPExecutor"
        logging = CerebroLogger("worker-{}".format(worker_id))
        self.logger = logging.create_logger("ddp-worker")

        self.mode = None
        self.epoch = epoch
        self.model_id = None
        self.params = Params()
        self.worker_id = worker_id
        self.kvs = KeyValueStore()
        self.seed = self.kvs.get_seed()
        self.hyperparams = model_config
        self.model_path = model_checkpoint_path
        self.world_size = torch.cuda.device_count()

        # create state dict path
        self.state_dict_path = os.path.join(os.path.dirname(self.model_path), "state_dicts")
        Path(self.state_dict_path).mkdir(parents=True, exist_ok=True)

    def save_local_metrics(self, rank, metrics, user_metrics_func):
        grouped_metrics = defaultdict(list)

        # group-by on key
        for item in metrics:
            for k, v in item.items():
                grouped_metrics[k].append(v)

        # convert to tensors and move to appropriate rank
        for k in grouped_metrics:
            grouped_metrics[k] = torch.Tensor(grouped_metrics[k]).to(rank)
        grouped_metrics_list = torch.stack(list(grouped_metrics.values())).to(rank)

        # reduce across ranks to get mean
        dist.all_reduce(grouped_metrics_list, op=dist.ReduceOp.AVG)
        if rank == 0:
            # remove extra list layer and repack into dict
            reduced_metrics = dict(zip(grouped_metrics.keys(), map(lambda x: x.tolist(), grouped_metrics_list)))

            # plot only train and val metrics on tensorboard
            if self.mode == "train":
                SaveMetrics.save_to_tensorboard(reduced_metrics, self.mode, self.model_id, self.epoch)
                SaveMetrics.save_to_file(reduced_metrics, self.mode, "{}.csv".format(self.model_id))
            elif self.mode == "val":
                result = user_metrics_func(self.mode, self.hyperparams, reduced_metrics)
                SaveMetrics.save_to_tensorboard(result, self.mode, self.model_id, self.epoch)
            elif self.mode == "test":
                # run through user's metrics aggregator
                result = user_metrics_func(self.mode, self.hyperparams, reduced_metrics)
                output_filename = "test_output_{}.csv".format(self.model_id)
                SaveMetrics.save_to_file(result, self.mode, output_filename)

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

        # call user's ML function
        if self.mode == "sample":
            for k, minibatch in enumerate(dataloader):
                user_func(updated_obj, minibatch, self.hyperparams, current_device)

        elif self.mode == "train":
            for k, minibatch in enumerate(dataloader):
                updated_obj, minibatch_metrics = user_func(updated_obj, minibatch, self.hyperparams, current_device)
                self.save_local_metrics(rank, minibatch_metrics, user_metrics_func)

            # save checkpoint of uploaded model object
            self.save_checkpoint(updated_obj)

        elif self.mode == "val":
            val_metrics = []
            for k, minibatch in dataloader:
                minibatch_metrics = user_func(updated_obj, minibatch, self.hyperparams, current_device)
                val_metrics.append(minibatch_metrics)

            self.save_local_metrics(rank, val_metrics, user_metrics_func)

        elif self.mode == "test":
            test_outputs = []
            for k, minibatch in dataloader:
                output = user_func(updated_obj, minibatch, self.hyperparams, current_device)
                test_outputs.append(output)

            self.save_local_metrics(rank, test_outputs, user_metrics_func)

        elif self.mode == "predict":
            predict_outputs = []
            for k, minibatch in dataloader:
                output = user_func(updated_obj, minibatch, self.hyperparams, current_device)
                output["row_id"] = minibatch[3]
                predict_outputs.append(output)

            if rank == 0:
                output_filename = os.path.join(self.params.mop["predict_output_path"], "predict_output_{}.csv".format(self.model_id))
                output_df = pd.DataFrame(predict_outputs)
                output_df.to_csv(output_filename)

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
        mode = "sample"
        user_train_func_str = base64.b64encode(dill.dumps(user_train_func))

        # spawn DDP workers
        mp.spawn(self._execute_inner, args=(mode, user_train_func_str, dataset), nprocs=self.world_size, join=True)

    def execute_train(self, dataset, model_id):
        # get train and metrics_agg functions
        self.mode = "train"
        self.model_id = model_id
        spec = self.kvs.mop_get_spec()
        user_train_func, user_metrics_func = spec.train, spec.metrics_agg
        user_train_func_str = base64.b64encode(dill.dumps(user_train_func))
        user_metrics_func_str = base64.b64encode(dill.dumps(user_metrics_func))

        # spawn DDP workers
        mp.spawn(self._execute_inner, args=(user_train_func_str, dataset, user_metrics_func_str), nprocs=self.world_size, join=True)

    def execute_val(self, dataset, model_id):
        # get val and metrics_agg functions
        self.mode = "val"
        self.model_id = model_id
        spec = self.kvs.mop_get_spec()
        user_val_func, user_metrics_func = spec.valtest, spec.metrics_agg
        user_val_func_str = base64.b64encode(dill.dumps(user_val_func))
        user_metrics_func_str = base64.b64encode(dill.dumps(user_metrics_func))

        # spawn DDP workers
        mp.spawn(self._execute_inner, args=(user_val_func_str, dataset, user_metrics_func_str), nprocs=self.world_size, join=True)

    def execute_test(self, dataset):
        # get val and metrics_agg functions
        self.mode = "test"
        spec = self.kvs.mop_get_spec()
        user_test_func, user_metrics_func = spec.valtest, spec.metrics_agg
        user_test_func_str = base64.b64encode(dill.dumps(user_test_func))

        # spawn DDP workers
        mp.spawn(self._execute_inner, args=(user_test_func_str, dataset, user_metrics_func), nprocs=self.world_size, join=True)

    def execute_predict(self, dataset):
        # get val and metrics_agg functions
        self.mode = "predict"
        spec = self.kvs.mop_get_spec()
        user_pred_func = spec.predict
        user_pred_func_str = base64.b64encode(dill.dumps(user_pred_func))

        # spawn DDP workers
        mp.spawn(self._execute_inner, args=(user_pred_func_str, dataset), nprocs=self.world_size, join=True)