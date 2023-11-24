import os
import gc
import dill
import base64

import torch
from torch.utils.data import DataLoader

# from torchgpipe import GPipe
# from torchgpipe.balance import balance_by_time

from cerebro.parallelisms.parallelism_spec import Parallelism


class GPipeExecutor(Parallelism):
    def __init__(self, models, optimizer, model_config, model_checkpoint_path=None):
        self.name = "GPipeExecutor"
        self.models = models
        self.optimizer = optimizer
        self.hyperparams = model_config
        self.model_path = model_checkpoint_path
        self.world_size = torch.cuda.device_count()
        self.gpus = [torch.device('cuda', g) for g in range(self.world_size)]

        self.microbatch_chunks = 4
        self.balance_strategy = balance_by_time

    def checkpoint(self, method, states=None):
        if method == "save":
            torch.save(states, self.model_path)
        elif method == "load":
            if os.path.isfile(self.model_path):
                chkpnt = torch.load(self.model_path)
                return chkpnt
            else:
                return None

    def dummy_checkpoint(self, method=None, states=None):
        # do nothing - checkpoints skipped for trial runs
        return None

    def _execute_inner(self, mode, user_func_str, dataset):
        # disable tensorflow warnings and errors
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # empty GPU cache
        torch.cuda.empty_cache()

        # load user_func from serialized str
        user_func = dill.loads(base64.b64decode(user_func_str))

        # create dataloader
        dataloader = DataLoader(dataset, batch_size=self.hyperparams["batch_size"])
        input_tensor = next(iter(dataloader))[0]

        # parallelize models
        p_models = []
        for model in self.models:
            balance = self.balance_strategy(self.world_size, model, input_tensor)
            m = GPipe(
                model,
                balance,
                self.microbatch_chunks
            )
            p_models.append(m)

        # set input device and output devices (first and last GPU)
        input_device = self.gpus[0]
        output_device = self.gpus[-1]

        if mode == "train":
            metrics = user_func(p_models, self.optimizer, self.checkpoint, dataloader, self.hyperparams, input_device, output_device)
        elif mode == "sample":
            metrics = user_func(p_models, self.optimizer, self.dummy_checkpoint, dataloader, self.hyperparams, input_device, output_device)
        elif mode == "valid":
            metrics = user_func(p_models, self.checkpoint, dataloader, self.hyperparams, input_device, output_device)
        elif mode == "test":
            output = user_func(p_models, self.checkpoint, dataloader, self.hyperparams, input_device)

        if mode in ["train", "sample", "valid"]:
            return metrics
        elif mode == "test":
            return output

        # clean up resources
        del dataloader
        del p_models
        gc.collect()

    def execute_sample(self, user_train_func, dataset):
        user_train_func_str = base64.b64encode(dill.dumps(user_train_func))
        self._execute_inner("sample", user_train_func_str, dataset)

    def execute_train(self, user_train_func, dataset):
        user_train_func_str = base64.b64encode(dill.dumps(user_train_func))
        metrics = self._execute_inner("train", user_train_func_str, dataset)
        return metrics

    def execute_valid(self, user_valid_func, dataset):
        user_valid_func_str = base64.b64encode(dill.dumps(user_valid_func))
        metrics = self._execute_inner("valid", user_valid_func_str, dataset)
        return metrics

    def execute_test(self, user_test_func, dataset):
        user_valid_func_str = base64.b64encode(dill.dumps(user_test_func))
        output = self._execute_inner("test", user_valid_func_str, dataset)
        return output
