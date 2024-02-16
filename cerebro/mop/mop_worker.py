# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import gc
import glob
import json
import time
import traceback
import pandas as pd
from pathlib import Path
from kubernetes import client, config

from cerebro.util.params import Params
import cerebro.kvs.constants as kvs_constants
from cerebro.util.save_metrics import SaveMetrics
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.util.cerebro_logger import CerebroLogger
from cerebro.parallelisms.parallelism_init import get_parallelism_executor


class CerebroWorker:
    def __init__(self):
        # obtain worker_id from env variable
        worker_name = os.environ.get("ORDINAL_ID")
        worker_id = int(worker_name.split("-")[-1])

        logging = CerebroLogger("worker-{}".format(worker_id))
        self.logger = logging.create_logger("mop-worker")
        self.logger.info("Starting MOP worker {}".format(worker_id))

        self.params = None
        self.minibatch_spec = None
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
        self.minibatch_spec = self.kvs.mop_get_spec()
        self.minibatch_spec.initialize_worker()
        self.logger.info("Subepoch init worker called")

    def sample_parallelism(self, model_id, parallelism_name):
        self.logger.info(
            "Sampling model {} with parallelism {} on worker {}".format(model_id, parallelism_name, self.worker_id))

        # get model hyperparameters
        model_config = self.kvs.mop_get_model_mapping(model_id)

        # create parallelism object of given Parallelism class
        ParallelismExecutor = get_parallelism_executor(parallelism_name)
        model_checkpoint_path = os.path.join(self.params.mop["checkpoint_storage_path"], "model_" + str(model_id),
                                             "model_object_{}.pt".format(model_id))
        parallelism = ParallelismExecutor(self.worker_id, model_config, model_checkpoint_path, 0)

        self.logger.info("Created parallelism object")

        # call the train function
        start = time.time()
        try:
            parallelism.execute_sample(self.minibatch_spec, self.sample_size)
        except Exception as e:
            gc.collect()
            err_msg = str(e) + "\n" + traceback.format_exc()
            self.kvs.set_error(str(err_msg))
            return
        end = time.time()

        time_elapsed = end - start
        self.kvs.mop_set_sample_time(model_id, parallelism_name, time_elapsed)
        self.logger.info("Completed parallelism sampling of model {} with parallelism {} on worker {}".format(model_id, parallelism_name, self.worker_id))

        # set worker status as complete
        self.kvs.mop_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)

    def train_model(self, model_id, epoch, is_last_worker, is_last_epoch):
        print("Training model {} on worker {}".format(model_id, self.worker_id))
        self.logger.info("Training model {} on worker {}".format(model_id, self.worker_id))

        # get the best parallelism class for this model
        p_name = self.kvs.mop_get_parallelism_mapping(model_id)
        ParallelismExecutor = get_parallelism_executor(p_name)

        # define necessary paths and create parallelism object
        model_checkpoint_path = os.path.join(self.params.mop["checkpoint_storage_path"], "model_" + str(model_id),
                                             "model_object_{}.pt".format(model_id))
        model_config = self.kvs.mop_get_model_mapping(model_id)
        parallelism = ParallelismExecutor(self.worker_id, model_config, model_checkpoint_path, epoch)

        # call the train function
        try:
            parallelism.execute_train(self.minibatch_spec, model_id)
        except Exception as e:
            gc.collect()
            err_msg = str(e) + "\n" + traceback.format_exc()
            self.kvs.set_error(str(err_msg))
            return

        if is_last_worker:
            # aggregate and plot train epoch metrics
            csv_path = os.path.join(self.params.mop["metrics_storage_path"]["user_metrics"], "train",
                                    "{}.csv".format(model_id))
            metrics_df = pd.read_csv(csv_path)
            reduced_df = self.minibatch_spec.metrics_agg("train", model_config, metrics_df)
            SaveMetrics.save_to_tensorboard(reduced_df, "train_epoch", model_id, epoch)

            # run validation
            self.validate_model(model_id, parallelism, is_last_epoch)
            self.logger.info("Completed validation of model {} on worker {}".format(model_id, self.worker_id))
            print("Completed validation of model {} on worker {}".format(model_id, self.worker_id))

        # set worker status as complete
        self.kvs.mop_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)

    def validate_model(self, model_id, parallelism, is_last_epoch):
        # run validation via parallelism
        try:
            parallelism.execute_val(self.minibatch_spec, model_id, is_last_epoch)
        except Exception as e:
            gc.collect()
            err_msg = str(e) + "\n" + traceback.format_exc()
            self.kvs.set_error(str(err_msg))
            return

    def test_model(self, model_tag, batch_size):
        # get model checkpoint path
        if model_tag.isdigit():
            model_checkpoint_path = os.path.join(self.params.mop["checkpoint_storage_path"], "model_" + str(model_tag),
                                                 "model_object_{}.pt".format(model_tag))
        else:
            model_checkpoint_path = os.path.join(self.params.mop["checkpoint_storage_path"], model_tag)

        # default parallelism to DDP
        p_name = "DDP"
        ParallelismExecutor = get_parallelism_executor(p_name)

        # create parallelism object
        model_config = {"batch_size": batch_size}
        parallelism = ParallelismExecutor(self.worker_id, model_config, model_checkpoint_path, 0)

        # run test via parallelism
        model_tag_stem = str(Path(model_tag).stem)
        try:
            parallelism.execute_test(self.minibatch_spec, model_tag_stem)
        except Exception as e:
            gc.collect()
            err_msg = str(e) + "\n" + traceback.format_exc()
            self.kvs.set_error(str(err_msg))
            return

        # set worker status as complete
        self.kvs.mop_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)

    def predict_model(self, model_tag, batch_size):
        # get model checkpoint path
        if model_tag.isdigit():
            model_checkpoint_path = os.path.join(self.params.mop["checkpoint_storage_path"], "model_" + str(model_tag),
                                                 "model_object_{}.pt".format(model_tag))
        else:
            model_checkpoint_path = os.path.join(self.params.mop["checkpoint_storage_path"], model_tag)

        # default parallelism to DDP
        p_name = "DDP"
        ParallelismExecutor = get_parallelism_executor(p_name)

        # create parallelism object
        model_config = {"batch_size": batch_size}
        parallelism = ParallelismExecutor(self.worker_id, model_config, model_checkpoint_path, 0)

        # run test via parallelism
        model_tag_stem = str(Path(model_tag).stem)
        predict_output_path = self.params.mop["predict_output_path"]
        Path(os.path.dirname(predict_output_path)).mkdir(exist_ok=True)
        try:
            parallelism.execute_predict(self.minibatch_spec, model_tag_stem)
        except Exception as e:
            gc.collect()
            err_msg = str(e) + "\n" + traceback.format_exc()
            self.kvs.set_error(str(err_msg))
            return

        # combine inference files of all ranks
        predict_output_rank_files = os.path.join(predict_output_path,
                                                 f"predict_output_{model_tag_stem}_{self.worker_id}_*.csv")
        combined_output_file = os.path.join(predict_output_path,
                                            f"predict_output_{model_tag_stem}_{self.worker_id}.csv")
        files = glob.glob(predict_output_rank_files)
        combined_df = pd.DataFrame()
        for file in files:
            df = pd.read_csv(file, header=0)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            os.remove(file)
        combined_df.to_csv(combined_output_file, index=False)

        # set worker status as complete
        self.kvs.mop_set_worker_status(self.worker_id, kvs_constants.PROGRESS_COMPLETE)

    def server_forever(self):
        done = False
        prev_task, prev_task_id = None, None
        task_id, task = self.kvs.mop_get_task(self.worker_id)
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
                    self.logger.info("Received task - Sampling in worker {}".format(self.worker_id))
                    model_id, parallelism_name = self.kvs.mop_get_model_parallelism_on_worker(self.worker_id)
                    self.sample_parallelism(model_id, parallelism_name)
                elif task == kvs_constants.MOP_TASK_TRAIN_VAL:
                    self.logger.info("Received task - Train/Val in worker {}".format(self.worker_id))
                    d = self.kvs.mop_get_models_on_worker(self.worker_id)
                    self.train_model(d["model_id"], d["epoch"], d["is_last_worker"], d["is_last_epoch"])
                elif task == kvs_constants.MOP_TASK_TEST:
                    self.logger.info("Received task - Test in worker {}".format(self.worker_id))
                    model_tag, batch_size = self.kvs.mop_get_test_params()
                    self.test_model(model_tag, batch_size)
                elif task == kvs_constants.MOP_TASK_PREDICT:
                    self.logger.info("Received task - Inference in worker {}".format(self.worker_id))
                    model_tag, batch_size = self.kvs.mop_get_predict_params()
                    self.predict_model(model_tag, batch_size)
                elif task == kvs_constants.PROGRESS_COMPLETE:
                    done = True
                    self.logger.info("MOP tasks complete on worker{}".format(self.worker_id))

                # attempting garbage collection of previous threads
                gc.collect()

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
                task_id, task = self.kvs.mop_get_task(self.worker_id)

def main():
    worker = CerebroWorker()
    worker.server_forever()


if __name__ == '__main__':
    main()
