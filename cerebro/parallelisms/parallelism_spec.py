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

class Parallelism:
    def __init__(self, worker_id, model_config, model_checkpoint_path, epoch):
        pass

    def save_local_metrics(self, rank, metrics, user_metrics_func):
        pass

    def load_checkpoint(self, model_object):
        pass

    def save_checkpoint(self, model_object):
        pass

    def execute_sample(self, minibatch_spec, sample_size):
        pass

    def execute_train(self, minibatch_spec, model_id):
        pass

    def execute_val(self, minibatch_spec, model_id, is_last_epoch):
        pass

    def execute_test(self, minibatch_spec, model_tag):
        pass

    def execute_predict(self, minibatch_spec, model_tag):
        pass
