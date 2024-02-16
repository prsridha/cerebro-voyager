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

class MiniBatchSpec:
    def __init__(self):
        pass

    def initialize_worker(self):
        pass

    def create_model_components(self, hyperparams):
        pass

    def train(self, model_object, minibatch, hyperparams, device):
        pass

    def valtest(self, model_object, minibatch, hyperparams, device):
        pass

    def predict(self, model_object, minibatch, hyperparams, device):
        pass
