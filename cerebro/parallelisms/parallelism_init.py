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

from cerebro.parallelisms.ddp import DDPExecutor
from cerebro.parallelisms.fsdp import FSDPExecutor
from cerebro.parallelisms.gpipe import GPipeExecutor

PARALLELISMS_LIST = [
    "FSDP",
    "DDP"
]


def get_parallelism_executor(name):
    if name == "FSDP":
        return FSDPExecutor
    elif name == "DDP":
        return DDPExecutor
    elif name == "GPIPE":
        return GPipeExecutor
