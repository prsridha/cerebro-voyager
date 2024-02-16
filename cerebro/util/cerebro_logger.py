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
import logging
from pathlib import Path


class CerebroLogger:
    def __init__(self, filename):
        log_root_path = '/data/logs'
        Path(log_root_path).mkdir(parents=True, exist_ok=True)

        log_format = '%(levelname)5s: %(name)5s - %(message)10s [%(filename)5s:%(lineno)d @ %(asctime)s]'
        logging.basicConfig(level=logging.INFO,
                            format=log_format,
                            filename=os.path.join(log_root_path, filename + ".log"),
                            filemode='a+')
        self.console = logging.StreamHandler()
        self.console.setLevel(logging.INFO)
        self.console.setFormatter(logging.Formatter(log_format))

    def create_logger(self, name):
        return logging.getLogger(name)