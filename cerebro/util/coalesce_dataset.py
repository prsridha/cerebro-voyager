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

import gc
import pandas as pd
from torch.utils.data import Dataset


class CoalesceDataset(Dataset):
    def __init__(self, file_path):
        gc.collect()
        self.file_path = file_path

        # this reads entire file to memory
        self.df = pd.read_pickle(self.file_path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        gc.collect()

        row_id = self.df["id"].iloc[idx]
        input_tensor = self.df["input_tensor"].iloc[idx]
        if "output_tensor" in self.df:
            output_tensor = self.df["output_tensor"].iloc[idx]
            return input_tensor, output_tensor, row_id
        else:
            return input_tensor, row_id
