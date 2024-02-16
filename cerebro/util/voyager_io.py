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
import shutil
from pathlib import Path


class VoyagerIO:
    def __init__(self, update_progress_fn=None):
        super().__init__()
        self.progress = 0
        self.total_size = 0
        self.update_progress_fn = update_progress_fn

    def update_progress(self, chunk):
        if self.total_size > 0:
            self.progress = chunk / self.total_size * 100
        if self.update_progress_fn:
            self.update_progress_fn(self.progress)

    def set_total_size(self, path):
        if os.path.isfile(path):
            total_size = os.path.getsize(path)
        else:
            total_size = shutil.disk_usage(path).used
        self.total_size = total_size

    def upload(self, local_prefix, remote_prefix, exclude_prefix=""):
        completed_size = 0
        self.set_total_size(local_prefix)

        all_files = []
        if os.path.isfile(local_prefix):
            all_files.append(local_prefix)
        else:
            for root, dirs, files in os.walk(local_prefix):
                for file in files:
                    all_files.append(os.path.join(root, file))

        for full_src_path in all_files:
            completed_size += os.path.getsize(full_src_path)
            remaining_prefix = full_src_path.replace(exclude_prefix, "").lstrip(os.path.sep)
            dest_path = os.path.join(remote_prefix, remaining_prefix) if remaining_prefix else remote_prefix
            Path(os.path.dirname(dest_path)).mkdir(parents=True, exist_ok=True)
            shutil.copy(full_src_path, dest_path)
            self.update_progress(completed_size)

    def download(self, local_prefix, remote_prefix, exclude_prefix=""):
        completed_size = 0
        self.set_total_size(remote_prefix)

        all_files = []
        if os.path.isfile(remote_prefix):
            all_files.append(remote_prefix)
        else:
            for root, dirs, files in os.walk(remote_prefix):
                for file in files:
                    all_files.append(os.path.join(root, file))

        for full_src_path in all_files:
            completed_size += os.path.getsize(full_src_path)
            remaining_prefix = full_src_path.replace(exclude_prefix, "").lstrip(os.path.sep)
            dest_path = os.path.join(local_prefix, remaining_prefix) if remaining_prefix else local_prefix
            Path(os.path.dirname(dest_path)).mkdir(parents=True, exist_ok=True)
            shutil.copy(full_src_path, dest_path)
            self.update_progress(completed_size)

    def list_files(self, path):
        files = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files
