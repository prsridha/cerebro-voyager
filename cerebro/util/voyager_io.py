import os
import shutil
from pathlib import Path
from cerebro.util.file_io import FileIO


class VoyagerIO(FileIO):
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

        for root, dirs, files in os.walk(local_prefix):
            for file in files:
                full_src_path = os.path.join(root, file)
                completed_size += os.path.getsize(full_src_path)
                remaining_prefix = full_src_path.replace(exclude_prefix, "").lstrip(os.path.sep)
                dest_path = os.path.join(remote_prefix, remaining_prefix) if remaining_prefix else remote_prefix
                Path(os.path.dirname(dest_path)).mkdir(parents=True, exist_ok=True)
                shutil.copy(full_src_path, dest_path)
                self.update_progress(completed_size)

    def download(self, local_prefix, remote_prefix, exclude_prefix=""):
        completed_size = 0
        self.set_total_size(remote_prefix)

        for root, dirs, files in os.walk(remote_prefix):
            for file in files:
                full_src_path = os.path.join(root, file)
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
