import os
import boto3
from pathlib import Path
from cerebro.util.file_io import FileIO


class S3IO(FileIO):
    def __init__(self, bucket_name, update_progress_fn=None) -> None:
        super().__init__()
        self.progress = 0
        self.total_size = 0
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.update_progress_fn = update_progress_fn

    def update_progress(self, chunk):
        if self.update_progress_fn:
            self.progress = (chunk * 100) / self.total_size
            self.update_progress_fn(self.progress)

    def list_files(self, s3_prefix):
        s3_prefix = s3_prefix.split("/", 3)[3]
        objs = list(boto3.resource('s3').Bucket(self.bucket_name).objects.filter(Prefix=s3_prefix))[1:]
        filtered_objs = [os.path.basename(obj.key) for obj in objs]
        return filtered_objs

    def upload(self, local_path, remote_prefix, s3_prefix=''):
        s3_prefix = s3_prefix.split("/", 3)[3]
        is_file = os.path.isfile(local_path)
        # get total size
        if is_file:
            self.total_size = os.path.getsize(local_path)
        else:
            total_size = 0
            for dirpath, _, filenames in os.walk(local_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(file_path)
            self.total_size = total_size

        # upload file(s)
        if is_file:
            s3_prefix = os.path.join(s3_prefix, local_path)
            self.s3_client.upload_file(local_path, self.bucket_name, s3_prefix, Callback=self.update_progress)
        else:
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, local_path)
                    s3_key = os.path.join(s3_prefix, relative_path).replace('\\', '/')

                    self.s3_client.upload_file(file_path, self.bucket_name, s3_key, Callback=self.update_progress)

    def download(self, local_path, s3_prefix, exclude_prefix):
        # local path is the dir in which the files are to be downloaded
        # s3_prefix is the file location on S3

        s3_prefix = s3_prefix.split("/", 3)[3]
        exclude_prefix = exclude_prefix.split("/", 3)[3]
        objs = list(boto3.resource('s3').Bucket(self.bucket_name).objects.filter(Prefix=s3_prefix))

        # get total size
        if len(objs) == 1:
            response = self.s3_client.head_object(Bucket=objs[0].bucket_name, Key=objs[0].key)
            self.total_size = response['ContentLength']
        else:
            total_size = 0
            for obj in objs:
                response = self.s3_client.head_object(Bucket=obj.bucket_name, Key=obj.key)
                total_size += response['ContentLength']
            self.total_size = total_size

        # download file(s)
        for obj in objs:
            remaining_prefix = str(obj.key).replace(exclude_prefix, "").lstrip(os.path.sep)
            to_path = os.path.join(local_path, remaining_prefix) if remaining_prefix else local_path
            Path(os.path.dirname(to_path)).mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(obj.bucket_name, obj.key, to_path, Callback=self.update_progress)
