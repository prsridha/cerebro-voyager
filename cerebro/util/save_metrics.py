import os
import csv
import json
import warnings
from pathlib import Path
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from cerebro.util.params import Params

class SaveMetrics:
    def __init__(self, mode, model_id, epoch=None):
        warnings.filterwarnings("ignore")

        self.mode = mode
        self.epoch = epoch
        self.model_id = model_id

        params = Params()
        self.metrics_metadata_path = os.path.join(params.mop["metrics_storage_path"]["meta_metrics"],
                                                  "model{}.json".format(model_id))
        self.user_metrics_path = os.path.join(params.mop["metrics_storage_path"]["user_metrics"], mode,
                                              "model{}".format(model_id))
        self.tensorboard_path = os.path.join(params.mop["metrics_storage_path"]["tensorboard"],
                                             mode, "model", str(model_id))
        Path(os.path.dirname(self.metrics_metadata_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.user_metrics_path)).mkdir(parents=True, exist_ok=True)
        Path(self.tensorboard_path).mkdir(parents=True, exist_ok=True)

    def save_to_file(self, metrics, output_path):
        with open(output_path) as f:
            json.dump(metrics, f)

    def save_to_tensorboard(self, metrics):
        user_metrics = defaultdict(list)
        summary_writer = tf.summary.create_file_writer(self.tensorboard_path, name=self.mode + "_" + str(self.model_id))

        if self.mode == "train":
            if os.path.isfile(self.metrics_metadata_path):
                with open(self.metrics_metadata_path, "r") as f:
                    offset_metrics = json.load(f)
                offset_step = offset_metrics["total_step"]
            else:
                offset_step = 0

            for stat_name in metrics:
                user_metrics[stat_name].append((offset_step + 1, metrics[stat_name]))
                with summary_writer.as_default():
                    tf.summary.scalar(stat_name, float(metrics[stat_name]), step=(offset_step + 1))

            # save metrics metadata to files
            new_metrics_metadata = {
                "total_step": offset_step + 1
            }
            with open(self.metrics_metadata_path, "w+") as f:
                json.dump(new_metrics_metadata, f)

        elif self.mode == "val":
            for stat_name in metrics:
                user_metrics[stat_name].append((self.epoch, metrics[stat_name]))
                with summary_writer.as_default():
                    tf.summary.scalar(stat_name, metrics[stat_name], step=self.epoch)

        summary_writer.close()

        # save metrics to separate files (apart from Tensorboard)
        with open(self.user_metrics_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=user_metrics.keys())

            # if the file doesn't exist, write the header row
            if not os.path.isfile(self.user_metrics_path):
                writer.writeheader()

            # write the data row
            writer.writerow(user_metrics)
