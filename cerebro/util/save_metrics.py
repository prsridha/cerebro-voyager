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
    def __init__(self):
        warnings.filterwarnings("ignore")

    @staticmethod
    def save_to_file(metrics, mode, output_filename):
        params = Params()
        user_metrics_path = os.path.join(params.mop["metrics_storage_path"]["user_metrics"], mode,
                                              output_filename)
        Path(os.path.dirname(user_metrics_path)).mkdir(parents=True, exist_ok=True)

        # save metrics to separate files (apart from Tensorboard)
        with open(user_metrics_path, mode='a+', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())

            # if the file doesn't exist, write the header row
            if not os.path.isfile(user_metrics_path):
                writer.writeheader()

            # write the data row
            writer.writerow(metrics)

    @staticmethod
    def save_to_tensorboard(metrics, mode, model_id, epoch=None):
        # get paths
        params = Params()
        metrics_metadata_path = os.path.join(params.mop["metrics_storage_path"]["meta_metrics"],
                                             "model{}.json".format(model_id))
        tensorboard_path = os.path.join(params.mop["metrics_storage_path"]["tensorboard"],
                                        mode, "model", str(model_id))
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(metrics_metadata_path)).mkdir(parents=True, exist_ok=True)

        # create other data structures
        user_metrics = defaultdict(list)
        summary_writer = tf.summary.create_file_writer(tensorboard_path, name=mode + "_" + str(model_id))

        if mode == "train":
            if os.path.isfile(metrics_metadata_path):
                with open(metrics_metadata_path, "r") as f:
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
            with open(metrics_metadata_path, "w+") as f:
                json.dump(new_metrics_metadata, f)

        elif mode == "val" or mode == "train_epoch":
            for stat_name in metrics:
                user_metrics[stat_name].append((epoch, metrics[stat_name]))
                with summary_writer.as_default():
                    tf.summary.scalar(stat_name, metrics[stat_name], step=epoch)

        summary_writer.close()
