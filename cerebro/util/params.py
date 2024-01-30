import re
import os
import json
from collections import defaultdict
from cerebro.kvs.KeyValueStore import KeyValueStore


class Params:
    def __init__(self):
        # load from KVS
        config = self._load_from_kvs()

        # assign all variables to the class
        self.etl = config.get('etl')
        self.mop = config.get('mop')
        self.miscellaneous = config.get('miscellaneous')

    def _load_from_kvs(self):
        kvs = KeyValueStore()
        params = kvs.get_dataset_locators()

        pattern = r"\/voyager\/ceph\/.*\/datasets"
        new_params = {}
        for k, v in params.items():
            if k == "misc":
                misc_v = []
                for i in v:
                    misc_v.append(re.sub(pattern, "/datasets", i))
                new_params[k] = misc_v
            else:
                new_params[k] = re.sub(pattern, "/datasets", v)
        params = new_params

        # set misc params
        miscellaneous = {
            "download_paths": [i for i in params["misc"] if i],
            "output_path": "/data/data_storage/miscellaneous"
        }

        # set ETL params
        etl = self._set_etl_params(params)

        # set MOP params
        mop = self._set_mop_params(params)

        return {
            "etl": etl,
            "mop": mop,
            "miscellaneous": miscellaneous
        }

    def _set_etl_params(self, params):
        ordinal_name = os.environ.get("ORDINAL_ID", "")
        if ordinal_name:
            ordinal_id = ordinal_name.split("-")[-1]
        else:
            ordinal_id = ""

        # initialize empty params
        etl = {
            "train": defaultdict(lambda: None),
            "val": defaultdict(lambda: None),
            "test": defaultdict(lambda: None),
            "predict": defaultdict(lambda: None),
            "download_type": "url",
            "etl_dir": params["etl_dir"] if "etl_dir" in params else None
        }

        modes = ["train", "val", "test", "predict"]
        for mode in modes:
            if "{}_main".format(mode) in params and "{}_dir".format(mode) in params:
                etl[mode]["metadata_url"] = params["{}_main".format(mode)]
                etl[mode]["multimedia_url"] = params["{}_dir".format(mode)]
                etl[mode]["multimedia_download_path"] = "/data_storage_worker/{}/downloaded/{}".format(ordinal_id, mode)
                etl[mode]["metadata_download_path"] = "/data/data_storage/metadata/{}_metadata.csv".format(mode)
                etl[mode]["partition_path"] = "/data/data_storage/partitions/{}".format(mode)
            if mode == "val":
                # save processed val data on shared storage
                etl[mode]["output_path"] = "/data/data_storage/post_etl/{}".format(mode)
            else:
                etl[mode]["output_path"] = "/data_storage_worker/{}/post_etl/{}".format(ordinal_id, mode)

        return etl

    def _set_mop_params(self, params):
        mop = {
            "metrics_storage_path": {},
            "models_dir": params["models_dir"] if "models_dir" in params else None,
            "output_dir": params["output_dir"] if "output_dir" in params else None,
            "test_output_path": "/data/metrics_storage/user_metrics/test",
            "checkpoint_storage_path": "/data/checkpoint_storage",
            "predict_output_path": "/data/data_storage/prediction_output"
        }

        # MOP Params
        mop["metrics_storage_path"]["tensorboard"] = "/data/metrics_storage/tensorboard"
        mop["metrics_storage_path"]["meta_metrics"] = "/data/metrics_storage/meta_metrics"
        mop["metrics_storage_path"]["user_metrics"] = "/data/metrics_storage/user_metrics"

        return mop
