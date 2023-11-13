import re
import os
import json
from collections import defaultdict
from cerebro.kvs.KeyValueStore import KeyValueStore


class Params:
    _instance = None

    # "singleton" class
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Params, cls).__new__(cls)
            cls._instance._load_params()
        return cls._instance

    def _load_params(self):
        if os.path.isfile('/cerebro-repo/cerebro-kube/params.json'):
            with open('/cerebro-repo/cerebro-kube/params.json', 'r') as file:
                config = json.load(file)
        else:
            # load from KVS and then save to params.json file
            config = self._load_from_kvs()
            with open('/cerebro-repo/cerebro-kube/params.json', 'w') as file:
                json.dump(config, file, indent=4)

        # assign all variables to the class
        self.etl = config.get('etl')
        self.mop = config.get('mop')
        self.bucket_name = config.get('bucket_name')
        self.miscellaneous = config.get('miscellaneous')

    def _load_from_kvs(self):
        kvs = KeyValueStore()
        params = kvs.get_dataset_locators()

        # get bucket name
        s3_url_pattern = r"'s3:\/\/.*?'"
        match = re.search(s3_url_pattern, str(params))
        s3_url = match.group().replace("'", "") if match else None
        bucket_name = s3_url.split("/")[2]

        # set misc params
        miscellaneous = {
            "download_paths": [i for i in params["misc"] if i],
            "output_path": "/data/cerebro_data_storage/miscellaneous"
        }

        # set ETL params
        etl = self._set_etl_params(params)

        # set MOP params
        mop = self._set_mop_params(params)

        return {
            "etl": etl,
            "mop": mop,
            "bucket_name": bucket_name,
            "miscellaneous": miscellaneous
        }

    def _set_etl_params(self, params):
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
                etl[mode]["multimedia_download_path"] = "/cerebro_data_storage_worker/downloaded/{}".format(mode)
                etl[mode]["metadata_download_path"] = "/data/cerebro_data_storage/metadata/{}_metadata.csv".format(mode)
                etl[mode]["partition_path"] = "/data/cerebro_data_storage/partitions/{}".format(mode)
            if mode == "val":
                # save processed val data on shared storage
                etl[mode]["output_path"] = "/data/cerebro_data_storage/post_etl/{}".format(mode)
            else:
                etl[mode]["output_path"] = "/cerebro_data_storage_worker/post_etl/{}".format(mode)

        return etl

    def _set_mop_params(self, params):
        mop = {
            "metrics_storage_path": {},
            "models_dir": params["models_dir"] if "models_dir" in params else None,
            "output_dir": params["output_dir"] if "output_dir" in params else None,
            "test_output_path": "/data/cerebro_data_storage/test_output",
            "checkpoint_storage_path": "/data/cerebro_checkpoint_storage",
            "prediction_output_path": "/data/cerebro_data_storage/prediction_output"
        }

        # MOP Params
        mop["metrics_storage_path"]["tensorboard"] = "/data/cerebro_metrics_storage/tensorboard"
        mop["metrics_storage_path"]["meta_metrics"] = "/data/cerebro_metrics_storage/meta_metrics"
        mop["metrics_storage_path"]["user_metrics"] = "/data/cerebro_metrics_storage/user_metrics"

        return mop
