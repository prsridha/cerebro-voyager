import os
import dill
import json
import base64
from pathlib import Path
from kubernetes import client, config


class KeyValueStore:
    def __init__(self, init_tables=False):
        base_path = "/key_value_store"
        self.key_paths = {
            "kvs_init": os.path.join(base_path, "kvs_init.json"),
            "seed": os.path.join(base_path, "seed.json"),
            "error": os.path.join(base_path, "error.json"),
            "restarts": os.path.join(base_path, "restarts"),
            "dataset_locators": os.path.join(base_path, "dataset_locators.json"),
            "etl_task": os.path.join(base_path, "etl_task.json"),
            "etl_spec": os.path.join(base_path, "etl_spec.json"),
            "etl_worker_status": os.path.join(base_path, "etl_worker_status"),
            "etl_worker_progress": os.path.join(base_path, "etl_worker_progress"),
            "mop_spec": os.path.join(base_path, "mop_spec.json"),
            "mop_task": os.path.join(base_path, "mop_task.json"),
            "mop_sample_time": os.path.join(base_path, "mop_sample_time"),
            "mop_worker_status": os.path.join(base_path, "mop_worker_status"),
            "mop_model_mapping": os.path.join(base_path, "mop_model_mapping.json"),
            "mop_model_on_worker": os.path.join(base_path, "mop_model_on_worker"),
            "mop_parallelism_mapping": os.path.join(base_path, "mop_model_parallelism_mapping.json"),
            "mop_model_parallelism_on_worker": os.path.join(base_path, "mop_model_parallelism_on_worker.json"),
        }

        # create Kubernetes handle object
        config.load_kube_config()
        v1 = client.CoreV1Api()
        namespace = os.environ['NAMESPACE']

        # get number of workers
        cm = v1.read_namespaced_config_map(name='cerebro-node-hardware-info', namespace=namespace)
        node_info = json.loads(cm.data["data"])
        self.num_workers = len(node_info)

        # initialize KVS with default values
        if os.path.isfile(self.key_paths["kvs_init"]):
            with open(self.key_paths["kvs_init"]) as f:
                init_done = str(json.load(f)) == "true"
        else:
            init_done = False
        if init_tables and not init_done:
            self.initialize_tables()

    def initialize_tables(self):
        num_workers = self.num_workers

        # create required KVS directories
        Path(self.key_paths["restarts"]).mkdir(exist_ok=True)
        Path(self.key_paths["etl_worker_status"]).mkdir(exist_ok=True)
        Path(self.key_paths["etl_worker_progress"]).mkdir(exist_ok=True)
        Path(self.key_paths["mop_sample_time"]).mkdir(exist_ok=True)
        Path(self.key_paths["mop_worker_status"]).mkdir(exist_ok=True)
        Path(self.key_paths["mop_model_on_worker"]).mkdir(exist_ok=True)

        # set seed
        self.set_seed(0)

        # set error
        self.set_error("")

        # mark all nodes as healthy
        for w in range(num_workers):
            path = os.path.join(self.key_paths["restarts"], str(w) + ".json")
            with open(path, "w") as f:
                json.dump(0, f)

        # set etl task and spec
        self.etl_set_task("", "")
        self.etl_set_spec("")

        # set etl worker progress to 0 for all workers, worker status to idle
        for w in range(num_workers):
            self.etl_set_worker_status(w, "")
            self.etl_set_worker_progress(w, 0)

        # set mop task and spec
        self.mop_set_spec("")
        self.mop_set_task("")

        # set all mop worker statuses to empty
        for w in range(num_workers):
            self.mop_set_worker_status(w, "")

        # mark as initialized
        path = self.key_paths["kvs_init"]
        with open(path, "w+") as f:
            json.dump("true", f)

    # dataset locators
    def set_dataset_locators(self, locators):
        path = self.key_paths["dataset_locators"]
        with open(path, "w+") as f:
            json.dump(locators, f)

    def get_dataset_locators(self):
        path = self.key_paths["dataset_locators"]
        with open(path, "r") as f:
            val = json.load(f)
        return val

    # seed value for randomness
    def set_seed(self, seed):
        path = self.key_paths["seed"]
        with open(path, "w+") as f:
            json.dump(str(seed), f)

    def get_seed(self):
        path = self.key_paths["seed"]
        with open(path, "r") as f:
            val = int(json.load(f))
        return val

    # error on user's code
    def set_error(self, err):
        path = self.key_paths["error"]
        with open(path, "w+") as f:
            json.dump(err, f)

    def get_error(self):
        path = self.key_paths["error"]
        with open(path, "r") as f:
            val = json.load(f)
        return val

    # health of the cluster
    def set_restarts(self, worker_id):
        path = os.path.join(self.key_paths["restarts"], str(worker_id) + ".json")
        if os.path.isfile(path):
            with open(path, "r") as f:
                count = int(json.load(f))
        else:
            count = 0
        count += 1
        with open(path, "w+") as f:
            json.dump(str(count), f)

    def get_restarts(self, worker_id):
        path = os.path.join(self.key_paths["restarts"], str(worker_id) + ".json")
        with open(path, "r") as f:
            count = int(json.load(f))
        return count

    # current etl task
    def etl_set_task(self, task, mode):
        d = {"task": task, "mode": mode}
        path = self.key_paths["etl_task"]
        with open(path, "w+") as f:
            json.dump(d, f)

    def etl_get_task(self):
        path = self.key_paths["etl_task"]
        with open(path, "r") as f:
            d = json.load(f)
        task, mode = d["task"], d["mode"]
        return task, mode

    # save etl function strings
    def etl_set_spec(self, func):
        func_str = base64.b64encode(dill.dumps(func, byref=False)).decode("ascii")
        path = self.key_paths["etl_spec"]
        with open(path, "w+") as f:
            json.dump(func_str, f)

    def etl_get_spec(self):
        path = self.key_paths["etl_spec"]
        with open(path, "r") as f:
            func_str = json.load(f)
        func = dill.loads(base64.b64decode(func_str))
        return func

    # current etl-worker status
    def etl_set_worker_status(self, worker_id, status):
        path = os.path.join(self.key_paths["etl_worker_status"], str(worker_id) + ".json")
        with open(path, "w+") as f:
            json.dump(status, f)

    def etl_get_worker_status(self, worker_id):
        path = os.path.join(self.key_paths["etl_worker_status"], str(worker_id) + ".json")
        with open(path, "r") as f:
            val = json.load(f)
        return val

    # current etl-worker progress
    def etl_set_worker_progress(self, worker_id, progress):
        path = os.path.join(self.key_paths["etl_worker_progress"], str(worker_id) + ".json")
        with open(path, "w+") as f:
            json.dump(progress, f)

    def etl_get_worker_progress(self, worker_id):
        path = os.path.join(self.key_paths["etl_worker_progress"], str(worker_id) + ".json")
        with open(path, "r") as f:
            progress = float(json.load(f))
        return progress

    # save mop function strings
    def mop_set_spec(self, func):
        func_str = base64.b64encode(dill.dumps(func, byref=False)).decode("ascii")
        path = self.key_paths["mop_spec"]
        with open(path, "w+") as f:
            json.dump(func_str, f)

    def mop_get_spec(self):
        path = self.key_paths["mop_spec"]
        with open(path, "r") as f:
            func_str = json.load(f)
        func = dill.loads(base64.b64decode(func_str))
        return func

    # current mop task
    def mop_set_task(self, value):
        path = self.key_paths["mop_task"]
        with open(path, "w+") as f:
            json.dump(value, f)

    def mop_get_task(self):
        path = self.key_paths["mop_task"]
        with open(path, "r") as f:
            val = json.load(f)
        return val

    # model for each mop-worker to run
    def mop_set_model_on_worker(self, worker_id, epoch, model_id, is_last_worker):
        path = os.path.join(self.key_paths["mop_model_on_worker"], str(worker_id) + ".json")
        if os.path.isfile(path):
            with open(path, "r") as f:
                records = json.load(f)
        else:
            records = list()
        d = {
            "epoch": epoch,
            "model_id": model_id,
            "is_last_worker": is_last_worker,
        }
        records.append(d)

        with open(path, "w+") as f:
            json.dump(records, f)

    def mop_get_models_on_worker(self, worker_id, latest_only=True):
        path = os.path.join(self.key_paths["mop_model_on_worker"], str(worker_id) + ".json")
        with open(path, "r") as f:
            prev = json.load(f)
        if latest_only:
            d = prev[-1]
            d["epoch"] = int(d["epoch"])
            d["model_id"] = int(d["model_id"])
            return d
        else:
            # TODO: fix this - convert to int / switch to pickle from json
            return prev

    # (model, parallelism) for each mop-worker to sample
    def mop_set_model_parallelism_on_worker(self, worker_id, mpl):
        path = self.key_paths["mop_model_parallelism_on_worker"]
        if os.path.isfile(path):
            with open(path, "r") as f:
                records = json.load(f)
        else:
            records = {}
        records[str(worker_id)] = mpl
        with open(path, "w+") as f:
            json.dump(records, f)

    def mop_get_model_parallelism_on_worker(self, worker_id):
        path = self.key_paths["mop_model_parallelism_on_worker"]
        with open(path, "r") as f:
            records = json.load(f)
        return records[str(worker_id)]

    # current mop-worker status
    def mop_set_worker_status(self, worker_id, status):
        path = os.path.join(self.key_paths["mop_worker_status"], str(worker_id) + ".json")
        with open(path, "w+") as f:
            json.dump(status, f)

    def mop_get_worker_status(self, worker_id):
        path = os.path.join(self.key_paths["mop_worker_status"], str(worker_id) + ".json")
        with open(path, "r") as f:
            status = json.load(f)
        return status

    # model_id to model mapping
    def mop_set_model_mapping(self, model_map):
        path = self.key_paths["mop_model_mapping"]
        with open(path, "w+") as f:
            json.dump(model_map, f)

    def mop_get_model_mapping(self, model_id=None):
        path = self.key_paths["mop_model_mapping"]
        with open(path, "r") as f:
            model_map = json.load(f)
        if model_id is not None:
            val = model_map[str(model_id)]
            return val
        else:
            val = {int(key): value for key, value in model_map.items()}
            return val

    # model_id to parallelism mapping
    def mop_set_parallelism_mapping(self, model_id, parallelism):
        path = self.key_paths["mop_parallelism_mapping"]
        if os.path.isfile(path):
            with open(path, "r") as f:
                records = json.load(f)
        else:
            records = {}
        records[str(model_id)] = parallelism
        with open(path, "w+") as f:
            json.dump(records, f)

    def mop_get_parallelism_mapping(self, model_id):
        path = self.key_paths["mop_parallelism_mapping"]
        with open(path, "r") as f:
            records = json.load(f)
            val = records[str(model_id)]
        return val

    # temp save sample time
    def mop_set_sample_time(self, model_id, parallelism, time_taken):
        path = os.path.join(self.key_paths["mop_sample_time"], str(model_id) + ".json")
        mp = json.dumps((model_id, parallelism))
        d = {mp: time_taken}
        with open(path, "w+") as f:
            json.dump(d, f)

    def mop_get_sample_time(self, model_id, parallelism):
        path = os.path.join(self.key_paths["mop_sample_time"], str(model_id) + ".json")
        mp = json.dumps((model_id, parallelism))
        with open(path, "r") as f:
            d = json.load(f)
            val = float(d[mp])
        return val
