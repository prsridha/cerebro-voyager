import os
import dill
import json
import redis
import base64
from kubernetes import client, config


class KeyValueStore:
    def __init__(self, init_tables=False):
        # get username and namespace
        # config.load_incluster_config()
        config.load_kube_config()
        v1 = client.CoreV1Api()
        namespace = os.environ['NAMESPACE']
        username = os.environ['USERNAME']
        cm = v1.read_namespaced_config_map(name='{}-cerebro-info'.format(username), namespace=namespace)
        cm_data = json.loads(cm.data["data"])
        username = cm_data["username"]

        # create redis handle object
        host = "{}-redis-master.{}.svc.cluster.local".format(username, namespace)
        port = 6379
        passwrd = "cerebro"
        r = redis.Redis(host, port, decode_responses=True, password=passwrd)
        self.r = r

        # get number of workers
        cm = v1.read_namespaced_config_map(name='cerebro-node-hardware-info', namespace=namespace)
        node_info = json.loads(cm.data["data"])
        self.num_workers = len(node_info)

        # initialize KVS with default values
        init_done = r.get("kvs_init") == "true"
        if init_tables and not init_done:
            self.initialize_tables()

    def initialize_tables(self):
        r = self.r
        num_workers = self.num_workers

        r.set("etl_task", json.dumps(["", ""]))

        # set empty dataset locators
        r.set("dataset_locators", "")

        # mark all nodes as healthy
        for w in range(num_workers):
            r.hset("restarts", str(w), 0)

        # set etl worker progress to 0 for all workers
        for w in range(num_workers):
            r.hset("etl_worker_progress", str(w), 0)

        # set etl worker status to idle for all tasks for all workers
        for w in range(num_workers):
            r.hset("etl_worker_status", str(w), json.dumps(list()))

        r.set("mop_task", "")

        # set all mop worker statuses to empty
        for w in range(num_workers):
            r.hset("mop_worker_status", str(w), "")

        # set all mop model on workers to empty
        for w in range(num_workers):
            r.hset("mop_model_on_worker", str(w), json.dumps(list()))

        # mark as initialized
        r.set("kvs_init", "true")

        # set error value as empty
        r.set("error", "")

    # dataset locators
    def set_dataset_locators(self, locators):
        d = json.dumps(locators)
        res = self.r.set("dataset_locators", d)
        return res

    def get_dataset_locators(self):
        val = json.loads(self.r.get("dataset_locators"))
        return val

    # seed value for randomness
    def set_seed(self, seed):
        res = self.r.set("seed", str(seed))
        return res

    def get_seed(self):
        val = self.r.get("seed")
        return int(val)

    # error on user's code
    def set_error(self, err):
        self.r.set("etl_task", "")
        res = self.r.set("error", str(err))
        return res

    def get_error(self):
        val = self.r.get("error")
        return val

    # health of the cluster
    def set_restarts(self, worker_id):
        count = int(self.r.hget("restarts", str(worker_id)))
        res = self.r.hset("restarts", str(worker_id), str(count + 1))
        return res

    def get_restarts(self, worker_id):
        val = self.r.hget("restarts", str(worker_id))
        return int(val)

    # current etl task
    def etl_set_task(self, task, mode):
        res = self.r.set("etl_task", json.dumps([task, mode]))
        return res

    def etl_get_task(self):
        task, mode = json.loads(self.r.get("etl_task"))
        return task, mode

    # save etl function strings
    def etl_set_spec(self, func):
        func_str = base64.b64encode(dill.dumps(func, byref=False)).decode("ascii")
        res = self.r.set("etl_spec", func_str)
        return res

    def etl_get_spec(self):
        func_str = self.r.get("etl_spec")
        func = dill.loads(base64.b64decode(func_str))
        return func

    # current etl-worker status
    def etl_set_worker_status(self, worker_id, status):
        res = self.r.hset("etl_worker_status", str(worker_id), status)
        return res

    def etl_get_worker_status(self, worker_id):
        val = self.r.hget("etl_worker_status", str(worker_id))
        return val

    # current etl-worker progress
    def etl_set_worker_progress(self, worker_id, progress):
        res = self.r.hset("etl_worker_progress", str(worker_id), progress)
        return res

    def etl_get_worker_progress(self, worker_id):
        val = self.r.hget("etl_worker_progress", str(worker_id))
        return float(val)

    # save mop function strings
    def mop_set_spec(self, func):
        func_str = base64.b64encode(dill.dumps(func, byref=False)).decode("ascii")
        res = self.r.set("mop_spec", func_str)
        return res

    def mop_get_spec(self):
        func_str = self.r.get("mop_spec")
        func = dill.loads(base64.b64decode(func_str))
        return func

    # current mop task
    def mop_set_task(self, value):
        res = self.r.set("mop_task", value)
        return res

    def mop_get_task(self):
        tasks = self.r.get("mop_task")
        return tasks

    # model for each mop-worker to run
    def mop_set_model_on_worker(self, worker_id, epoch, model_id, is_last_worker):
        records = json.loads(self.r.hget("mop_model_on_worker", str(worker_id)))
        d = {
            "epoch": epoch,
            "model_id": model_id,
            "is_last_worker": is_last_worker,
        }
        records.append(d)
        res = self.r.hset("mop_model_on_worker", str(worker_id), json.dumps(records))
        return res

    def mop_get_models_on_worker(self, worker_id, latest_only=True):
        prev = json.loads(self.r.hget("mop_model_on_worker", str(worker_id)))
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
        d = json.dumps(mpl)
        res = self.r.hset("mop_model_parallelism_on_worker", str(worker_id), d)
        return res

    def mop_get_model_parallelism_on_worker(self, worker_id):
        val = json.loads(self.r.hget("mop_model_parallelism_on_worker", str(worker_id)))
        return val

    # current mop-worker status
    def mop_set_worker_status(self, worker_id, status):
        res = self.r.hset("mop_worker_status", str(worker_id), status)
        return res

    def mop_get_worker_status(self, worker_id):
        val = self.r.hget("mop_worker_status", str(worker_id))
        return val

    # model_id to model mapping
    def mop_set_model_mapping(self, model_map):
        res = self.r.set("mop_model_mapping", json.dumps(model_map))
        return res

    def mop_get_model_mapping(self, model_id=None):
        d = json.loads(self.r.get("mop_model_mapping"))
        if model_id is not None:
            val = d[str(model_id)]
            return val
        else:
            val = {int(key): value for key, value in d.items()}
            return val

    # model_id to parallelism mapping
    def mop_set_parallelism_mapping(self, model_id, parallelism):
        res = self.r.hset("mop_parallelism_mapping", str(model_id), parallelism)
        return res

    def mop_get_parallelism_mapping(self, model_id):
        val = self.r.hget("mop_parallelism_mapping", str(model_id))
        return val

    # temp save sample time
    def mop_set_sample_time(self, model_id, parallelism, time_taken):
        mp = json.dumps((model_id, parallelism))
        res = self.r.hset("mop_sample_time", mp, str(time_taken))
        return res

    def mop_get_sample_time(self, model_id, parallelism):
        mp = json.dumps((model_id, parallelism))
        val = self.r.hget("mop_sample_time", mp)
        return float(val)
