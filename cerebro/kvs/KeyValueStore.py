import os
import dill
import json
import base64
import sqlite3
from kubernetes import client, config


class KeyValueStore:
    def __init__(self, init_tables=False):
        base_path = "/key_value_store/cerebro.db"
        self.conn = sqlite3.connect(base_path)
        self.cursor = self.conn.cursor()

        # create tables
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tables_file = os.path.join(current_dir, "tables.sql")
        with open(tables_file, 'r') as file:
            tables_script = file.read()
        self.cursor.executescript(tables_script)
        self.conn.commit()

        # get number of workers
        config.load_kube_config()
        v1 = client.CoreV1Api()
        username = os.environ['USERNAME']
        namespace = os.environ['NAMESPACE']
        cm1 = v1.read_namespaced_config_map(name='{}-cerebro-info'.format(username), namespace=namespace)
        self.num_workers = json.loads(cm1.data["data"])["num_nodes"]

        # initialize KVS with default values
        if init_tables:
            self.initialize_tables()

    def initialize_tables(self):
        num_workers = self.num_workers

        # set seed
        self.set_seed(0)

        # set error
        self.set_error("")

        # mark all nodes as healthy
        for w in range(num_workers):
            self.set_restarts(w)

        # set etl task and spec
        self.etl_set_task("", "")
        self.etl_set_spec("")

        # set etl worker progress to 0 for all workers, worker status to idle
        for w in range(num_workers):
            self.etl_set_worker_status(w, "")
            self.etl_set_worker_progress(w, 0)

        # # set mop task and spec
        # self.mop_set_spec("")
        # self.mop_set_task("")
        #
        # # set all mop worker statuses to empty
        # for w in range(num_workers):
        #     self.mop_set_worker_status(w, "")
        #
        # # mark as initialized
        # path = self.key_paths["kvs_init"]
        # with open(path, "w+") as f:
        #     json.dump("true", f)

    # dataset locators
    def set_dataset_locators(self, locators):
        query = """
            INSERT OR REPLACE
            INTO dataset_locators (id, params)
            VALUES (?, ?)
        """
        locators_str = json.dumps(locators)
        self.cursor.execute(query, (0, locators_str))
        self.conn.commit()

    def get_dataset_locators(self):
        query = """
            SELECT params
            FROM dataset_locators
            WHERE id = ?
        """
        self.cursor.execute(query, (0,))
        params = self.cursor.fetchone()[0]
        locators = json.loads(params)
        return locators

    # seed value for randomness
    def set_seed(self, seed):
        query = """
            INSERT OR REPLACE
            INTO seed (id, seed_val)
            VALUES (?, ?)
        """
        self.cursor.execute(query, (0, seed))
        self.conn.commit()

    def get_seed(self):
        query = """
            SELECT seed_val
            FROM seed
            WHERE id = ?
        """
        self.cursor.execute(query, (0,))
        seed = self.cursor.fetchone()[0]
        return seed

    # error on user's code
    def set_error(self, err):
        query = """
            INSERT OR REPLACE
            INTO debug_errors (id, error_message)
            VALUES (?, ?)
        """
        self.cursor.execute(query, (0, err))
        self.conn.commit()

    def get_error(self):
        query = """
            SELECT error_message
            FROM debug_errors
            WHERE id = ?
        """
        self.cursor.execute(query, (0,))
        err = self.cursor.fetchone()[0]
        return err

    # health of the cluster
    def set_restarts(self, worker_id):
        query = """
            INSERT OR REPLACE INTO restarts (worker_id, restart_count)
            VALUES (?, COALESCE((SELECT restart_count + 1 FROM restarts WHERE worker_id = ?), 0))
            """
        self.cursor.execute(query, (worker_id, worker_id))
        self.conn.commit()

    def get_restarts(self, worker_id):
        query = """
            SELECT restart_count
            FROM restarts
            WHERE worker_id = ?
        """
        self.cursor.execute(query, (worker_id,))
        count = self.cursor.fetchone()[0]
        return count

    # current etl task
    def etl_set_task(self, task, mode):
        query = """
            INSERT OR REPLACE
            INTO etl_task (id, task, mode)
            VALUES (?, ?, ?)
        """
        self.cursor.execute(query, (0, task, mode))
        self.conn.commit()

    def etl_get_task(self):
        query = """
            SELECT task, mode
            FROM etl_task
            WHERE id = ?
        """
        self.cursor.execute(query, (0,))
        task, mode = self.cursor.fetchone()
        return task, mode

    # save etl function strings
    def etl_set_spec(self, func):
        func_str = base64.b64encode(dill.dumps(func, byref=False)).decode("ascii")
        query = """
            INSERT OR REPLACE
            INTO etl_spec (id, spec_str)
            VALUES (?, ?)
        """
        self.cursor.execute(query, (0, func_str))
        self.conn.commit()

    def etl_get_spec(self):
        query = """
            SELECT spec_str
            FROM etl_spec
            WHERE id = ?
        """
        self.cursor.execute(query, (0,))
        func_str = self.cursor.fetchone()[0]
        func = dill.loads(base64.b64decode(func_str))
        return func

    # current etl-worker status
    def etl_set_worker_status(self, worker_id, status):
        query = """
            INSERT OR REPLACE
            INTO etl_worker_status (worker_id, worker_status)
            VALUES (?, ?)
        """
        self.cursor.execute(query, (worker_id, status))
        self.conn.commit()

    def etl_get_worker_status(self, worker_id):
        query = """
            SELECT worker_status
            FROM etl_worker_status
            WHERE worker_id = ?
        """
        self.cursor.execute(query, (worker_id,))
        status = self.cursor.fetchone()[0]
        return status

    # current etl-worker progress
    def etl_set_worker_progress(self, worker_id, progress):
        query = """
            INSERT OR REPLACE
            INTO etl_worker_progress (worker_id, worker_progress)
            VALUES (?, ?)
        """
        self.cursor.execute(query, (worker_id, progress))
        self.conn.commit()

    def etl_get_worker_progress(self, worker_id):
        query = """
            SELECT worker_progress
            FROM etl_worker_progress
            WHERE worker_id = ?
        """
        self.cursor.execute(query, (worker_id,))
        progress = self.cursor.fetchone()[0]
        return progress

    # save mop function strings
    def mop_set_spec(self, func):
        pass
        # func_str = base64.b64encode(dill.dumps(func, byref=False)).decode("ascii")
        # path = self.key_paths["mop_spec"]
        # with open(path, "w+") as f:
        #     json.dump(func_str, f)

    def mop_get_spec(self):
        pass
        # path = self.key_paths["mop_spec"]
        # with open(path, "r") as f:
        #     func_str = json.load(f)
        # func = dill.loads(base64.b64decode(func_str))
        # return func

    # current mop task
    def mop_set_task(self, value):
        pass
        # path = self.key_paths["mop_task"]
        # with open(path, "w+") as f:
        #     json.dump(value, f)

    def mop_get_task(self):
        pass
        # path = self.key_paths["mop_task"]
        # with open(path, "r") as f:
        #     val = json.load(f)
        # return val

    # model for each mop-worker to run
    def mop_set_model_on_worker(self, worker_id, epoch, model_id, is_last_worker):
        pass
        # path = os.path.join(self.key_paths["mop_model_on_worker"], str(worker_id) + ".json")
        # if os.path.isfile(path):
        #     with open(path, "r") as f:
        #         records = json.load(f)
        # else:
        #     records = list()
        # d = {
        #     "epoch": epoch,
        #     "model_id": model_id,
        #     "is_last_worker": is_last_worker,
        # }
        # records.append(d)

        # with open(path, "w+") as f:
        #     json.dump(records, f)

    def mop_get_models_on_worker(self, worker_id, latest_only=True):
        pass
        # path = os.path.join(self.key_paths["mop_model_on_worker"], str(worker_id) + ".json")
        # with open(path, "r") as f:
        #     prev = json.load(f)
        # if latest_only:
        #     d = prev[-1]
        #     d["epoch"] = int(d["epoch"])
        #     d["model_id"] = int(d["model_id"])
        #     return d
        # else:
        #     # TODO: fix this - convert to int / switch to pickle from json
        #     return prev

    # (model, parallelism) for each mop-worker to sample
    def mop_set_model_parallelism_on_worker(self, worker_id, mpl):
        pass
        # path = self.key_paths["mop_model_parallelism_on_worker"]
        # if os.path.isfile(path):
        #     with open(path, "r") as f:
        #         records = json.load(f)
        # else:
        #     records = {}
        # records[str(worker_id)] = mpl
        # with open(path, "w+") as f:
        #     json.dump(records, f)

    def mop_get_model_parallelism_on_worker(self, worker_id):
        pass
        # path = self.key_paths["mop_model_parallelism_on_worker"]
        # with open(path, "r") as f:
        #     records = json.load(f)
        # return records[str(worker_id)]

    # current mop-worker status
    def mop_set_worker_status(self, worker_id, status):
        pass
        # path = os.path.join(self.key_paths["mop_worker_status"], str(worker_id) + ".json")
        # with open(path, "w+") as f:
        #     json.dump(status, f)

    def mop_get_worker_status(self, worker_id):
        pass
        # path = os.path.join(self.key_paths["mop_worker_status"], str(worker_id) + ".json")
        # with open(path, "r") as f:
        #     status = json.load(f)
        # return status

    # model_id to model mapping
    def mop_set_model_mapping(self, model_map):
        pass
        # path = self.key_paths["mop_model_mapping"]
        # with open(path, "w+") as f:
        #     json.dump(model_map, f)

    def mop_get_model_mapping(self, model_id=None):
        pass
        # path = self.key_paths["mop_model_mapping"]
        # with open(path, "r") as f:
        #     model_map = json.load(f)
        # if model_id is not None:
        #     val = model_map[str(model_id)]
        #     return val
        # else:
        #     val = {int(key): value for key, value in model_map.items()}
        #     return val

    # model_id to parallelism mapping
    def mop_set_parallelism_mapping(self, model_id, parallelism):
        pass
        # path = self.key_paths["mop_parallelism_mapping"]
        # if os.path.isfile(path):
        #     with open(path, "r") as f:
        #         records = json.load(f)
        # else:
        #     records = {}
        # records[str(model_id)] = parallelism
        # with open(path, "w+") as f:
        #     json.dump(records, f)

    def mop_get_parallelism_mapping(self, model_id):
        pass
        # path = self.key_paths["mop_parallelism_mapping"]
        # with open(path, "r") as f:
        #     records = json.load(f)
        #     val = records[str(model_id)]
        # return val

    # temp save sample time
    def mop_set_sample_time(self, model_id, parallelism, time_taken):
        pass
        # path = os.path.join(self.key_paths["mop_sample_time"], str(model_id) + ".json")
        # mp = json.dumps((model_id, parallelism))
        # d = {mp: time_taken}
        # with open(path, "w+") as f:
        #     json.dump(d, f)

    def mop_get_sample_time(self, model_id, parallelism):
        pass
        # path = os.path.join(self.key_paths["mop_sample_time"], str(model_id) + ".json")
        # mp = json.dumps((model_id, parallelism))
        # with open(path, "r") as f:
        #     d = json.load(f)
        #     val = float(d[mp])
        # return val
