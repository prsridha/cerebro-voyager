import os
import dill
import time
import json
import base64
import sqlite3

from kubernetes import client, config


class KeyValueStore:
    def __init__(self, init_tables=False):
        base_path = "/key_value_store/cerebro.db"
        self.conn = sqlite3.connect(base_path, timeout=5000)

        # initialize KVS with default values
        if init_tables:
            self.initialize_tables()

    def initialize_tables(self):
        # get number of workers
        try:
            config.load_config()
        except config.config_exception.ConfigException as e:
            time.sleep(1)
            config.load_config()
        v1 = client.CoreV1Api()
        username = os.environ['USERNAME']
        namespace = os.environ['NAMESPACE']
        cm1 = v1.read_namespaced_config_map(name='{}-cerebro-info'.format(username), namespace=namespace)
        num_workers = json.loads(cm1.data["data"])["num_workers"]

        # create tables
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tables_file = os.path.join(current_dir, "tables.sql")
        with open(tables_file, 'r') as file:
            tables_script = file.read()
        cursor = self.conn.cursor()
        cursor.executescript(tables_script)
        cursor.close()
        self.conn.commit()

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

        # set mop task and spec
        self.mop_set_spec("")
        # self.mop_set_task("")

        # set all mop worker statuses to empty
        for w in range(num_workers):
            self.mop_set_worker_status(w, "")

    # dataset locators
    def set_dataset_locators(self, locators):
        query = """
            INSERT OR REPLACE
            INTO dataset_locators (id, params)
            VALUES (?, ?)
        """
        locators_str = json.dumps(locators)
        cursor = self.conn.cursor()
        cursor.execute(query, (0, locators_str))
        cursor.close()
        self.conn.commit()

    def get_dataset_locators(self):
        query = """
            SELECT params
            FROM dataset_locators
            WHERE id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0,))
        params = cursor.fetchone()[0]
        cursor.close()
        locators = json.loads(params)
        return locators

    # seed value for randomness
    def set_seed(self, seed):
        query = """
            INSERT OR REPLACE
            INTO seed (id, seed_val)
            VALUES (?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0, seed))
        cursor.close()
        self.conn.commit()

    def get_seed(self):
        query = """
            SELECT seed_val
            FROM seed
            WHERE id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0,))
        seed = cursor.fetchone()[0]
        cursor.close()
        return seed

    # error on user's code
    def set_error(self, err):
        query = """
            INSERT OR REPLACE
            INTO debug_errors (id, error_message)
            VALUES (?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0, err))
        cursor.close()
        self.conn.commit()

    def get_error(self):
        query = """
            SELECT error_message
            FROM debug_errors
            WHERE id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0,))
        err = cursor.fetchone()[0]
        cursor.close()
        return err

    # health of the cluster
    def set_restarts(self, worker_id):
        query = """
            INSERT OR REPLACE INTO restarts (worker_id, restart_count)
            VALUES (?, COALESCE((SELECT restart_count + 1 FROM restarts WHERE worker_id = ?), 0))
            """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id, worker_id))
        cursor.close()
        self.conn.commit()

    def get_restarts(self, worker_id):
        query = """
            SELECT restart_count
            FROM restarts
            WHERE worker_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id,))
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    # current etl task
    def etl_set_task(self, task, mode):
        query = """
            INSERT OR REPLACE
            INTO etl_task (id, task, mode)
            VALUES (?, ?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0, task, mode))
        cursor.close()
        self.conn.commit()

    def etl_get_task(self):
        query = """
            SELECT task, mode
            FROM etl_task
            WHERE id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0,))
        task, mode = cursor.fetchone()
        cursor.close()
        return task, mode

    # save etl function strings
    def etl_set_spec(self, func):
        func_str = base64.b64encode(dill.dumps(func, byref=False)).decode("ascii")
        query = """
            INSERT OR REPLACE
            INTO etl_spec (id, spec_str)
            VALUES (?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0, func_str))
        cursor.close()
        self.conn.commit()

    def etl_get_spec(self):
        query = """
            SELECT spec_str
            FROM etl_spec
            WHERE id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0,))
        func_str = cursor.fetchone()[0]
        cursor.close()
        func = dill.loads(base64.b64decode(func_str))
        return func

    # current etl-worker status
    def etl_set_worker_status(self, worker_id, status):
        query = """
            INSERT OR REPLACE
            INTO etl_worker_status (worker_id, worker_status)
            VALUES (?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id, status))
        cursor.close()
        self.conn.commit()

    def etl_get_worker_status(self, worker_id):
        query = """
            SELECT worker_status
            FROM etl_worker_status
            WHERE worker_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id,))
        status = cursor.fetchone()[0]
        cursor.close()
        return status

    # current etl-worker progress
    def etl_set_worker_progress(self, worker_id, progress):
        query = """
            INSERT OR REPLACE
            INTO etl_worker_progress (worker_id, worker_progress)
            VALUES (?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id, progress))
        cursor.close()
        self.conn.commit()

    def etl_get_worker_progress(self, worker_id):
        query = """
            SELECT worker_progress
            FROM etl_worker_progress
            WHERE worker_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id,))
        progress = cursor.fetchone()[0]
        cursor.close()
        return progress

    # save mop function strings
    def mop_set_spec(self, func):
        func_str = base64.b64encode(dill.dumps(func, byref=False)).decode("ascii")
        query = """
            INSERT OR REPLACE
            INTO mop_spec (id, spec_str)
            VALUES (?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0, func_str))
        cursor.close()
        self.conn.commit()

    def mop_get_spec(self):
        query = """
            SELECT spec_str
            FROM mop_spec
            WHERE id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0,))
        func_str = cursor.fetchone()[0]
        cursor.close()
        func = dill.loads(base64.b64decode(func_str))
        return func

    # current mop task
    def mop_set_task(self, task, worker_id, task_id):
        query = """
            INSERT OR REPLACE
            INTO mop_task (worker_id, task_id, task)
            VALUES (?, ?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id, task_id, task))
        cursor.close()
        self.conn.commit()

    def mop_get_task(self, worker_id):
        query = """
            SELECT task_id, task
            FROM mop_task
            WHERE worker_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id,))
        task_id, task = cursor.fetchone()
        cursor.close()
        return task_id, task

    # model for each mop-worker to run
    def mop_set_model_on_worker(self, worker_id, epoch, model_id, is_last_worker, is_last_epoch):
        query = """
            INSERT OR REPLACE
            INTO mop_model_on_worker (worker_id, epoch, model_id, is_last_worker, is_last_epoch)
            VALUES (?, ?, ?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id, epoch, model_id, is_last_worker, is_last_epoch))
        cursor.close()
        self.conn.commit()

    def mop_get_models_on_worker(self, worker_id):
        query = """
            SELECT epoch, model_id, is_last_worker, is_last_epoch
            FROM mop_model_on_worker
            WHERE worker_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id,))
        epoch, model_id, is_last_worker, is_last_epoch = cursor.fetchone()
        cursor.close()
        d = {
            "epoch": epoch,
            "model_id": model_id,
            "is_last_worker": is_last_worker,
            "is_last_epoch": is_last_epoch
        }
        return d

    # (model, parallelism) for each mop-worker to sample
    def mop_set_model_parallelism_on_worker(self, worker_id, model_id, parallelism):
        query = """
            INSERT OR REPLACE
            INTO mop_model_parallelism_on_worker (worker_id, model_id, parallelism)
            VALUES (?, ?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id, model_id, parallelism))
        cursor.close()
        self.conn.commit()

    def mop_get_model_parallelism_on_worker(self, worker_id):
        query = """
            SELECT model_id, parallelism
            FROM mop_model_parallelism_on_worker
            WHERE worker_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id,))
        model_id, parallelism = cursor.fetchone()
        cursor.close()
        return model_id, parallelism

    # current mop-worker status
    def mop_set_worker_status(self, worker_id, status):
        query = """
            INSERT OR REPLACE
            INTO mop_worker_status (worker_id, worker_status)
            VALUES (?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id, status))
        cursor.close()
        self.conn.commit()

    def mop_get_worker_status(self, worker_id):
        query = """
            SELECT worker_status
            FROM mop_worker_status
            WHERE worker_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id,))
        status = cursor.fetchone()[0]
        cursor.close()
        return status

    def mop_set_worker_progress(self, worker_id, progress):
        query = """
            INSERT OR REPLACE
            INTO mop_worker_progress (worker_id, worker_progress)
            VALUES (?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id, progress))
        cursor.close()
        self.conn.commit()

    def mop_get_worker_progress(self, worker_id):
        query = """
            SELECT worker_progress
            FROM mop_worker_progress
            WHERE worker_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (worker_id,))
        progress = cursor.fetchone()[0]
        cursor.close()
        return progress

    # model_id to model mapping
    def mop_set_model_mapping(self, model_map):
        cursor = self.conn.cursor()
        for model_id, model_config in model_map.items():
            query = """
                INSERT OR REPLACE
                INTO mop_model_mapping (model_id, model_config)
                VALUES (?, ?)
            """
            cursor.execute(query, (model_id, json.dumps(model_config)))
        cursor.close()
        self.conn.commit()

    def mop_get_model_mapping(self, model_id=None):
        cursor = self.conn.cursor()
        if model_id is not None:
            query = """
                SELECT model_config
                FROM mop_model_mapping
                WHERE model_id = ?
            """
            cursor.execute(query, (model_id,))
            row = cursor.fetchone()[0]
            cursor.close()
            return json.loads(row)
        else:
            query = """
                SELECT *
                FROM mop_model_mapping
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            data_dict = [json.loads(d) for d in rows]
            return data_dict

    # model_id to parallelism mapping
    def mop_set_parallelism_mapping(self, model_id, parallelism):
        query = """
            INSERT OR REPLACE
            INTO mop_parallelism_mapping (model_id, parallelism)
            VALUES (?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (model_id, parallelism))
        cursor.close()
        self.conn.commit()

    def mop_get_parallelism_mapping(self, model_id):
        query = """
            SELECT parallelism
            FROM mop_parallelism_mapping
            WHERE model_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (model_id,))
        parallelism = cursor.fetchone()[0]
        cursor.close()
        return parallelism

    # temp save sample time
    def mop_set_sample_time(self, model_id, parallelism, time_taken):
        query = """
            INSERT OR REPLACE
            INTO mop_sample_time (model_id, parallelism, time_taken)
            VALUES (?, ?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (model_id, parallelism, time_taken))
        cursor.close()
        self.conn.commit()

    def mop_get_sample_time(self, model_id, parallelism):
        query = """
            SELECT time_taken
            FROM mop_sample_time
            WHERE model_id = ? AND parallelism = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (model_id, parallelism))
        time_taken = cursor.fetchone()[0]
        cursor.close()
        return time_taken

    def mop_set_test_params(self, model_tag, batch_size):
        query = """
            INSERT OR REPLACE
            INTO mop_test_params (id, model_tag, batch_size)
            VALUES (?, ?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0, model_tag, batch_size))
        cursor.close()
        self.conn.commit()

    def mop_get_test_params(self):
        query = """
            SELECT model_tag, batch_size
            FROM mop_test_params
            WHERE id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (0, ))
        model_tag, batch_size = cursor.fetchone()
        cursor.close()
        return model_tag, batch_size

    def mop_set_predict_params(self, model_tag, batch_size):
        query = """
                    INSERT OR REPLACE
                    INTO mop_predict_params (id, model_tag, batch_size)
                    VALUES (?, ?, ?)
                """
        cursor = self.conn.cursor()
        cursor.execute(query, (0, model_tag, batch_size))
        cursor.close()
        self.conn.commit()

    def mop_get_predict_params(self):
        query = """
                    SELECT model_tag, batch_size
                    FROM mop_predict_params
                    WHERE id = ?
                """
        cursor = self.conn.cursor()
        cursor.execute(query, (0,))
        model_tag, batch_size = cursor.fetchone()
        cursor.close()
        return model_tag, batch_size
