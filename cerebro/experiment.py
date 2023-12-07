import os
import json
import subprocess
from pathlib import Path
import ipywidgets as widgets
from kubernetes import client, config
from IPython.display import display, Javascript

from cerebro.util.params import Params
from cerebro.util.voyager_io import VoyagerIO
from cerebro.kvs.KeyValueStore import KeyValueStore
from cerebro.etl.etl_controller import ETLController
from cerebro.mop.mop_controller import MOPController
from cerebro.util.cerebro_logger import CerebroLogger


def run(cmd, shell=True, capture_output=True, text=True, halt_exception=True):
    try:
        out = subprocess.run(cmd, shell=shell, capture_output=capture_output, text=text)
        # print(cmd)
        if out.stderr:
            if halt_exception:
                raise Exception("Command Error:" + str(out.stderr))
            else:
                print("Command Error:" + str(out.stderr))
        if capture_output:
            return out.stdout.rstrip("\n")
        else:
            return None
    except Exception as e:
        print("Command Unsuccessful:", cmd)
        print(str(e))
        raise Exception


def display_buttons(url_data):
    out = widgets.Output()

    def _window_open_button(url):
        display(Javascript(f'window.open("{url.tooltip}");'))

    button1 = widgets.Button(description=url_data[0][0], tooltip=url_data[0][1], layout=widgets.Layout(width='200px'))
    button1.on_click(_window_open_button)

    # button2 = widgets.Button(description=url_data[1][0], tooltip=url_data[1][1], layout=widgets.Layout(width='200px'))
    # button2.on_click(_window_open_button)

    buttons_container = widgets.HBox([button1])
    display(buttons_container, out)


class Experiment:
    logging = CerebroLogger("controller")
    logger = logging.create_logger("experiment")

    def __init__(self, cli_params=None):
        # initialize key value store object
        self.mop = None
        self.num_epochs = None
        self.param_grid = None
        self.sub_epoch_spec = None
        self.kvs = KeyValueStore()

        # create controller objects
        self.etl = ETLController()
        self.mop = MOPController()

        # load values from cerebro-info configmap
        namespace = os.environ['NAMESPACE']
        username = os.environ['USERNAME']
        config.load_kube_config()
        v1 = client.CoreV1Api()
        cm = v1.read_namespaced_config_map(name='{}-cerebro-info'.format(username), namespace=namespace)
        cm_data = json.loads(cm.data["data"])
        self.user_code_path = cm_data["user_code_path"]
        self.tensorboard_port = cm_data["tensorboard_node_port"]

        self.logger.info("Starting Cerebro session...")

        if cli_params:
            self.initialize_via_cli(cli_params)

        self.params = Params()
        self.download_misc_files()

    def initialize_via_cli(self, params):
        if os.path.isfile(os.path.join(self.user_code_path, "requirements.txt")):
            run("pip install -r {}".format(os.path.join(self.user_code_path, "requirements.txt")))
            self.logger.info("Installed user's python dependencies")

        # save params
        self.kvs.set_dataset_locators(params)

        # get URLs
        tensorboard_url = "http://localhost:{}".format(self.tensorboard_port)
        url_data = [("Tensorboard Dashboard", tensorboard_url)]
        display_buttons(url_data)

        self.logger.info("Initialized via CLI")

    def download_misc_files(self):
        file_io = VoyagerIO()

        Path(self.params.miscellaneous["output_path"]).mkdir(parents=True, exist_ok=True)
        for from_path in self.params.miscellaneous["download_paths"]:
            filename = from_path.split("/")[-1]
            to_path = os.path.join(self.params.miscellaneous["output_path"], filename)
            exclude = os.path.dirname(from_path)
            file_io.download(to_path, from_path, exclude)

        print("Downloaded miscellaneous files")
        self.logger.info("Downloaded miscellaneous files")

    def run_etl(self, etl_spec=None, fraction=1, seed=0):
        # set seed value in KVS
        self.kvs.set_seed(seed)

        self.etl.initialize_controller(etl_spec, fraction)
        self.etl.run_etl()

        print("ETL complete")
        self.etl.exit_etl()

    def run_fit(self, sub_epoch_spec, param_grid, num_epochs, seed=0, save_artifacts=True):
        # set seed value in KVS
        self.kvs.set_seed(seed)

        # save values in class variables
        self.num_epochs = num_epochs
        self.param_grid = param_grid
        self.sub_epoch_spec = sub_epoch_spec

        self.mop.initialize_controller(num_epochs, param_grid, sub_epoch_spec)

        # start grid search
        self.logger.info("Starting grid search...")
        self.mop.grid_search()
        self.logger.info("Model selection complete")

        # save metrics to S3
        if save_artifacts:
            self.mop.save_metrics_s3()

    def run_test(self, model_tag, batch_size, output_filename, sub_epoch_spec=None):
        if sub_epoch_spec:
            self.mop.initialize_controller(0, None, sub_epoch_spec)

        if self.params.mop["models_dir"]:
            self.mop.download_models()

        self.mop.testing(model_tag, batch_size, output_filename)

    def run_predict(self, model_tag, batch_size, output_filename, sub_epoch_spec=None):
        if sub_epoch_spec:
            self.mop.initialize_controller(0, None, sub_epoch_spec)

        if self.params.mop["models_dir"]:
            self.mop.download_models()

        self.mop.prediction(model_tag, batch_size, output_filename)
