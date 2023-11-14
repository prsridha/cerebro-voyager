import os
import logging
from pathlib import Path


class CerebroLogger:
    def __init__(self, filename):
        log_root_path = '/cerebro-repo/cerebro-kube/logs'
        Path(log_root_path).mkdir(parents=True, exist_ok=True)

        log_format = '%(levelname)5s: %(name)5s - %(message)10s [%(filename)5s:%(lineno)d @ %(asctime)s]'
        logging.basicConfig(level=logging.INFO,
                            format=log_format,
                            filename=os.path.join(log_root_path, filename + ".log"),
                            filemode='a+')
        self.console = logging.StreamHandler()
        self.console.setLevel(logging.INFO)
        self.console.setFormatter(logging.Formatter(log_format))

    def create_logger(self, name):
        return logging.getLogger(name)