from cerebro.parallelisms.ddp import DDPExecutor
from cerebro.parallelisms.fsdp import FSDPExecutor
from cerebro.parallelisms.gpipe import GPipeExecutor

PARALLELISMS_LIST = [
    "FSDP",
    "DDP"
]


def get_parallelism_executor(name):
    if name == "FSDP":
        return FSDPExecutor
    elif name == "DDP":
        return DDPExecutor
    elif name == "GPIPE":
        return GPipeExecutor
