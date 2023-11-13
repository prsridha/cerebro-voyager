#!/bin/bash

# install requirements for user repo and cerebro repo
if [ -f /user-repo/requirements.txt ]; then
    pip install -r /user-repo/requirements.txt
fi
if [ -f /cerebro-repo/cerebro-kube/requirements.txt ]; then
    pip install -r /cerebro-repo/cerebro-kube/requirements.txt
fi

# run etl worker
python3 /cerebro-repo/cerebro-kube/cerebro/etl/etl_worker.py --id $WORKER_ID_SELF
# sleep infinity

# check if command exited with error code
exit_code=$?
if [ $exit_code -eq 0 ]; then
  echo "ETL command complete"
else
  echo "ETL command exited with error"
fi
