#!/bin/bash

# install newly added requirements (if any) for cerebro-voyager
#if [ -f /cerebro-voyager/requirements.txt ]; then
#    pip install -r /cerebro-voyager/requirements.txt
#fi

# install requirements for user repo
if [ -f /user/requirements.txt ]; then
    pip install -r /user/requirements.txt
fi

# run etl worker
python3 /cerebro-voyager/cerebro/etl/etl_worker.py
# sleep infinity

# check if command exited with error code
exit_code=$?
if [ $exit_code -eq 0 ]; then
  echo "ETL command complete"
else
  echo "ETL command exited with error"
fi
