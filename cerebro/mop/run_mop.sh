#!/bin/bash

# install newly added requirements (if any) for cerebro-core
#if [ -f /cerebro-core/requirements.txt ]; then
#    pip install -r /cerebro-core/requirements.txt
#fi

# install requirements for user repo
if [ -f /user/requirements.txt ]; then
    pip install -r /user/requirements.txt
fi

# run mop worker
#python3 /cerebro-core/cerebro/mop/mop_worker.py
 sleep infinity

# check if command exited with error code
exit_code=$?
if [ $exit_code -eq 0 ]; then
  echo "MOP command complete"
else
  echo "MOP command exited with error"
fi
