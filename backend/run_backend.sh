#!/bin/bash

# create and navigate directories
mkdir -p /cerebro-repo/cerebro-kube/logs
cd /cerebro-repo/cerebro-kube/backend

# run backend server
flask run --host=0.0.0.0 -p 8083  2>&1 |tee /cerebro-repo/cerebro-kube/logs/backend_logs.log