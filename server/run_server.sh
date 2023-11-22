#!/bin/bash

# create and navigate directories
mkdir -p /cerebro-core/logs
cd /cerebro-core/server

# run backend server
flask run --host=0.0.0.0 -p 8083  2>&1 |tee /cerebro-core/logs/server_logs.log