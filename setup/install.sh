#!/bin/bash

# for OpenSSL issue with Kubernetes
pip install pip --upgrade
pip install pyopenssl --upgrade

# add Cerebro installer's dependencies
pip install -r requirements.txt --user

echo "Installed Cerebro dependencies.\n"