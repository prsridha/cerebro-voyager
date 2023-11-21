#!/bin/bash

# run tensorboard as a background process
nohup tensorboard --host=0.0.0.0 --logdir=$ML_METRICS_LOGDIR --port=6006 &

# start notebook in user dir
JUPYTER_TOKEN=$(echo "projectcerebro" | xxd -u -l 14 -p)
jupyter notebook --generate-config
sed -i "448s/.*/c.NotebookApp.notebook_dir = '\/user'/" /root/.jupyter/jupyter_notebook_config.py
jupyter nbextension enable --py widgetsnbextension
jupyter notebook --NotebookApp.token=$JUPYTER_TOKEN --NotebookApp.password=$JUPYTER_TOKEN --ip 0.0.0.0 --allow-root --no-browser