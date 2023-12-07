#!/bin/bash

# run tensorboard as a background process
nohup tensorboard --host=0.0.0.0 --logdir=$ML_METRICS_LOGDIR --port=6006 &

# copy notebook template to user dir
cp /cerebro-core/cerebro/misc/experiment.ipynb /user/experiment.ipynb

# start notebook in user dir
JUPYTER_TOKEN=$(echo "projectcerebro" | xxd -u -l 14 -p)
jupyter notebook --generate-config
sed -i "448s/.*/c.NotebookApp.notebook_dir = '\/user'/" $HOME/.jupyter/jupyter_notebook_config.py
jupyter nbextension enable --py widgetsnbextension
jupyter notebook --NotebookApp.token=$JUPYTER_TOKEN --NotebookApp.password=$JUPYTER_TOKEN --ip 0.0.0.0 --allow-root --no-browser