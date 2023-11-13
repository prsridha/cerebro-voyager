#!/bin/bash

# TODO: remove this - needed for quick restarts for debugging
##########################################################################
# also remove git secrets env and volume mount from yaml
mkdir $HOME/.ssh
cp /etc/git-secret/* $HOME/.ssh/
mv $HOME/.ssh/ssh $HOME/.ssh/id_rsa.git
touch $HOME/.ssh/config

# create git config file in .ssh
echo "
Host github.com
    Hostname github.com
    IdentityFile $HOME/.ssh/id_rsa.git
    IdentitiesOnly yes
" > $HOME/.ssh/config

git config --global --add safe.directory /cerebro-repo/cerebro-kube
##########################################################################

# run tensorboard as a background process
nohup tensorboard --host=0.0.0.0 --logdir=$ML_METRICS_LOGDIR --port=6006 &

# start notebook in user-repo dir
JUPYTER_TOKEN=$(echo "projectcerebro" | xxd -u -l 14 -p)
jupyter notebook --generate-config
sed -i "448s/.*/c.NotebookApp.notebook_dir = '\/user-repo'/" /root/.jupyter/jupyter_notebook_config.py
jupyter nbextension enable --py widgetsnbextension
jupyter notebook --NotebookApp.token=$JUPYTER_TOKEN --NotebookApp.password=$JUPYTER_TOKEN --ip 0.0.0.0 --allow-root --no-browser