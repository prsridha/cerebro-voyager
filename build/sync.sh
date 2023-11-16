#!/bin/bash

if [[ "$PLATFORM" == "prod" ]]; then
    echo "Platform set to Production"
    mkdir -p /cerebro-core
    cp -r /home/cerebro-core/* /cerebro-core/
    pip install -v -e /cerebro-core
elif [[ "$PLATFORM" == "dev" ]]; then
    echo "Platform set to dev"
    if [ -z "${GIT_SYNC_BRANCH}" ]; then
        git clone $GIT_SYNC_REPO /cerebro-core
    else
        git clone $GIT_SYNC_REPO -b $GIT_SYNC_BRANCH /cerebro-core
    fi
    clone_status=$?
    if [ $clone_status -eq 0 ]; then
        echo "Successfully cloned the repository."
    else
        echo "An error occurred while running git clone."
    fi
else
    echo "Unknown platform"
fi
