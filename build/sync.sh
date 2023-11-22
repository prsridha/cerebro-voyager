#!/bin/bash

if [[ "$PLATFORM" == "prod" ]]; then
    echo "Platform set to Production"
elif [[ "$PLATFORM" == "dev" ]]; then
    echo "Platform set to dev"
    rm -rf /cerebro-core/*
    if [ -z "${GIT_SYNC_BRANCH}" ]; then
        git clone $GIT_SYNC_REPO /cerebro-core
    else
        git clone $GIT_SYNC_REPO -b $GIT_SYNC_BRANCH /cerebro-core
    fi
    clone_status=$?
    if [ $clone_status -eq 0 ]; then
        echo "Successfully cloned the repository."

        # add dir as safe in git
        git config --global --add safe.directory /cerebro-core

        # make all run files executable
        chmod +x /cerebro-core/init.sh
        chmod +x /cerebro-core/init.sh
        chmod +x /cerebro-core/cerebro/etl/run_etl.sh
        chmod +x /cerebro-core/cerebro/mop/run_mop.sh
        chmod +x /cerebro-core/server/run_server.sh
        chmod +x /cerebro-core/ui/run_ui.sh
    else
        echo "An error occurred while running git clone."
    fi
else
    echo "Unknown platform"
fi
