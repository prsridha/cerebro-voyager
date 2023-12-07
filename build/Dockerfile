# FROM prsridha/cerebro-base-aws:latest
FROM prsridha/cerebro-base-voyager:latest

# CHANGE THIS BASED ON YOUR ENVIRONMENT
ARG ENVIRONMENT=dev

# update packages
RUN apt-get update

# create required directories
RUN mkdir -p /home/cerebro-core

# copy necessary paths
COPY ./cerebro /cerebro-core/cerebro
COPY init.sh /cerebro-core/init.sh
COPY setup.py /cerebro-core/setup.py
COPY requirements.txt /cerebro-core/requirements.txt

# install dependencies
RUN pip install -r /cerebro-core/requirements.txt

# set python path env variables
ENV PYTHONPATH="${PYTHONPATH}:/cerebro-core"
ENV PYTHONPATH="${PYTHONPATH}:/cerebro-core/cerebro"
ENV PYTHONPATH="${PYTHONPATH}:/user"

# install cerebro package
RUN pip install -v -e /cerebro-core

# Copy sync.sh script
COPY build/sync.sh /sync.sh

# Set executable permissions
RUN chmod +x /cerebro-core/init.sh
RUN chmod +x /sync.sh
RUN chmod +x /cerebro-core/cerebro/etl/run_etl.sh
RUN chmod +x /cerebro-core/cerebro/mop/run_mop.sh

# make it permissive if in dev enviroment
RUN if [ "$ENVIRONMENT" = "dev" ]; then \
        chmod -R 777 /cerebro-core \
    fi


# dummy task (Kubernetes will overrride)
ENTRYPOINT ["tail", "-f", "/dev/null"]

# run this from cerebro-core dir
# FOR PROD: sudo docker build --build-arg ENVIRONMENT=prod -t prsridha/cerebro-core -f build/Dockerfile .
# FOR DEV: sudo docker build --build-arg ENVIRONMENT=dev -t prsridha/cerebro-core -f build/Dockerfile .
