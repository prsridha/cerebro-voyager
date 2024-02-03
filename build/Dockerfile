# FROM prsridha/cerebro-base-aws:latest
FROM prsridha/cerebro-base-voyager:latest

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

# Set permissions to user
RUN chmod -R +rx /cerebro-core
RUN chmod +x /sync.sh

# dummy task (Kubernetes will overrride)
ENTRYPOINT ["tail", "-f", "/dev/null"]

# run this from cerebro-core dir
# sudo docker build --build-arg -t prsridha/cerebro-core -f build/Dockerfile .
