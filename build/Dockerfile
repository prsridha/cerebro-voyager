# FROM prsridha/cerebro-base-aws:latest
FROM prsridha/cerebro-base-voyager:latest

# update packages
RUN apt-get update

# copy necessary paths
COPY ./cerebro /cerebro-voyager/cerebro
COPY init.sh /cerebro-voyager/init.sh
COPY setup.py /cerebro-voyager/setup.py
COPY requirements.txt /cerebro-voyager/requirements.txt

# install dependencies
RUN pip install -r /cerebro-voyager/requirements.txt

# set python path env variables
ENV PYTHONPATH="${PYTHONPATH}:/cerebro-voyager"
ENV PYTHONPATH="${PYTHONPATH}:/cerebro-voyager/cerebro"
ENV PYTHONPATH="${PYTHONPATH}:/user"

# install cerebro package
RUN pip install -v -e /cerebro-voyager

# Copy sync.sh script
COPY build/sync.sh /sync.sh

# Set permissions to user
RUN chmod -R +rx /cerebro-voyager
RUN chmod +x /sync.sh

# dummy task (Kubernetes will overrride)
ENTRYPOINT ["tail", "-f", "/dev/null"]

# run this from cerebro-voyager dir
# sudo docker build -t prsridha/cerebro-voyager -f build/Dockerfile .
