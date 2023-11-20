# FROM prsridha/cerebro-base-aws:latest
FROM prsridha/cerebro-base-voyager:latest

# update packages
RUN apt-get update

# install AngularJS
ENV NODE_VERSION=18.14.2
RUN apt install -y curl
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"
RUN node --version
RUN npm --version
RUN npm install -g @angular/cli

# create required directories
RUN mkdir -p /home/cerebro-core

# copy necessary paths
COPY ./ui /cerebro-core/ui
COPY ./cerebro /cerebro-core/cerebro
COPY ./server /cerebro-core/server
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

# dummy task (Kubernetes will overrride)
ENTRYPOINT ["tail", "-f", "/dev/null"]

# run this from cerebro-core dir
# sudo docker build -t prsridha/cerebro-core -f build/Dockerfile .
