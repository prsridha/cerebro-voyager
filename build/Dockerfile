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
RUN mkdir -p /cerebro-repo/cerebro-core

# copy necessary paths
COPY ./ui /cerebro-repo/cerebro-core/ui
COPY ./core /cerebro-repo/cerebro-core/core
COPY ./server /cerebro-repo/cerebro-core/server
COPY init.sh /cerebro-repo/cerebro-core/init.sh
COPY setup.py /cerebro-repo/cerebro-core/setup.py
COPY requirements.txt /cerebro-repo/cerebro-kube/requirements.txt

# set working directory
WORKDIR /cerebro-repo/cerebro-core

# install dependencies and create cerebro package
RUN pip install -r requirements.txt
RUN pip install -v -e /cerebro-repo/cerebro-core

# set python path env variables
ENV PYTHONPATH="${PYTHONPATH}:/cerebro-repo/cerebro-core"
ENV PYTHONPATH="${PYTHONPATH}:/cerebro-repo/cerebro-core/core"
ENV PYTHONPATH="${PYTHONPATH}:/user-repo"

# outsource task to Kubernetes config
ENTRYPOINT ["tail", "-f", "/dev/null"]

# run this from cerebro-core dir
# sudo docker build -t prsridha/cerebro-core -f build/Dockerfile .
