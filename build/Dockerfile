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
RUN mkdir -p /project/cerebro-core

# copy necessary paths
COPY ./ui /project/cerebro-core/ui
COPY ./cerebro /project/cerebro-core/cerebro
COPY ./server /project/cerebro-core/server
COPY init.sh /project/cerebro-core/init.sh
COPY setup.py /project/cerebro-core/setup.py
COPY requirements.txt /project/cerebro-core/requirements.txt

# set working directory
WORKDIR /project/cerebro-core

# install dependencies and create cerebro package
RUN pip install -r requirements.txt
RUN pip install -v -e /project/cerebro-core

# set python path env variables
ENV PYTHONPATH="${PYTHONPATH}:/project/cerebro-core"
ENV PYTHONPATH="${PYTHONPATH}:/project/cerebro-core/cerebro"
ENV PYTHONPATH="${PYTHONPATH}:/project/user"

# outsource task to Kubernetes config
ENTRYPOINT ["tail", "-f", "/dev/null"]

# run this from cerebro-core dir
# sudo docker build -t prsridha/project/cerebro-core -f build/Dockerfile .
