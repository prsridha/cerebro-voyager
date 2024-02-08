FROM vault.habana.ai/gaudi-docker/1.11.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:latest

# install basic packages
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y nano
RUN apt-get install -y git
RUN apt-get install -y lshw
RUN apt-get install -y zip
RUN apt-get install -y python3-pip
RUN python3 -m pip install --upgrade pip

COPY requirements.txt .

# install base requirements
RUN pip install -r requirements.txt

# install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# set environment variables
ENV PATH="/root/.local/bin:${PATH}"