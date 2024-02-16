# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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