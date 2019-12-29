# Github private repo pull issue
## Refer to https://vsupalov.com/build-docker-image-clone-private-repo-ssh-key/
#FROM ubuntu as intermediate
#
## install git
#RUN apt-get update
#RUN apt-get install -y git
#
## add credentials on build
#ARG SSH_PRIVATE_KEY_PATH
#RUN mkdir /root/.ssh/
#ADD $SSH_PRIVATE_KEY_PATH /root/.ssh/
#RUN chmod 400 /root/.ssh/id_rsa
#
## make sure your domain is accepted
## RUN touch /root/.ssh/known_hosts
## RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
#
## https://medium.com/paperchain/fetching-private-github-repos-from-a-docker-container-273f25ec5a74
## Use git with SSH instead of https
#RUN echo "[url \"git@github.com:\"]\n\tinsteadOf = https://github.com/" >> /root/.gitconfig
#
## Skip Host verification for git
#RUN echo "StrictHostKeyChecking no" > /root/.ssh/config
#
## Install repo
#RUN cd / \
# && git clone git@github.com:qiminchen/CoralNet.git


ARG CUDA="10.0"
ARG CUDNN="7"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04
# copy the repository form the previous image
# COPY --from=intermediate /CoralNet /CoralNet

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git vim curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py37 python=3.7.3 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install vscode debugger (ptvsd), aws and aws-plugin
RUN pip install ptvsd

# Set up the timezone
RUN apt install -y tzdata
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install wget, nano and tmux
RUN apt install -y wget nano tmux unzip

SHELL ["/bin/bash", "-c"]

# Clone git repository
# https://stackoverflow.com/questions/19643946/dockerfile-for-cloning-private-git-repo
ARG git_personal_token
RUN git clone https://${git_personal_token}@github.com/qiminchen/CoralNet.git \
 && mkdir ~/.aws \
 && cd CoralNet \
 && cp credentials ~/.aws/credentials \
 && conda env create -f environment.yml \
 && source ../miniconda/etc/profile.d/conda.sh \
 && conda activate coralnet \
 && aws configure set plugins.endpoint awscli_plugin_endpoint \
 && aws configure set s3api.endpoint_url https://s3.nautilus.optiputer.net \
 && aws configure set s3.endpoint_url https://s3.nautilus.optiputer.net

WORKDIR /CoralNet
