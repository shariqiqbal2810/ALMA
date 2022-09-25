FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
# FROM ubuntu:16.04

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && \
    apt-get install -y sudo curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev \
    protobuf-compiler apt-utils nano vim man build-essential wget && \
    curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y python3.7

RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py" && python3.7 get-pip.py && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy scipy pyyaml==5.4.1 matplotlib imageio pygame imageio-ffmpeg \
    tensorboard-logger ruamel.base ryd jsonpickle==0.9.6 wandb torch==1.1.0 \
    ipdb ipython \
    git+https://github.com/oxwhirl/smac.git

RUN pip3 install ortools

ENV SC2PATH /task-allocation/3rdparty/StarCraftII

# Setup weights and biases
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    WANDB_CONFIG_DIR=/task-allocation/.config/wandb
RUN wandb login $WANDB_API_KEY

WORKDIR /task-allocation
