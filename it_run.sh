#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
GPU=$1
name=${USER}_pymarl_GPU_${GPU}_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}'"
# Launches a docker container using our image, and runs the provided command

# runs container interactively to enable debugging
docker run \
    --gpus '"device='$GPU'"' \
    --rm \
    --name $name \
    --cap-add=SYS_PTRACE \
    --net host \
    --user $(id -u) \
    -v `pwd`:/task-allocation \
    -v ${SC2PATH}:/task-allocation/3rdparty/StarCraftII \
    -it task-allocation \
    ${@:2}
