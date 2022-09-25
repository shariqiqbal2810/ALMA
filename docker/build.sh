#!/bin/bash

echo 'Building Dockerfile with image name task-allocation'
docker build --build-arg WANDB_API_KEY=${WANDB_API_KEY} -t task-allocation .
