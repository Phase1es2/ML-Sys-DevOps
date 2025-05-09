#!/bin/bash
# this script is used to bring up a jupyer notebook
# with git repo {ML-Sys-DevOps} in it ß

sudo groupadd -f docker; sudo usermod -aG docker $USER

# Define paths
REPO_DIR=~/ML-Sys-DevOps
CONTAINER_DIR=/home/jovyan/ML-Sys-DevOps
curl -sSL https://get.docker.com/ | sudo sh

# Run Jupyter Docker container with volume mount
docker run -it --rm \ß
  -p 8888:8888 \
  -v ${REPO_DIR}:${CONTAINER_DIR} \
  jupyter/pytorch-notebook

echo Using FIP:8888/token
