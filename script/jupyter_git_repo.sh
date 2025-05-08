#!/bin/bash
# this script is use to bring up a jupyer notebook
# with git repo {ML-Sys-DevOps} in it ß


# Define paths
REPO_DIR=~/ML-Sys-DevOps
CONTAINER_DIR=/home/jovyan/ML-Sys-DevOps

# Run Jupyter Docker container with volume mount
docker run -it --rm \ß
  -p 8888:8888 \
  -v ${REPO_DIR}:${CONTAINER_DIR} \
  jupyter/pytorch-notebook