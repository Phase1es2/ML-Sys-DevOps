#!/bin/sh
curl -sSL https://get.docker.com/ | sudo sh
sudo groupadd -f docker; sudo usermod -aG docker $USER
docker run hello-world
docker run -it --rm -p 8888:8888 -v "$PWD":/home/jovyan/work jupyter/base-notebook
echo Using FIP:8888/token