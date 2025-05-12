#!/bin/bash

sudo mkdir -p /mnt/block
sudo mount /dev/vdc1 /mnt/block

echo "list all under /mnt/block"
ls /mnt/block