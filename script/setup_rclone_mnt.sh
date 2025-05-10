#!/bin/bash
# using this code to set up rclone, if first time launch instance.
set -e

echo "Installing rclone..."
curl https://rclone.org/install.sh | sudo bash

echo "Enabling FUSE allow_other support..."
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

echo "Creating rclone config directory..."
mkdir -p ~/.config/rclone

# copy the cc@node1-cloud-project45:~/ML-Sys-DevOps/docker$ ls ~/.config/rclone/
# rclone.conf
echo "Opening rclone.conf — paste your remote config (press :wq to save and exit)..."
vim ~/.config/rclone/rclone.conf

echo "Creating /mnt/object with correct ownership..."
sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object

echo "Now mounting chi_tacc:object-persist-project45..."

rclone mount chi_tacc:object-persist-project45 /mnt/object --read-only --allow-other --daemon

echo "Mount complete. Verifying contents:"
ls -lh /mnt/object