#!/bin/sh
#!/usr/bin/env bash
set -euo pipefail

# —— Configuration (override these in your ~/.bashrc) —— #
: "${RCLONE_REMOTE:=chi_tacc:}"                     # your rclone remote
: "${RCLONE_CONTAINER:=object-persist-project45}"  # remote “bucket”/container name
: "${COMPOSE_FILE:=$HOME/ML-Sys-DevOps/docker/docker-compose-etl.yaml}"
: "${MOUNT_POINT:=/mnt/object}"

# —— Preflight checks —— #
for cmd in curl rclone docker fusermount; do
  command -v "$cmd" >/dev/null || {
    echo "ERROR: '$cmd' not found in PATH." >&2
    exit 1
  }
done

# —— Install rclone if missing —— #
if ! command -v rclone >/dev/null; then
  echo " Installing rclone…"
  curl https://rclone.org/install.sh | bash
fi

# —— Enable user_allow_other in fuse.conf —— #
if ! grep -q '^user_allow_other' /etc/fuse.conf; then
  echo " Enabling user_allow_other in /etc/fuse.conf…"
  sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
fi

# —— Ensure config dir exists —— #
mkdir -p "$HOME/.config/rclone"

# —— Smoke‑test remote —— #
echo "Listing top‑level of $RCLONE_REMOTE …"
rclone lsd "$RCLONE_REMOTE"

# —— Run your ETL steps —— #
echo "Extracting data…"
docker compose -f "$COMPOSE_FILE" run extract-data

echo "Transforming data…"
docker compose -f "$COMPOSE_FILE" run transform-data

echo "Loading data…"
docker compose -f "$COMPOSE_FILE" run load-data

# —— Prepare mount point —— #
echo "Cleaning up any stale mount at $MOUNT_POINT…"
fusermount -u "$MOUNT_POINT" 2>/dev/null || sudo umount "$MOUNT_POINT" 2>/dev/null || true

echo "Creating and chown’ing $MOUNT_POINT…"
sudo mkdir -p "$MOUNT_POINT"
sudo chown "${SUDO_USER:-$USER}":"${SUDO_USER:-$USER}" "$MOUNT_POINT"

# —— Mount the DIV2K tree —— #
echo "Mounting $RCLONE_REMOTE/$RCLONE_CONTAINER/div2k → $MOUNT_POINT …"
rclone mount \
  "$RCLONE_REMOTE/$RCLONE_CONTAINER/div2k" \
  "$MOUNT_POINT" \
  --read-only --allow-other --daemon

echo "Done. You should now see:"
ls "$MOUNT_POINT"