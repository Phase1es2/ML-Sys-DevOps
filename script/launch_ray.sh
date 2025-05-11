#!/bin/bash

set -e

echo "🚀 准备启动 Ray CUDA 集群..."

# ✅ 设置浮动 IP 给 compose 中用到的变量
export RAY_HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
echo "📡 当前浮动 IP: $RAY_HOST_IP"

# ✅ 创建 external volume：训练数据路径（只需执行一次）
docker volume create --name depthpro_data \
  --opt type=none \
  --opt device=/mnt/object \
  --opt o=bind

# ✅ 创建 external volume：训练代码路径（只需执行一次）
docker volume create --name depthpro_code \
  --opt type=none \
  --opt device=/home/jovyan/workspace/ML-Sys-DevOps/mlflow-scripts \
  --opt o=bind

# ✅ 启动 Ray 集群
docker compose -f docker/docker-compose-ray-cuda.yaml up -d

echo "✅ Ray 集群已启动！你可以访问以下地址："
echo "  📊 Ray Dashboard:    http://${RAY_HOST_IP}:8265"
echo "  📈 Grafana Dashboard: http://${RAY_HOST_IP}:3000"
