import torch
import time
import os
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.audio import SignalNoiseRatio
import torch.nn.functional as F
from model_def import DepthProForSuperResolution, get_depthpro_model

# ==== 配置模型路径与测试图像（使用模拟数据） ====
model_path = "depthpro_sr_quantized.pt"  # 替换成你的模型
is_torchscript = model_path.endswith(".pt")

# ==== 构建模型结构 & 加载参数 ====
if is_torchscript:
    model = torch.jit.load(model_path, map_location="cpu")
else:
    depth_model = get_depthpro_model(compressed=True)
    model = DepthProForSuperResolution(depth_model)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

model.eval()

# ==== 模型大小 ====
model_size = os.path.getsize(model_path) / 1e6
print(f"📦 Model Size on Disk: {model_size:.2f} MB")

# ==== 模拟输入数据（如有实际数据可替换） ====
B, C, H, W = 4, 3, 64, 64
input_batch = torch.randn(B, C, H, W)
target_batch = torch.randn(B, C, H * 4, W * 4)  # 假设超分 x4，按实际设置改

# ==== 定义指标 ====
mse_fn = torch.nn.MSELoss()
psnr = PeakSignalNoiseRatio().cpu()
ssim = StructuralSimilarityIndexMeasure().cpu()
snr = SignalNoiseRatio().cpu()

# ==== 推理并计算图像指标 ====
with torch.no_grad():
    preds = model(input_batch)
    preds = F.interpolate(preds, size=target_batch.shape[2:])  # resize

    test_results = {
        "test_mse_loss": mse_fn(preds, target_batch).item(),
        "test_psnr": psnr(preds, target_batch).item(),
        "test_ssim": ssim(preds, target_batch).item(),
        "test_snr": snr(preds, target_batch).item()
    }

# ==== 推理延迟评估（单样本） ====
single_input = input_batch[0:1]
num_trials = 100

# 预热
with torch.no_grad():
    model(single_input)

latencies = []
with torch.no_grad():
    for _ in range(num_trials):
        start = time.time()
        _ = model(single_input)
        latencies.append(time.time() - start)

latencies = np.array(latencies)
print(f"Inference Latency (median): {np.percentile(latencies, 50)*1000:.2f} ms")
print(f"Inference Latency (95th): {np.percentile(latencies, 95)*1000:.2f} ms")
print(f"Inference Latency (99th): {np.percentile(latencies, 99)*1000:.2f} ms")
print(f"Inference Throughput (1 sample): {num_trials/latencies.sum():.2f} FPS")

# ==== 批量推理吞吐评估 ====
num_batches = 50
with torch.no_grad():
    model(input_batch)  # 预热

batch_times = []
with torch.no_grad():
    for _ in range(num_batches):
        start = time.time()
        _ = model(input_batch)
        batch_times.append(time.time() - start)

batch_fps = (input_batch.shape[0] * num_batches) / np.sum(batch_times)

# ==== 输出总结 ====
print("\n===== Summary =====")
print(f"📦 Model Size: {model_size:.2f} MB")
for k, v in test_results.items():
    print(f"📊 {k}: {v:.4f}")
print(f"⚡ Latency (median): {np.percentile(latencies, 50)*1000:.2f} ms")
print(f"⚡ Latency (95th): {np.percentile(latencies, 95)*1000:.2f} ms")
print(f"⚡ Latency (99th): {np.percentile(latencies, 99)*1000:.2f} ms")
print(f"🚀 Single-sample FPS: {num_trials/latencies.sum():.2f}")
print(f"🚀 Batch Throughput: {batch_fps:.2f} FPS")