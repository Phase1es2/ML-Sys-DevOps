# compress_export.py

import os
import torch
import torch.nn as nn
from model_def import DepthProForSuperResolution, get_depthpro_model

# ==== 1. 加载压缩版 DepthPro 模型 ====
print("📦 Loading compressed DepthPro model...")
depth_model = get_depthpro_model(compressed=True)

# ==== 2. 构建超分模型 ====
model = DepthProForSuperResolution(depth_model)
model.eval()

# 上传原始的
# # 构建结构（保持压缩版配置一致）
# depth_model = get_depthpro_model(compressed=True)
# model = DepthProForSuperResolution(depth_model)

# # ✅ 加载你训练好的参数
# model.load_state_dict(torch.load("depthpro_sr_trained.pth", map_location="cpu"))

# model.eval()


# ==== 3. 保存原始（未量化）模型 ====
torch.save(model.state_dict(), "depthpro_sr_compressed.pth")
print("✅ Saved: depthpro_sr_compressed.pth")

# ==== 4. 动态量化 ====
print("🔧 Applying dynamic quantization...")
# 只量化 image_head 和 head，而不是整个 DepthPro 模型
model.image_head = torch.quantization.quantize_dynamic(
    model.image_head, {nn.Linear}, dtype=torch.qint8
)
model.head = torch.quantization.quantize_dynamic(
    model.head, {nn.Linear}, dtype=torch.qint8
)

# 注意：不触碰 model.depthpro_for_depth_estimation，保持原样
quantized_model = model  # 安全地继续用它
torch.save(quantized_model.state_dict(), "depthpro_sr_quantized.pth")
print("✅ Saved: depthpro_sr_quantized.pth")

# ==== 5. 导出为 ONNX ====
print("📤 Exporting to ONNX...")
dummy_input = torch.randn(1, 3, 64, 64)
onnx_path = "depthpro_sr_compressed.onnx"
torch.onnx.export(
    quantized_model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    export_params=True,
    opset_version=17,
    do_constant_folding=True
)
print(f"✅ Saved: {onnx_path}")

# ==== 6. TorchScript 导出 ====
print("📤 Exporting to TorchScript...")
traced = torch.jit.trace(quantized_model, dummy_input)
traced.save("depthpro_sr_quantized.pt")
print("✅ Saved: depthpro_sr_quantized.pt")

# ==== 7. 显示模型大小 ====
def show_size(path):
    mb = os.path.getsize(path) / (1024 * 1024)
    print(f"📦 {path}: {mb:.2f} MB")

for path in [
    "depthpro_sr_compressed.pth",
    "depthpro_sr_quantized.pth",
    "depthpro_sr_quantized.pt",
    "depthpro_sr_compressed.onnx"
]:
    show_size(path)