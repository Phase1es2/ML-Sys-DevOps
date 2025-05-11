# fastapi/main_with_upload.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import io
import os
from transformers import DepthProConfig, DepthProImageProcessorFast, DepthProForDepthEstimation

app = FastAPI(title="Super-Resolution API with Save", version="1.0.0")

# 模型加载
config = DepthProConfig(
    patch_size=32,
    patch_embeddings_size=4,
    num_hidden_layers=12,
    intermediate_hook_ids=[11, 8, 7, 5],
    intermediate_feature_dims=[256, 256, 256, 256],
    scaled_images_ratios=[0.5, 1.0],
    scaled_images_overlap_ratios=[0.5, 0.25],
    scaled_images_feature_dims=[1024, 512],
    use_fov_model=False,
)

depthpro_for_depth_estimation = DepthProForDepthEstimation(config)

class DepthProForSuperResolution(torch.nn.Module):
    def __init__(self, depthpro_for_depth_estimation):
        super().__init__()
        self.depthpro_for_depth_estimation = depthpro_for_depth_estimation
        hidden_size = self.depthpro_for_depth_estimation.config.fusion_hidden_size

        self.image_head = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(config.num_channels, hidden_size, 4, 2, 1),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_size, config.num_channels, 3, 1, 1),
        )

    def forward(self, pixel_values):
        x = pixel_values
        encoder_features = self.depthpro_for_depth_estimation.depth_pro(x).features
        fused_hidden_state = self.depthpro_for_depth_estimation.fusion_stage(encoder_features)[-1]
        x = self.image_head(x)
        x = torch.nn.functional.interpolate(x, size=fused_hidden_state.shape[2:])
        x = x + fused_hidden_state
        x = self.head(x)
        return x

# 加载模型
MODEL_PATH = "/mnt/block/minio_data/temp_model/model-epoch=07-val_psnr=24.88.ckpt"
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}

model = DepthProForSuperResolution(depthpro_for_depth_estimation)
model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# 预处理器
image_processor = DepthProImageProcessorFast(do_resize=False, do_rescale=True, do_normalize=True)

@app.post("/predict_with_save")
async def predict_with_save(file: UploadFile = File(...)):
    raw_bytes = await file.read()

    # 保存上传图像
    UPLOAD_DIR = "/mnt/block/MyRetrainSet/train/hr"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(raw_bytes)

    # 推理部分
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image.thumbnail((256, 256))
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    output = outputs.squeeze(0).cpu()
    output = torch.permute(output, (1, 2, 0))
    output = output * 0.5 + 0.5
    output = output * 255.0
    output = output.clip(0, 255).numpy().astype("uint8")
    output_image = Image.fromarray(output)

    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
