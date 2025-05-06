from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import base64
import io
from PIL import Image
import numpy as np

from transformers import DepthProConfig
from depthpro_sr import DepthProForSuperResolution  # 你自己定义的模型类（建议放到 depthpro_sr.py 中）

app = FastAPI(
    title="Image Super-Resolution API",
    description="Upscales low-res images using DepthPro-based SR model",
    version="1.0.0"
)

# Request/response schema
class ImageRequest(BaseModel):
    image: str  # base64-encoded image

class ImageResponse(BaseModel):
    sr_image: str  # base64-encoded super-resolved image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
config = DepthProConfig.from_pretrained("geetu040/DepthPro", revision="project")
model = DepthProForSuperResolution(config)
model.load_state_dict(torch.load("depthpro_sr_best.pth", map_location=device))
model.to(device)
model.eval()

# Preprocess input image
def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = image.resize((256, 256), Image.BICUBIC)
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    return image

# Postprocess output tensor to image
def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    tensor = (tensor * 0.5 + 0.5).clip(0, 1)  # Unnormalize
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)

@app.post("/predict", response_model=ImageResponse)
def super_resolve(request: ImageRequest):
    try:
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        input_tensor = preprocess_image(image).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            output = F.interpolate(output, scale_factor=4, mode="bilinear", align_corners=False)

        sr_image = tensor_to_image(output)
        buffered = io.BytesIO()
        sr_image.save(buffered, format="PNG")
        sr_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return ImageResponse(sr_image=sr_base64)
    except Exception as e:
        return {"error": str(e)}

