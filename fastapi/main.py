from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
from transformers import DepthProConfig, DepthProImageProcessorFast, DepthProForDepthEstimation
import io

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Super-Resolution API", version="1.0.0")

# Model setup
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

# Load model from Lightning .ckpt
MODEL_PATH = "/mnt/minio_data/temp_model/model-epoch=07-val_psnr=24.88.ckpt"
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

state_dict = checkpoint["state_dict"]

# Remove 'model.' prefix
clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

model = DepthProForSuperResolution(depthpro_for_depth_estimation)
model.load_state_dict(clean_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Processor
image_processor = DepthProImageProcessorFast(do_resize=False, do_rescale=True, do_normalize=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")  # Ensure 3 channels
    image.thumbnail((256, 256))

    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    output = outputs.squeeze(0).cpu()  # safer than outputs[0] if output is [1, C, H, W]
    output = torch.permute(output, (1, 2, 0))  # CHW -> HWC
    output = output * 0.5 + 0.5
    output = output * 255.0
    output = output.clip(0, 255).numpy().astype("uint8")
    output_image = Image.fromarray(output)

    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/", response_class=HTMLResponse)
async def frontend():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Super Resolution Upload</title>
    </head>
    <body>
        <h2>Upload an Image for Super-Resolution</h2>
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required />
            <button type="submit">Upload</button>
        </form>
        <h3>Output Image:</h3>
        <img id="output-img" style="max-width: 512px; border: 1px solid #ccc;" />
        <script>
            const form = document.getElementById("upload-form");
            const outputImg = document.getElementById("output-img");

            form.onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData(form);
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData,
                });

                if (!response.ok) {
                    alert("Upload failed.");
                    return;
                }

                const blob = await response.blob();
                outputImg.src = URL.createObjectURL(blob);
            };
        </script>
    </body>
    </html>
    """