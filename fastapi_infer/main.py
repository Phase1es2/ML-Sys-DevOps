# fastapi_infer/main.py

from fastapi import FastAPI
from pydantic import BaseModel, Field
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import base64
import io
import numpy as np

app = FastAPI(
    title="Food Classification API",
    description="API for classifying food items from images",
    version="1.0.0"
)

# Define request and response formats
class ImageRequest(BaseModel):
    image: str  # base64-encoded image

class PredictionResponse(BaseModel):
    prediction: str
    probability: float = Field(..., ge=0, le=1)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_PATH = "food11.pth"
model = torch.load(MODEL_PATH, map_location=device)
model.to(device)
model.eval()

# Class labels
classes = np.array([
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"
])

# Image preprocessing
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

@app.post("/predict", response_model=PredictionResponse)
def predict_image(request: ImageRequest):
    try:
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = preprocess_image(image).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, 1).item()
            conf = probs[0, pred_idx].item()

        return PredictionResponse(prediction=classes[pred_idx], probability=conf)
    except Exception as e:
        return {"error": str(e)}
