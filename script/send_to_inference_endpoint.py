import os
import requests

# Server URL
URL = "http://localhost:5000/predict"  # Change if server is remote

# New image directory on mounted path
IMAGE_DIR = "/mnt/object/bsd100/bsd100/bicubic_2x/train/HR"

# Output directory
os.makedirs("inference_outputs", exist_ok=True)

# Loop through images
for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith(".png"):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    print(f"Sending {filename}...")

    with open(image_path, "rb") as f:
        files = {"file": (filename, f, "image/png")}
        response = requests.post(URL, files=files)

    if response.status_code == 200:
        out_path = os.path.join("inference_outputs", filename)
        with open(out_path, "wb") as out_file:
            out_file.write(response.content)
        print(f"Saved output to {out_path}")
    else:
        print(f"Failed to process {filename}: {response.status_code}")