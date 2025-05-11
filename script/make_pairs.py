# scripts/make_pairs.py
from PIL import Image
import os

hr_dir = "/mnt/block/MyRetrainSet/train/hr"
lr_dir = "/mnt/block/MyRetrainSet/train/lr"
os.makedirs(lr_dir, exist_ok=True)

for fname in os.listdir(hr_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        hr_path = os.path.join(hr_dir, fname)
        lr_path = os.path.join(lr_dir, fname)
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = hr_img.resize((hr_img.width // 4, hr_img.height // 4), Image.BICUBIC)
        lr_img.save(lr_path)

print("✅ Low-resolution training images have been successfully generated.")
