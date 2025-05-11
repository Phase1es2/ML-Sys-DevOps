### ✅ dataloader.py（替换 kagglehub）
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DepthProImageProcessorFast

LR_SIZE = 256
SCALE_FACTOR = 4
HR_SIZE = LR_SIZE * SCALE_FACTOR
BATCH_SIZE = 1

processor = DepthProImageProcessorFast(do_resize=False, do_rescale=True, do_normalize=True)

def collate_fn(samples):
    lrs = [i[0] for i in samples]
    hrs = [i[1] for i in samples]
    lrs = processor(lrs, return_tensors="pt")['pixel_values']
    hrs = processor(hrs, return_tensors="pt")['pixel_values']
    return lrs, hrs

class Div2kDataset(Dataset):
    def __init__(self, split, base_path="/mnt/object", lr_size=256, hr_size=1024, keep_aspect_ratio=False):
        self.split = split
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.keep_aspect_ratio = keep_aspect_ratio
        if split == "train":
            self.images_paths = [os.path.join(base_path, "div2k/train", i) for i in os.listdir(os.path.join(base_path, "div2k/train"))]
        elif split == "validation":
            self.images_paths = [os.path.join(base_path, "div2k/validation", i) for i in os.listdir(os.path.join(base_path, "div2k/validation"))]
        else:
            raise ValueError(f"Invalid split={split}")

    def __len__(self):
        return len(self.images_paths)

    def _crop_image(self, image):
        w, h = image.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        return image.crop((left, top, left + min_dim, top + min_dim))

    def __getitem__(self, idx):
        image = Image.open(self.images_paths[idx]).convert("RGB")
        if not self.keep_aspect_ratio:
            image = self._crop_image(image)
            lr = image.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)
            hr = image.resize((self.hr_size, self.hr_size), Image.Resampling.BICUBIC)
        else:
            lr = image.copy()
            hr = image.copy()
            lr.thumbnail((self.lr_size, self.lr_size))
            hr.thumbnail((self.hr_size, self.hr_size))
        return lr, hr


class Urban100Dataset(Dataset):
    """
    Pytorch Dataset for Urban100 dataset.
    """
    base_path = "/mnt/object/urban100"
    lr_path = 'Urban 100/X4 Urban100/X4/LOW x4 URban100'
    hr_path = 'Urban 100/X4 Urban100/X4/HIGH x4 URban100'

    def __init__(self):
        lr_images_path, hr_images_path = self.configure()
        self.lr_images_path = lr_images_path
        self.hr_images_path = hr_images_path

    def configure(self):
        lr_images_path = os.path.join(self.base_path, self.lr_path)
        hr_images_path = os.path.join(self.base_path, self.hr_path)
        lr_images_path = [os.path.join(lr_images_path, i) for i in os.listdir(lr_images_path)
        hr_images_path = [os.path.join(hr_images_path, i) for i in os.listdir(hr_images_path)
        print(f"lr_images_path: {lr_images_path}")
        print(f"hr_images_path: {hr_images_path}")
        return lr_images_path, hr_images_path

    def __len__(self):
        """
        Number of Images in the dataset.
        """
        return len(self.lr_images_path)

    def __getitem__(self, idx):
        """
        Load and return the Low Resolution (lr) and High Resolution (hr) image.
        """
        lr = Image.open(self.lr_images_path[idx]).convert("RGB")
        hr = Image.open(self.hr_images_path[idx]).convert("RGB")
        return lr, hr
        
def collate_fn(samples):
    lrs = [i[0] for i in samples]
    hrs = [i[1] for i in samples]
    lrs = processor(lrs, return_tensors="pt")['pixel_values']
    hrs = processor(hrs, return_tensors="pt")['pixel_values']
    return lrs, hrs

def get_dataloaders(batch_size=BATCH_SIZE):
    train_dataset = Div2kDataset("train", lr_size=LR_SIZE, hr_size=HR_SIZE, keep_aspect_ratio=True)
    val_dataset = Div2kDataset("validation", lr_size=LR_SIZE, hr_size=HR_SIZE, keep_aspect_ratio=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    return train_loader, val_loader