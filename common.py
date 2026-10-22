from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def load_rgb(path, size):
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def pil_to_tensor(image, size):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def split_normal(paths, val_fraction=0.15, seed=42):
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    if len(paths) < 3:
        raise ValueError(f"Need at least 3 normal training images; found {len(paths)}.")
    n_val = max(1, int(round(len(paths) * val_fraction)))
    n_val = min(n_val, len(paths) - 2)
    return paths[n_val:], paths[:n_val]

def image_score_from_map(anomaly_map):
    arr = np.asarray(anomaly_map, dtype=np.float32)
    return float(np.percentile(arr, 99))

def safe_torch_load(path, map_location="cpu"):
    # Explicit weights_only=False is needed because checkpoints contain metadata arrays.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
