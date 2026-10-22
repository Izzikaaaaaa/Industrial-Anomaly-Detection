from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from .common import list_images, load_rgb, pil_to_tensor, choose_device, split_normal, image_score_from_map, safe_torch_load

class ResNetPatchFeatures(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f2 = F.normalize(f2, p=2, dim=1)
        f3 = F.normalize(f3, p=2, dim=1)
        return torch.cat([f2, f3], dim=1)

def _features_for_paths(model, paths, device, size=224, batch_size=8):
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([load_rgb(p, size) for p in paths[start:start+batch_size]]).to(device)
            # ImageNet normalization for pretrained ResNet
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            feat = model((batch - mean) / std)
            feat = F.normalize(feat, p=2, dim=1)
            chunks.append(feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1]).cpu())
    return torch.cat(chunks, dim=0).numpy()

def _image_feature(model, image, device, size=224):
    x = pil_to_tensor(image, size).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        feat = model((x - mean) / std)
        feat = F.normalize(feat, p=2, dim=1)
    return feat

def train(data_dir="data/metal_nut", max_bank=5000, max_images=None):
    train_dir = Path(data_dir) / "train" / "good"
    paths = list_images(train_dir)
    if max_images:
        paths = paths[:max_images]
    fit_paths, calibration_paths = split_normal(paths)
    device = choose_device()
    try:
        model = ResNetPatchFeatures(pretrained=True).to(device)
    except Exception as exc:
        raise RuntimeError("Could not load pretrained ResNet18 weights. Check internet access and rerun. Original error: " + str(exc))
    feats = _features_for_paths(model, fit_paths, device)
    rng = np.random.default_rng(42)
    if len(feats) > max_bank:
        feats = feats[rng.choice(len(feats), size=max_bank, replace=False)]
    bank = torch.tensor(feats, dtype=torch.float32)
    # Build a neighbour index to calibrate normal scores on held-out normal images.
    from sklearn.neighbors import NearestNeighbors
    nn_index = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(bank.numpy())
    val_scores = []
    for path in calibration_paths:
        fmap = _image_feature(model, path, device)
        vecs = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]).cpu().numpy()
        distances, _ = nn_index.kneighbors(vecs)
        amap = distances.reshape(fmap.shape[-2], fmap.shape[-1])
        val_scores.append(image_score_from_map(amap))
    threshold = float(np.percentile(val_scores, 99)) if val_scores else 0.0
    out = Path("models/patchcore.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone_state": model.cpu().state_dict(), "memory_bank": bank,
                "threshold": threshold, "image_size": 224, "model_name": "PatchCore-style",
                "feature_dim": int(bank.shape[1])}, out)
    print(f"Saved {out} | bank patches={len(bank)} | normal-calibration threshold={threshold:.6f}")

def predict(image, checkpoint="models/patchcore.pt"):
    ckpt = safe_torch_load(checkpoint, map_location="cpu")
    device = choose_device()
    # Build architecture without downloading weights; load saved feature extractor weights.
    model = ResNetPatchFeatures(pretrained=False)
    model.load_state_dict(ckpt["backbone_state"])
    model.to(device).eval()
    bank = ckpt["memory_bank"].cpu().numpy()
    from sklearn.neighbors import NearestNeighbors
    nn_index = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(bank)
    fmap = _image_feature(model, image, device, int(ckpt.get("image_size", 224)))
    vecs = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]).cpu().numpy()
    distances, _ = nn_index.kneighbors(vecs)
    amap = distances.reshape(fmap.shape[-2], fmap.shape[-1]).astype(np.float32)
    # Upsample to a more useful heatmap resolution.
    amap_t = torch.from_numpy(amap).view(1, 1, *amap.shape)
    amap = F.interpolate(amap_t, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    score = image_score_from_map(amap)
    threshold = float(ckpt["threshold"])
    return {"score": score, "threshold": threshold, "heatmap": amap,
            "prediction": "POTENTIAL ANOMALY" if score > threshold else "NORMAL"}
