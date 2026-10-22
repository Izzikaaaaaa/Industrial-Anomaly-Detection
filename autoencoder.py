from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from .common import load_rgb, pil_to_tensor, choose_device, list_images, split_normal, image_score_from_map

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def _tensor_batch(paths, size=128):
    return torch.stack([load_rgb(p, size) for p in paths])

def train(data_dir="data/metal_nut", epochs=10, batch_size=16, lr=1e-3, max_images=None):
    train_dir = Path(data_dir) / "train" / "good"
    paths = list_images(train_dir)
    if max_images:
        paths = paths[:max_images]
    fit_paths, calibration_paths = split_normal(paths)
    device = choose_device()
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(fit_paths))
        losses = []
        for start in range(0, len(order), batch_size):
            batch_paths = [fit_paths[i] for i in order[start:start + batch_size]]
            x = _tensor_batch(batch_paths).to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch + 1, "train_mse": mean_loss})
        print(f"Epoch {epoch+1}/{epochs} - train MSE: {mean_loss:.6f}")

    model.eval()
    val_scores = []
    with torch.no_grad():
        for path in calibration_paths:
            x = load_rgb(path, 128).unsqueeze(0).to(device)
            recon = model(x)
            amap = (x - recon).abs().mean(dim=1, keepdim=True)
            amap = F.interpolate(amap, size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            val_scores.append(image_score_from_map(amap))
    threshold = float(np.percentile(val_scores, 99)) if val_scores else 0.0
    out = Path("models/autoencoder.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "threshold": threshold,
                "image_size": 128, "history": history, "model_name": "CNN Autoencoder"}, out)
    print(f"Saved {out} | normal-calibration threshold={threshold:.6f}")

def predict(image, checkpoint="models/autoencoder.pt"):
    from .common import safe_torch_load
    device = choose_device()
    ckpt = safe_torch_load(checkpoint, map_location="cpu")
    model = ConvAutoencoder()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    x = pil_to_tensor(image, int(ckpt.get("image_size", 128))).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(x)
        amap = (x - recon).abs().mean(dim=1, keepdim=True)
        amap = F.interpolate(amap, size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    score = image_score_from_map(amap)
    return {"score": score, "threshold": float(ckpt["threshold"]), "heatmap": amap,
            "prediction": "POTENTIAL ANOMALY" if score > float(ckpt["threshold"]) else "NORMAL"}
