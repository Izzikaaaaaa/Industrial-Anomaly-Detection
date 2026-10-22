# Industrial AI Inspector

A full-stack academic prototype for visual anomaly detection on the **MVTec AD `metal_nut`** category.

It includes:
- Streamlit dashboard with single-image and batch inspection
- CNN Autoencoder trained mainly on normal images
- PatchCore-style detector using a pretrained ResNet18 feature extractor and a normal-feature memory bank
- Anomaly heatmaps and threshold-based `NORMAL` / `POTENTIAL ANOMALY` results
- Image-level and pixel-level evaluation utilities
- CSV exports and model-comparison reporting

> This is a research/educational prototype, not a certified industrial safety system. A flagged image means the model found an unusual visual pattern; it does not identify the exact defect type or prove mechanical failure.

## 1. Requirements

Python 3.10 or 3.11 is recommended. A Mac with Apple Silicon can run PyTorch using MPS when supported; CPU is also supported.

### macOS setup

Open Terminal and run:

```bash
cd path/to/industrial-ai-inspector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3` is unavailable, install Python 3.10/3.11 first. Keep the virtual environment activated for all commands.

## 2. Add the dataset

Download MVTec AD from the official dataset page:
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Extract the dataset so this folder exists:

```text
data/metal_nut/
├── train/
│   └── good/
├── test/
│   ├── good/
│   ├── bent/
│   ├── color/
│   ├── flip/
│   └── scratch/
└── ground_truth/
    ├── bent/
    ├── color/
    ├── flip/
    └── scratch/
```

The exact defect folders may depend on the dataset package. The scripts discover image files recursively. **Do not put the dataset in GitHub**; follow the dataset licence and attribution requirements.

## 3. Train the models

From the project root, with the virtual environment active:

```bash
python -m src.train_autoencoder --data data/metal_nut --epochs 10
python -m src.train_patchcore --data data/metal_nut
```

The first PatchCore run attempts to download pretrained ResNet18 weights if they are not cached. An internet connection may be required. The checkpoint stores the feature extractor weights so inference uses the same backbone. If you want a quicker smoke test, use `--max-images 30 --max-bank 1000`; this is only for testing the pipeline, not for final evaluation.

## 4. Launch the app

```bash
streamlit run app.py
```

The app opens in your browser. Upload an image of a metal nut, select a trained model, and inspect the score and heatmap. If you have not trained a model yet, the app still launches and explains what is missing.

## 5. Evaluate on the official test set

Keep the official test set separate from training and threshold calibration:

```bash
python -m src.evaluate --data data/metal_nut
```

This writes metrics to `reports/metrics.json` and per-image results to `reports/test_predictions.csv`. It calculates image ROC-AUC when both classes are present, and pixel ROC-AUC / IoU / Dice when ground-truth masks are available. Results depend on the actual run; do not claim numbers before running the evaluation.

## 6. Project architecture

```text
MVTec AD metal_nut
       |
       +--> CNN Autoencoder --> reconstruction-error heatmap
       |
       +--> PatchCore-style --> pretrained CNN features
                                  -> normal feature memory
                                  -> nearest-neighbour distances
       |
       v
Anomaly score + calibrated threshold + heatmap
       |
       v
Streamlit UI -> single inspection / batch CSV / model comparison
```

## 7. Important methodological notes

- Models learn normal visual appearance; they are not trained here to classify `bent`, `color`, `flip`, or `scratch` as separate labels.
- The normal training images are split into a fit subset and a held-out normal calibration subset. The threshold is estimated from the calibration scores, not by tuning against the official test labels.
- PatchCore here is a **PatchCore-style educational implementation**: pretrained ResNet18 intermediate features, a capped normal patch memory bank, and nearest-neighbour distances. It is not a wrapper around Anomalib and does not claim bit-for-bit reproduction of the original paper.
- Autoencoder and PatchCore scores are on different scales. Compare ranking metrics (e.g., ROC-AUC), thresholded metrics, localization quality, and latency—not raw scores directly.
- Lighting, camera position, rotation, product alignment, and background changes can cause false alarms.
- Do not use the demo for real production decisions without validation on representative factory data.

## 8. Common fixes

**`No images found`:** Check that the path is `data/metal_nut/train/good` and contains image files.

**`torchvision` install error:** Use a clean Python 3.10/3.11 virtual environment and reinstall from `requirements.txt`. Avoid mixing unrelated Conda and pip environments.

**PatchCore weight download fails:** Check internet access and rerun the training command. The script stops with a clear error rather than silently claiming pretrained performance.

**App says model not found:** Train that model first. Model files are saved in `models/`.
