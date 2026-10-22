# Industrial AI Inspector — original frontend + live Python models

This package preserves the supplied HTML/CSS dashboard layout and connects single and batch image uploads to your trained Python models. It does not include the large model checkpoint files or MVTec dataset.

## 1. Keep your existing model files
Copy your trained files into this folder: `models/autoencoder.pt` and `models/patchcore.pt`. Keep the existing `src/` folder from your trained project and its model files. Keep `data/metal_nut/` if you need to train again.

## 2. Install dependencies (inside your existing `.venv`)
```bash
python -m pip install -r requirements.txt
```

## 3. Start the app
```bash
uvicorn server:app --reload
```
Open http://127.0.0.1:8000. Do not open `index.html` directly in Finder; the browser must use the local API server.

## Included live features
- Original frontend layout and styling
- Single image inference with either trained model
- Model-generated heatmap and overlay
- Batch inference (up to 30 images)
- CSV inspection history written to `reports/inspection_history.csv`
- API status at `/api/status` and history at `/api/history`

## Important limitations
- `score_percent` is a display scaling of the raw anomaly score against the calibrated threshold; it is not a probability or model confidence. The dashboard's confidence field is left as `—` for that reason.
- The original Overview and Model Analytics chart illustrations in the provided HTML are not converted into validated ROC/PR metrics by this integration. Do not present those illustrative chart values as measured model performance.
- Anomaly maps are model response visualizations, not verified defect segmentation or mechanical-damage diagnosis.
- The `src/` code and checkpoints must match the model versions you trained.
