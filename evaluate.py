from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from .common import list_images
from .autoencoder import predict as predict_ae
from .patchcore_model import predict as predict_pc

def mask_path_for(image_path, data_root):
    p = Path(image_path)
    # MVTec mask: ground_truth/<defect_folder>/<stem>_mask.png
    defect = p.parent.name
    if defect == "good":
        return None
    candidate = Path(data_root) / "ground_truth" / defect / f"{p.stem}_mask.png"
    return candidate if candidate.exists() else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/metal_nut")
    parser.add_argument("--model", choices=["both", "autoencoder", "patchcore"], default="both")
    args = parser.parse_args()
    root = Path(args.data)
    test_root = root / "test"
    image_paths = list_images(test_root)
    if not image_paths:
        raise SystemExit(f"No test images found in {test_root}")
    models = []
    if args.model in ("both", "autoencoder") and Path("models/autoencoder.pt").exists():
        models.append(("autoencoder", predict_ae))
    if args.model in ("both", "patchcore") and Path("models/patchcore.pt").exists():
        models.append(("patchcore", predict_pc))
    if not models:
        raise SystemExit("No requested trained model found. Train the model(s) first.")
    rows, metrics = [], {}
    for model_name, predictor in models:
        scores, labels, preds = [], [], []
        pixel_y, pixel_s = [], []
        latencies = []
        for path in image_paths:
            label = 0 if path.parent.name == "good" else 1
            with Image.open(path) as im:
                im = im.convert("RGB")
                t0 = time.perf_counter()
                result = predictor(im)
                latencies.append(time.perf_counter() - t0)
                scores.append(result["score"])
                labels.append(label)
                pred = int(result["score"] > result["threshold"])
                preds.append(pred)
                rows.append({"model": model_name, "image": str(path), "label_anomaly": label,
                             "score": result["score"], "threshold": result["threshold"],
                             "prediction": result["prediction"], "latency_seconds": latencies[-1]})
                mp = mask_path_for(path, root)
                if mp:
                    try:
                        mask = np.asarray(Image.open(mp).convert("L").resize((result["heatmap"].shape[1], result["heatmap"].shape[0]), Image.Resampling.NEAREST)) > 0
                        amap = np.asarray(result["heatmap"])
                        pixel_y.extend(mask.ravel().astype(int).tolist())
                        pixel_s.extend(amap.ravel().astype(float).tolist())
                    except Exception:
                        pass
        item = {"images": len(labels), "mean_latency_seconds": float(np.mean(latencies))}
        if len(set(labels)) > 1:
            item["image_roc_auc"] = float(roc_auc_score(labels, scores))
        item["precision"] = float(precision_score(labels, preds, zero_division=0))
        item["recall"] = float(recall_score(labels, preds, zero_division=0))
        item["f1"] = float(f1_score(labels, preds, zero_division=0))
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        item["confusion_matrix"] = {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}
        if len(set(pixel_y)) > 1:
            item["pixel_roc_auc"] = float(roc_auc_score(pixel_y, pixel_s))
            binary_map = np.asarray(pixel_s) >= float(np.percentile(pixel_s, 95))
            y = np.asarray(pixel_y, dtype=bool)
            inter = int(np.logical_and(binary_map, y).sum())
            union = int(np.logical_or(binary_map, y).sum())
            item["pixel_iou_at_global_p95"] = float(inter / union) if union else 1.0
            denom = int(binary_map.sum() + y.sum())
            item["pixel_dice_at_global_p95"] = float(2 * inter / denom) if denom else 1.0
        metrics[model_name] = item
        print(model_name, json.dumps(item, indent=2))
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("reports/test_predictions.csv", index=False)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved reports/metrics.json and reports/test_predictions.csv")

if __name__ == "__main__":
    main()
