from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from src.autoencoder import predict as predict_ae
from src.patchcore_model import predict as predict_pc

st.set_page_config(page_title="Industrial AI Inspector", page_icon="🔍", layout="wide")

ROOT = Path(__file__).parent
MODEL_FILES = {"CNN Autoencoder": ROOT / "models" / "autoencoder.pt",
               "PatchCore-style": ROOT / "models" / "patchcore.pt"}
PREDICTORS = {"CNN Autoencoder": predict_ae, "PatchCore-style": predict_pc}
HISTORY = ROOT / "reports" / "inspection_history.csv"

def history_append(row):
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(HISTORY, mode="a", header=not HISTORY.exists(), index=False)

def overlay_figure(image, heatmap, title="Anomaly heatmap"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image); axes[0].set_title("Original image"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="inferno"); axes[1].set_title(title); axes[1].axis("off")
    axes[2].imshow(image)
    axes[2].imshow(heatmap, cmap="inferno", alpha=0.45)
    axes[2].set_title("Overlay"); axes[2].axis("off")
    fig.tight_layout()
    return fig

def run_prediction(model_name, image, source_name):
    predictor = PREDICTORS[model_name]
    start = time.perf_counter()
    result = predictor(image)
    latency = time.perf_counter() - start
    history_append({"timestamp": pd.Timestamp.now().isoformat(), "filename": source_name,
                    "model": model_name, "prediction": result["prediction"],
                    "anomaly_score": result["score"], "threshold": result["threshold"],
                    "latency_seconds": latency})
    return result, latency

st.title("🔍 Industrial AI Inspector")
st.caption("Visual anomaly detection prototype • MVTec AD metal_nut • CNN Autoencoder + PatchCore-style")
st.info("This research prototype flags unusual visual appearance. It does not certify a part as safe, identify the exact defect, or predict machine failure.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose page", ["Overview", "Single Inspection", "Batch Inspection", "Model Analytics", "Inspection History"])
    st.divider()
    st.subheader("Model status")
    for name, path in MODEL_FILES.items():
        st.write(("🟢 " if path.exists() else "⚪ ") + name)
        if not path.exists():
            st.caption("Not trained yet")

if page == "Overview":
    st.subheader("Project overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detection approaches", "2")
    c2.metric("Input", "Metal-nut image")
    c3.metric("Output", "Score + heatmap")
    st.markdown("""
    ### How it works
    1. **CNN Autoencoder:** learns normal appearances and highlights reconstruction differences.
    2. **PatchCore-style:** compares pretrained CNN patch features with a memory bank of normal patches.
    3. **Decision:** compares the anomaly score with a threshold calibrated using held-out normal training images.
    4. **Evaluation:** evaluates image-level detection and, where masks are available, pixel-level localization.
    """)
    st.markdown("### Get started")
    st.code("python -m src.train_autoencoder --data data/metal_nut --epochs 10\npython -m src.train_patchcore --data data/metal_nut\nstreamlit run app.py", language="bash")
    st.warning("Models must be trained before real model predictions can be made. See README.md for dataset setup.")

elif page == "Single Inspection":
    st.subheader("Inspect one metal-nut image")
    model_name = st.selectbox("Model", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=uploaded.name, width=380)
        if not MODEL_FILES[model_name].exists():
            st.error(f"{model_name} is not trained yet. Run its training command from README.md.")
        elif st.button("Run inspection", type="primary"):
            try:
                with st.spinner("Analyzing image..."):
                    result, latency = run_prediction(model_name, image, uploaded.name)
                if result["prediction"] == "NORMAL":
                    st.success("Prediction: NORMAL")
                else:
                    st.warning("Prediction: POTENTIAL ANOMALY")
                a, b, c = st.columns(3)
                a.metric("Anomaly score", f"{result['score']:.5f}")
                b.metric("Decision threshold", f"{result['threshold']:.5f}")
                c.metric("Inference time", f"{latency:.3f} s")
                st.pyplot(overlay_figure(image, result["heatmap"]), clear_figure=True)
                st.caption("Heatmap colours show relative model anomaly response; they are not guaranteed defect boundaries.")
            except Exception as e:
                st.exception(e)

elif page == "Batch Inspection":
    st.subheader("Batch inspection")
    model_name = st.selectbox("Select model for batch", list(MODEL_FILES.keys()))
    uploads = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], accept_multiple_files=True)
    if uploads:
        st.write(f"{len(uploads)} image(s) selected.")
        if not MODEL_FILES[model_name].exists():
            st.error(f"{model_name} is not trained yet. Train it first.")
        elif st.button("Inspect batch", type="primary"):
            rows = []
            bar = st.progress(0)
            for i, f in enumerate(uploads):
                try:
                    image = Image.open(f).convert("RGB")
                    result, latency = run_prediction(model_name, image, f.name)
                    rows.append({"filename": f.name, "model": model_name, "prediction": result["prediction"],
                                 "anomaly_score": result["score"], "threshold": result["threshold"],
                                 "latency_seconds": latency})
                except Exception as e:
                    rows.append({"filename": f.name, "model": model_name, "error": str(e)})
                bar.progress((i + 1) / len(uploads))
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV report", df.to_csv(index=False).encode("utf-8"),
                               file_name="industrial_inspection_results.csv", mime="text/csv")

elif page == "Model Analytics":
    st.subheader("Model analytics")
    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        rows = []
        for name, m in metrics.items():
            row = {"Model": name}
            for key in ["images", "image_roc_auc", "pixel_roc_auc", "precision", "recall", "f1", "mean_latency_seconds"]:
                if key in m: row[key.replace("_", " ").title()] = m[key]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("These values come from the most recent evaluation run. Check reports/metrics.json for full details.")
    else:
        st.info("No evaluation results yet. Train one or both models, then run:")
        st.code("python -m src.evaluate --data data/metal_nut", language="bash")
    st.markdown("### What to compare")
    st.markdown("- **Image ROC-AUC:** ranking quality across normal and anomalous test images.\n- **Precision / Recall / F1:** performance at each model's calibrated threshold.\n- **Pixel ROC-AUC / IoU / Dice:** heatmap localization against ground-truth masks.\n- **Latency:** measured time for inference; hardware and warm-up affect it.")
    st.warning("The two models use different score scales. Do not compare raw anomaly scores directly.")

elif page == "Inspection History":
    st.subheader("Inspection history")
    if HISTORY.exists():
        df = pd.read_csv(HISTORY)
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("Download history CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="inspection_history.csv", mime="text/csv")
        if st.button("Clear history file"):
            HISTORY.unlink(missing_ok=True)
            st.rerun()
    else:
        st.info("No inspections recorded yet. Run a single or batch inspection first.")
