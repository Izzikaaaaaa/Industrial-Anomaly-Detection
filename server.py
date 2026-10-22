from __future__ import annotations
import base64, io, time, uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.autoencoder import predict as predict_ae
from src.patchcore_model import predict as predict_pc

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {"CNN Autoencoder": ROOT / "models" / "autoencoder.pt", "PatchCore-style": ROOT / "models" / "patchcore.pt"}
PREDICTORS = {"CNN Autoencoder": predict_ae, "PatchCore-style": predict_pc}
HISTORY = ROOT / "reports" / "inspection_history.csv"
LOCK = Lock()
app = FastAPI(title="Industrial AI Inspector API", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")

def png_data(arr):
    bio=io.BytesIO(); Image.fromarray(arr.astype(np.uint8)).save(bio, format="PNG")
    return "data:image/png;base64,"+base64.b64encode(bio.getvalue()).decode("ascii")

def render_maps(image, amap):
    image=image.convert("RGB")
    amap=np.asarray(amap,dtype=np.float32)
    lo=float(np.min(amap)); hi=float(np.max(amap))
    norm=(amap-lo)/(hi-lo+1e-8)
    heat=(plt.get_cmap("inferno")(norm)[...,:3]*255).astype(np.uint8)
    heat_img=Image.fromarray(heat).resize(image.size,Image.Resampling.BILINEAR)
    overlay=Image.blend(image,heat_img,0.48)
    return png_data(np.asarray(heat_img)), png_data(np.asarray(overlay))

def run(image, filename, model_name):
    if model_name not in PREDICTORS: raise HTTPException(400, "Unknown model selection")
    checkpoint=MODEL_FILES[model_name]
    if not checkpoint.exists(): raise HTTPException(503, f"Model checkpoint missing: {checkpoint.name}. Copy your trained .pt file into models/.")
    start=time.perf_counter()
    try: result=PREDICTORS[model_name](image)
    except Exception as exc: raise HTTPException(500, f"Model inference failed: {type(exc).__name__}: {exc}")
    latency=(time.perf_counter()-start)*1000
    threshold=float(result["threshold"]); score=float(result["score"])
    score_percent=float(np.clip(100*score/(threshold*1.5+1e-8),0,100))
    heat,overlay=render_maps(image,result["heatmap"])
    record={"id":"INSP-"+uuid.uuid4().hex[:8].upper(),"timestamp":datetime.now().isoformat(timespec="seconds"),"filename":filename,"model":model_name,"prediction":result["prediction"],"score":score,"score_percent":round(score_percent,2),"threshold":threshold,"latency_ms":round(latency,2),"time":datetime.now().strftime("%d %b %Y · %H:%M:%S"),"reason":"Model response above threshold" if result["prediction"]!="NORMAL" else "Model response within normal calibration range"}
    with LOCK:
        HISTORY.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame([record]).to_csv(HISTORY,mode="a",header=not HISTORY.exists(),index=False)
    return {**record,"heatmap_png":heat,"overlay_png":overlay}

async def read_image(file: UploadFile):
    if file.content_type not in {"image/jpeg","image/png","image/webp","image/bmp","image/tiff"}:
        raise HTTPException(400,f"Unsupported image type: {file.filename}")
    raw=await file.read()
    if len(raw)>10*1024*1024: raise HTTPException(413,f"Image too large: {file.filename}")
    try: return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception: raise HTTPException(400,f"Cannot decode image: {file.filename}")

@app.post("/api/predict")
async def predict(file: UploadFile=File(...), model: str=Form("PatchCore-style")):
    image=await read_image(file)
    return run(image,file.filename or "uploaded-image",model)

@app.post("/api/batch")
async def batch(files: list[UploadFile]=File(...), model: str=Form("PatchCore-style")):
    if not files: raise HTTPException(400,"Choose at least one image")
    if len(files)>30: raise HTTPException(400,"Maximum 30 images per batch")
    results=[]
    for f in files:
        image=await read_image(f); r=run(image,f.filename or "uploaded-image",model)
        results.append({k:r[k] for k in ("id","filename","prediction","score","score_percent","threshold","latency_ms","time","reason")})
    return {"results":results}

@app.get("/api/history")
def history():
    if not HISTORY.exists(): return {"records":[]}
    try: return {"records":pd.read_csv(HISTORY).replace({np.nan:None}).to_dict(orient="records")}
    except Exception as exc: raise HTTPException(500,str(exc))

@app.get("/api/status")
def status():
    return {"models":{name:{"ready":path.exists(),"file":path.name} for name,path in MODEL_FILES.items()},"history_exists":HISTORY.exists()}
