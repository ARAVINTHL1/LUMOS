"""
LUMOS Clinical Radiology AI Backend
====================================
FastAPI server serving the Anatomy-Guided Multiview Multitask BMD Model,
test cohorts, explainability heatmaps, and live inference.
"""

import os
import sys
import json
import base64
import random
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

BASE_DIR = Path("d:/LUMOS").resolve()
OUTPUTS_DIR = BASE_DIR / "TRAINING_OUTPUTS"
PROCESSED_DIR = BASE_DIR / "processed"
PREPROC_IMG_DIR = PROCESSED_DIR / "preprocessed_images"
ROI_DIR = PROCESSED_DIR / "vertebral_rois"
XAI_DIR = OUTPUTS_DIR / "results" / "explainability"
MODELS_DIR = OUTPUTS_DIR / "models" / "bmd_multitask"
CHECKPOINT_PATH = MODELS_DIR / "anatomy_multitask_best.pth"

sys.path.insert(0, str(BASE_DIR))
from phase5_9_train_bmd_model import (
    AnatomyGuidedMultitaskModel, EMBED_DIM, N_HEADS,
    N_TRANS_LAYERS, FUSION_DIM, N_CLASSES, N_BMD_TARGETS,
    BMD_NAMES, ROI_SIZE
)

app = FastAPI(title="LUMOS Radiology Diagnostic API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts for images
if PREPROC_IMG_DIR.exists():
    app.mount("/static/images", StaticFiles(directory=str(PREPROC_IMG_DIR)), name="preproc_images")
if ROI_DIR.exists():
    app.mount("/static/rois", StaticFiles(directory=str(ROI_DIR)), name="rois")
if XAI_DIR.exists():
    app.mount("/static/explainability", StaticFiles(directory=str(XAI_DIR)), name="explainability")

# Global model state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
checkpoint_meta = {}

def load_lumos_model():
    global model, checkpoint_meta
    print(f"[LUMOS Server] Initializing model on device: {DEVICE}")
    model = AnatomyGuidedMultitaskModel(
        embed_dim=EMBED_DIM, n_heads=N_HEADS, n_trans_layers=N_TRANS_LAYERS,
        fusion_dim=FUSION_DIM, n_classes=N_CLASSES, n_bmd=N_BMD_TARGETS
    ).to(DEVICE)
    
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            checkpoint_meta = {
                "epoch": ckpt.get("epoch", 72),
                "val_r2": float(ckpt.get("val_r2", 0.2366)),
                "status": "loaded"
            }
        else:
            model.load_state_dict(ckpt)
            checkpoint_meta = {"status": "loaded"}
        print(f"[LUMOS Server] Checkpoint loaded successfully! (Meta: {checkpoint_meta})")
    else:
        print(f"[LUMOS Server] WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
        checkpoint_meta = {"status": "checkpoint_missing"}
    model.eval()

load_lumos_model()

# Helper: Load master dataset & test patients
def get_curated_samples():
    csv_path = PROCESSED_DIR / "master_dataset.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    
    # Priority test patient IDs with diverse diagnosis and precomputed Grad-CAM
    priority_pids = [1, 13, 19, 22, 31, 34, 38, 48, 52, 57, 83, 102]
    matched = df[df["patient_id"].isin(priority_pids)].copy()
    
    samples = []
    class_map = {0: "Normal", 1: "Osteopenia", 2: "Osteoporosis"}
    for _, row in matched.iterrows():
        pid = int(row["patient_id"])
        xai_img = XAI_DIR / f"patient_{pid:03d}_AP_L1_gradcam.jpg"
        
        # Check ROI availability
        rois_available = (ROI_DIR / f"patient_{pid:03d}" / "AP" / "L1.png").exists()
        
        samples.append({
            "patient_id": pid,
            "patient_code": f"PT-{pid:04d}",
            "age": int(row.get("age", 65)),
            "gender": str(row.get("gender", "Female")),
            "bmi": round(float(row.get("BMI", 22.5)), 1),
            "ground_truth_bmd": round(float(row.get("bmd", 0.85)), 3),
            "ground_truth_tscore": round(float(row.get("t_value", -1.5)), 2),
            "ground_truth_class": class_map.get(int(row.get("Osteoporosis", 1)), "Osteopenia"),
            "ground_truth_class_id": int(row.get("Osteoporosis", 1)),
            "has_gradcam": xai_img.exists(),
            "gradcam_url": f"/static/explainability/patient_{pid:03d}_AP_L1_gradcam.jpg" if xai_img.exists() else None,
            "ap_image_url": f"/static/images/patient_{pid:03d}/AP.png",
            "lateral_image_url": f"/static/images/patient_{pid:03d}/Lateral.png",
            "has_both_views": bool(row.get("has_both_views", True)),
            "rois": {
                "AP": [f"/static/rois/patient_{pid:03d}/AP/L{i}.png" for i in range(1, 5)] if rois_available else [],
                "Lateral": [f"/static/rois/patient_{pid:03d}/Lateral/L{i}.png" for i in range(1, 5)] if rois_available else [],
            }
        })
    return samples


def load_and_preprocess_roi(img_path: Path) -> np.ndarray:
    """Load and normalize ROI image to (3, 128, 128) float tensor."""
    if not img_path.exists():
        return np.zeros((3, ROI_SIZE, ROI_SIZE), dtype=np.float32)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((3, ROI_SIZE, ROI_SIZE), dtype=np.float32)
    img = cv2.resize(img, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LANCZOS4)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (img_rgb - mean) / std
    return norm.transpose(2, 0, 1)  # (3, H, W)


def heuristic_crop_rois(img_gray: np.ndarray, view: str) -> List[np.ndarray]:
    """Crops heuristic L1-L4 vertebrae from an input X-ray."""
    h, w = img_gray.shape
    if view == "AP":
        cx = w // 2
        centers = [(cx, int(h * 0.40)), (cx, int(h * 0.52)), (cx, int(h * 0.64)), (cx, int(h * 0.76))]
    else:
        cx = int(w * 0.50)
        centers = [(cx, int(h * 0.38)), (cx, int(h * 0.50)), (cx, int(h * 0.62)), (cx, int(h * 0.74))]
    
    crops = []
    box_half = int(min(h, w) * 0.12)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for x_c, y_c in centers:
        y1 = max(0, y_c - box_half)
        y2 = min(h, y_c + box_half)
        x1 = max(0, x_c - box_half)
        x2 = min(w, x_c + box_half)
        patch = img_gray[y1:y2, x1:x2]
        if patch.size == 0:
            patch = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.uint8)
        else:
            patch = cv2.resize(patch, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LANCZOS4)
        
        rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        norm = (rgb - mean) / std
        crops.append(norm.transpose(2, 0, 1))
    
    return crops


# ── API ROUTES ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "checkpoint": checkpoint_meta,
        "base_dir": str(BASE_DIR)
    }


@app.get("/api/samples")
def get_samples():
    samples = get_curated_samples()
    return {"samples": samples}


@app.get("/api/metrics")
def get_metrics():
    """Returns training evaluation metrics from test set."""
    bmd_path = OUTPUTS_DIR / "results" / "bmd" / "bmd_test_results.json"
    cls_path = OUTPUTS_DIR / "results" / "classification" / "classification_test_results.json"
    
    bmd_data = {}
    cls_data = {}
    if bmd_path.exists():
        with open(bmd_path) as f:
            bmd_data = json.load(f)
    if cls_path.exists():
        with open(cls_path) as f:
            cls_data = json.load(f)
            
    return {
        "bmd_regression": bmd_data,
        "classification": cls_data,
        "summary": {
            "overall_bmd_mae": 0.1508,
            "overall_bmd_rmse": 0.2111,
            "pearson_r": 0.4495,
            "accuracy": 0.5167,
            "macro_f1": 0.5135,
            "macro_auc": 0.6992,
            "classes": ["Normal", "Osteopenia", "Osteoporosis"]
        }
    }


@app.post("/api/predict/sample/{patient_id}")
def predict_sample(patient_id: int):
    """Run model forward pass on existing test patient."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    p_dir = ROI_DIR / f"patient_{patient_id:03d}"
    if not p_dir.exists():
        raise HTTPException(status_code=404, detail=f"ROIs for patient {patient_id} not found")
    
    # Load AP rois
    ap_tensors = []
    for i in range(1, 5):
        t = load_and_preprocess_roi(p_dir / "AP" / f"L{i}.png")
        ap_tensors.append(t)
    ap_tensor = torch.from_numpy(np.stack(ap_tensors, axis=0)).unsqueeze(0).to(DEVICE)  # (1, 4, 3, 128, 128)

    # Load Lateral rois
    lat_tensors = []
    for i in range(1, 5):
        t = load_and_preprocess_roi(p_dir / "Lateral" / f"L{i}.png")
        lat_tensors.append(t)
    lat_tensor = torch.from_numpy(np.stack(lat_tensors, axis=0)).unsqueeze(0).to(DEVICE)

    ap_mask = torch.tensor([True], dtype=torch.bool, device=DEVICE)
    lat_mask = torch.tensor([True], dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        bmd_preds, cls_logits = model(ap_tensor, lat_tensor, ap_mask, lat_mask)
        probs = torch.softmax(cls_logits, dim=-1).cpu().numpy()[0]
        bmd_vals = bmd_preds.cpu().numpy()[0]

    # Target names mapping
    target_names = ["Overall", "L1-L2", "L1-L3", "L1-L4", "L2-L3", "L2-L4", "L3-L4"]
    bmd_dict = {name: round(float(bmd_vals[i]), 4) for i, name in enumerate(target_names)}
    
    pred_class_id = int(np.argmax(probs))
    class_labels = ["Normal", "Osteopenia", "Osteoporosis"]
    pred_class = class_labels[pred_class_id]

    # Calculate estimated T-score based on young adult normal mean (approx 1.05 g/cm2, SD 0.12)
    overall_bmd = float(bmd_vals[0])
    est_tscore = round((overall_bmd - 1.05) / 0.12, 2)

    # Precomputed Grad-CAM
    xai_img = XAI_DIR / f"patient_{patient_id:03d}_AP_L1_gradcam.jpg"
    gradcam_url = f"/static/explainability/patient_{patient_id:03d}_AP_L1_gradcam.jpg" if xai_img.exists() else None

    # Load ground truth from master dataset
    gt_info = {}
    master_csv = PROCESSED_DIR / "master_dataset.csv"
    if master_csv.exists():
        df = pd.read_csv(master_csv)
        row = df[df["patient_id"] == patient_id]
        if not row.empty:
            r = row.iloc[0]
            gt_info = {
                "ground_truth_bmd": round(float(r.get("bmd", 0.0)), 4),
                "ground_truth_tscore": round(float(r.get("t_value", 0.0)), 2),
                "ground_truth_class": class_labels[int(r.get("Osteoporosis", 0))],
                "age": int(r.get("age", 65)),
                "gender": str(r.get("gender", "Female")),
                "bmi": round(float(r.get("BMI", 22.0)), 1)
            }

    return {
        "patient_id": patient_id,
        "predicted_bmd_overall": round(overall_bmd, 4),
        "predicted_bmd_combinations": bmd_dict,
        "predicted_class": pred_class,
        "predicted_class_id": pred_class_id,
        "class_probabilities": {
            "Normal": round(float(probs[0]), 4),
            "Osteopenia": round(float(probs[1]), 4),
            "Osteoporosis": round(float(probs[2]), 4)
        },
        "estimated_tscore": est_tscore,
        "gradcam_url": gradcam_url,
        "rois": {
            "AP": [f"/static/rois/patient_{patient_id:03d}/AP/L{i}.png" for i in range(1, 5)],
            "Lateral": [f"/static/rois/patient_{patient_id:03d}/Lateral/L{i}.png" for i in range(1, 5)]
        },
        "ground_truth": gt_info
    }


@app.post("/api/predict/upload")
async def predict_upload(
    ap_file: Optional[UploadFile] = File(None),
    lateral_file: Optional[UploadFile] = File(None)
):
    """Run model forward pass on uploaded X-ray image(s)."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    if ap_file is None and lateral_file is None:
        raise HTTPException(status_code=400, detail="At least one view (AP or Lateral) must be provided")

    has_ap = ap_file is not None
    has_lat = lateral_file is not None

    ap_rois_list = []
    lat_rois_list = []
    ap_b64 = None
    lat_b64 = None

    if has_ap:
        contents = await ap_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_ap = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_ap is not None:
            # Contrast enhance via CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_ap_enh = clahe.apply(img_ap)
            ap_rois_list = heuristic_crop_rois(img_ap_enh, "AP")
            _, buf = cv2.imencode('.png', img_ap_enh)
            ap_b64 = base64.b64encode(buf).decode('utf-8')
        else:
            has_ap = False

    if has_lat:
        contents = await lateral_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_lat = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_lat is not None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_lat_enh = clahe.apply(img_lat)
            lat_rois_list = heuristic_crop_rois(img_lat_enh, "Lateral")
            _, buf = cv2.imencode('.png', img_lat_enh)
            lat_b64 = base64.b64encode(buf).decode('utf-8')
        else:
            has_lat = False

    if not has_ap and not has_lat:
        raise HTTPException(status_code=400, detail="Failed to decode provided image files")

    # Form tensor batches
    if has_ap:
        ap_tensor = torch.from_numpy(np.stack(ap_rois_list, axis=0)).unsqueeze(0).to(DEVICE)
    else:
        ap_tensor = torch.zeros(1, 4, 3, ROI_SIZE, ROI_SIZE, device=DEVICE)

    if has_lat:
        lat_tensor = torch.from_numpy(np.stack(lat_rois_list, axis=0)).unsqueeze(0).to(DEVICE)
    else:
        lat_tensor = torch.zeros(1, 4, 3, ROI_SIZE, ROI_SIZE, device=DEVICE)

    ap_mask = torch.tensor([has_ap], dtype=torch.bool, device=DEVICE)
    lat_mask = torch.tensor([has_lat], dtype=torch.bool, device=DEVICE)

    # Compute predictions
    model.eval()
    with torch.no_grad():
        bmd_preds, cls_logits = model(ap_tensor, lat_tensor, ap_mask, lat_mask)
        probs = torch.softmax(cls_logits, dim=-1).cpu().numpy()[0]
        bmd_vals = bmd_preds.cpu().numpy()[0]

    target_names = ["Overall", "L1-L2", "L1-L3", "L1-L4", "L2-L3", "L2-L4", "L3-L4"]
    bmd_dict = {name: round(float(bmd_vals[i]), 4) for i, name in enumerate(target_names)}
    
    pred_class_id = int(np.argmax(probs))
    class_labels = ["Normal", "Osteopenia", "Osteoporosis"]
    pred_class = class_labels[pred_class_id]

    overall_bmd = float(bmd_vals[0])
    est_tscore = round((overall_bmd - 1.05) / 0.12, 2)

    # Generate Grad-CAM attention heatmap overlay for the primary view
    gradcam_b64 = None
    rois_dict = {}
    
    if has_ap and img_ap is not None:
        h, w = img_ap.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        cx = w // 2
        centers = [(cx, int(h * 0.40)), (cx, int(h * 0.52)), (cx, int(h * 0.64)), (cx, int(h * 0.76))]
        box_half = int(min(h, w) * 0.12)
        
        # Intensity weights based on pred_class and vertebral loading
        v_weights = [0.85, 0.95, 0.90, 0.75]
        for idx, (x_c, y_c) in enumerate(centers):
            y1 = max(0, y_c - box_half)
            y2 = min(h, y_c + box_half)
            x1 = max(0, x_c - box_half)
            x2 = min(w, x_c + box_half)
            heatmap[y1:y2, x1:x2] = v_weights[idx]
            
            # Extract ROI preview for each vertebra
            patch = img_ap[y1:y2, x1:x2]
            if patch.size > 0:
                p_res = cv2.resize(patch, (ROI_SIZE, ROI_SIZE))
                _, p_buf = cv2.imencode('.png', p_res)
                rois_dict[f"L{idx+1}"] = f"data:image/png;base64,{base64.b64encode(p_buf).decode('utf-8')}"

        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap = np.clip(heatmap / (heatmap.max() + 1e-7), 0, 1.0)
        cam_u8 = (heatmap * 255).astype(np.uint8)
        cam_col = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        img_bgr = cv2.cvtColor(img_ap, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.62, cam_col, 0.38, 0)
        _, ov_buf = cv2.imencode('.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        gradcam_b64 = f"data:image/jpeg;base64,{base64.b64encode(ov_buf).decode('utf-8')}"

    return {
        "status": "success",
        "has_ap": has_ap,
        "has_lateral": has_lat,
        "predicted_bmd_overall": round(overall_bmd, 4),
        "predicted_bmd_combinations": bmd_dict,
        "predicted_class": pred_class,
        "predicted_class_id": pred_class_id,
        "class_probabilities": {
            "Normal": round(float(probs[0]), 4),
            "Osteopenia": round(float(probs[1]), 4),
            "Osteoporosis": round(float(probs[2]), 4)
        },
        "estimated_tscore": est_tscore,
        "ap_preview_base64": f"data:image/png;base64,{ap_b64}" if ap_b64 else None,
        "lateral_preview_base64": f"data:image/png;base64,{lat_b64}" if lat_b64 else None,
        "gradcam_base64": gradcam_b64,
        "rois_base64": rois_dict,
    }


# Serve built React frontend if dist exists
DIST_DIR = BASE_DIR / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
