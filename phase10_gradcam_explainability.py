"""
PHASE 10 — Grad-CAM++ Explainability
======================================
Project: Anatomy-Guided Multiview Multitask Deep Learning for Lumbar BMD
         Estimation and Osteoporosis Severity Assessment from X-ray Images

After the BMD model is trained, this script:
  1. Generates Grad-CAM++ heatmaps for the CNN encoder (shared backbone)
  2. Overlays heatmaps on original X-rays
  3. Computes quantitative anatomical consistency:
       overlap(Grad-CAM region, vertebral mask) → overlap score per vertebra
  4. Saves visualizations and metrics

REQUIREMENTS:
  pip install grad-cam
  Trained BMD model checkpoint (phase5_9_train_bmd_model.py)
  Segmentation masks (phase3_train_segmentation_model.py)
"""

import os
import json
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings('ignore')

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not installed.")
    print("Install with: pip install grad-cam")

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(os.environ.get("LUMOS_DIR", Path(__file__).resolve().parent))
PROCESSED_DIR = BASE_DIR / "processed"
PREDICTED_DIR = PROCESSED_DIR / "segmentation" / "predicted"
VERIFIED_DIR  = PROCESSED_DIR / "segmentation" / "verified"
PSEUDO_DIR    = PROCESSED_DIR / "segmentation" / "pseudo_labels"
MODELS_DIR    = BASE_DIR / "models" / "bmd_multitask"
XAI_DIR       = BASE_DIR / "results" / "explainability"
XAI_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT    = MODELS_DIR / "anatomy_multitask_best.pth"
ROI_MANIFEST  = PROCESSED_DIR / "roi_manifest.csv"

N_SAMPLES     = 20   # number of patients to visualize
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_mask(pid: int, view: str) -> np.ndarray | None:
    """Load segmentation mask: prefer verified > predicted > pseudo_labels."""
    fname = f"patient_{pid:03d}_{view}_mask.png"
    for d in [VERIFIED_DIR, PREDICTED_DIR, PSEUDO_DIR]:
        p = d / fname
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None


def compute_cam_mask_overlap(cam: np.ndarray, mask: np.ndarray,
                              threshold: float = 0.5) -> dict:
    """
    Computes overlap between high-activation Grad-CAM region
    and vertebral segmentation masks.

    cam:  (H, W) float in [0, 1]
    mask: (H, W) uint8 with labels 0-4

    Returns overlap scores per vertebra.
    """
    cam_binary = (cam >= threshold).astype(np.uint8)
    results = {}
    for label_id, label_name in {1:'L1', 2:'L2', 3:'L3', 4:'L4'}.items():
        seg_binary = (mask == label_id).astype(np.uint8)
        intersection = (cam_binary & seg_binary).sum()
        union        = (cam_binary | seg_binary).sum()
        cam_total    = cam_binary.sum()
        seg_total    = seg_binary.sum()
        iou          = float(intersection) / (float(union) + 1e-7)
        precision    = float(intersection) / (float(cam_total) + 1e-7)
        recall       = float(intersection) / (float(seg_total) + 1e-7)
        results[label_name] = {
            'iou': iou, 'precision': precision, 'recall': recall
        }
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 10: Grad-CAM++ Explainability")
    print("=" * 60)

    if not GRADCAM_AVAILABLE:
        print("  ERROR: pytorch-grad-cam not installed.")
        print("  pip install grad-cam")
        exit(1)

    if not CHECKPOINT.exists():
        print(f"  ERROR: Model checkpoint not found: {CHECKPOINT}")
        print("  Train the model first (phase5_9_train_bmd_model.py)")
        exit(1)

    # Load model
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from phase5_9_train_bmd_model import (
        AnatomyGuidedMultitaskModel, EMBED_DIM, N_HEADS,
        N_TRANS_LAYERS, FUSION_DIM, N_CLASSES, N_BMD_TARGETS
    )

    model = AnatomyGuidedMultitaskModel(
        embed_dim=EMBED_DIM, n_heads=N_HEADS, n_trans_layers=N_TRANS_LAYERS,
        fusion_dim=FUSION_DIM, n_classes=N_CLASSES, n_bmd=N_BMD_TARGETS
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"  Model loaded from: {CHECKPOINT}")

    # Target layer for Grad-CAM++ — last conv layer of MobileNetV2 features
    target_layer = [model.shared_cnn.features[-1]]

    # Load manifest and sample patients
    roi_df = pd.read_csv(ROI_MANIFEST)
    test_patients = roi_df[roi_df['split'] == 'test']['patient_id'].unique()
    sample_pids = list(test_patients[:N_SAMPLES])
    print(f"  Generating Grad-CAM++ for {len(sample_pids)} test patients")
    print()

    all_overlap_results = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    cam_engine = GradCAMPlusPlus(model=model.shared_cnn, target_layers=target_layer)

    for pid in sample_pids:
        print(f"  Patient {pid:03d} ...", end=' ', flush=True)

        # Get AP L1 ROI (representative)
        rec = roi_df[(roi_df['patient_id'] == pid) & (roi_df['view'] == 'AP')]
        if len(rec) == 0:
            print("SKIPPED (no AP record)")
            continue

        rec = rec.iloc[0]
        roi_path = rec.get('L1_roi_path', '')
        if not roi_path or not os.path.exists(str(roi_path)):
            print("SKIPPED (no L1 ROI)")
            continue

        img_gray = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print("SKIPPED (could not read image)")
            continue

        img_gray = cv2.resize(img_gray, (128, 128))
        img_3ch  = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32) / 255.0
        img_norm = (img_3ch - mean) / std
        input_t  = torch.from_numpy(img_norm.transpose(2,0,1)).unsqueeze(0).to(DEVICE)

        # Grad-CAM++ (target: Osteoporosis class = 2 for clinical relevance)
        targets  = [ClassifierOutputTarget(2)]
        try:
            grayscale_cam = cam_engine(input_tensor=input_t, targets=targets)
            cam = grayscale_cam[0]  # (H, W) in [0, 1]
        except Exception as e:
            print(f"ERROR ({str(e)[:40]})")
            continue

        # Overlay on image
        img_rgb_vis = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32) / 255.0
        cam_overlay = show_cam_on_image(img_rgb_vis, cam, use_rgb=True)

        # Save visualization
        viz_path = XAI_DIR / f"patient_{pid:03d}_AP_L1_gradcam.jpg"
        cv2.imwrite(str(viz_path), cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

        # Compute anatomical overlap with segmentation mask
        mask = get_mask(pid, 'AP')
        overlap_result = {'patient_id': pid, 'view': 'AP'}
        if mask is not None:
            # Resize mask to match CAM
            mask_resized = cv2.resize(mask, (cam.shape[1], cam.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            overlap = compute_cam_mask_overlap(cam, mask_resized)
            overlap_result['overlap'] = overlap
            overlap_str = ', '.join([f"{k}={v['iou']:.3f}" for k, v in overlap.items()])
            print(f"OK | IoU: {overlap_str}")
        else:
            overlap_result['overlap'] = None
            print("OK (no mask for overlap)")

        all_overlap_results.append(overlap_result)

    # Aggregate overlap metrics
    valid = [r for r in all_overlap_results if r.get('overlap')]
    if valid:
        print()
        print("  Anatomical Consistency (Grad-CAM vs Segmentation mask):")
        print(f"  {'Vertebra':<10} {'IoU Mean':<12} {'Precision':<12} {'Recall'}")
        print("  " + "-" * 44)
        for level in ['L1', 'L2', 'L3', 'L4']:
            ious  = [r['overlap'][level]['iou']       for r in valid if level in r.get('overlap', {})]
            precs = [r['overlap'][level]['precision']  for r in valid if level in r.get('overlap', {})]
            recs  = [r['overlap'][level]['recall']     for r in valid if level in r.get('overlap', {})]
            if ious:
                print(f"  {level:<10} {np.mean(ious):<12.4f} {np.mean(precs):<12.4f} {np.mean(recs):.4f}")

    # Save results
    results_path = XAI_DIR / 'xai_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_overlap_results, f, indent=2, default=str)
    print()
    print(f"  Visualizations saved: {XAI_DIR}")
    print(f"  XAI results:          {results_path}")
    print()
    print("=" * 60)
    print("PHASE 10 COMPLETE")
    print("=" * 60)
    print("Next: Run phase11_inference_pipeline.py (single X-ray inference)")
