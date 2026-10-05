"""
PHASE 2B — Segmentation Evaluation
====================================
After manually verifying/correcting SAM pseudo-labels, run this script to:
  1. Compute Dice, IoU, Precision, Recall for each vertebra (L1-L4)
  2. Report per-vertebra and overall metrics
  3. Save evaluation results to JSON
  4. Decide whether segmentation quality is acceptable before proceeding

REQUIREMENT:
  - Manually corrected masks must be in: processed/segmentation/verified/
  - Predicted masks (from SAM or trained model) in: processed/segmentation/pseudo_labels/
    (or processed/segmentation/predicted/)
  - Mask format: single-channel PNG, pixel values 0=bg,1=L1,2=L2,3=L3,4=L4

ACCEPTANCE CRITERION (from project plan):
  - Dice >= 0.85 per vertebra before proceeding to BMD model training
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from collections import defaultdict

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("d:/LUMOS")
PROCESSED_DIR = BASE_DIR / "processed"
VERIFIED_DIR  = PROCESSED_DIR / "segmentation" / "verified"
PREDICTED_DIR = PROCESSED_DIR / "segmentation" / "pseudo_labels"
RESULTS_DIR   = BASE_DIR / "results" / "segmentation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEG_MANIFEST  = PROCESSED_DIR / "segmentation_manifest.csv"

N_CLASSES  = 5  # 0=bg, 1=L1, 2=L2, 3=L3, 4=L4
CLASS_NAMES = {0: 'background', 1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4'}

DICE_THRESHOLD = 0.85  # Minimum acceptable Dice before proceeding

# ── Metric functions ──────────────────────────────────────────────────────────
def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, n_classes: int = 5):
    """
    Computes per-class Dice, IoU, Precision, Recall.
    Returns dict keyed by class index.
    """
    results = {}
    for cls in range(1, n_classes):  # skip background
        pred_bin = (pred_mask == cls).astype(np.float32)
        gt_bin   = (gt_mask   == cls).astype(np.float32)

        tp = (pred_bin * gt_bin).sum()
        fp = (pred_bin * (1 - gt_bin)).sum()
        fn = ((1 - pred_bin) * gt_bin).sum()

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        iou  = tp / (tp + fp + fn + 1e-7)
        prec = tp / (tp + fp + 1e-7)
        rec  = tp / (tp + fn + 1e-7)

        results[cls] = {
            'dice': float(dice),
            'iou':  float(iou),
            'precision': float(prec),
            'recall': float(rec),
        }
    return results


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 2B: Segmentation Evaluation")
    print("=" * 60)

    # Load manifest
    if not SEG_MANIFEST.exists():
        print("  ERROR: segmentation_manifest.csv not found.")
        print("  Run phase2_sam_pseudolabeling.py first.")
        exit(1)

    manifest_df = pd.read_csv(SEG_MANIFEST)

    # Find images with both verified GT and predicted masks
    eval_records = []
    for _, row in manifest_df.iterrows():
        pid  = int(row['patient_id'])
        view = row['view']

        # Look for verified (GT) mask
        gt_name  = f"patient_{pid:03d}_{view}_mask.png"
        gt_path  = VERIFIED_DIR / gt_name
        pred_path = Path(row['mask_path'])

        if gt_path.exists() and pred_path.exists():
            eval_records.append({
                'patient_id': pid,
                'view': view,
                'gt_path': str(gt_path),
                'pred_path': str(pred_path),
            })

    print(f"  Images with both GT and predicted masks: {len(eval_records)}")

    if len(eval_records) == 0:
        print()
        print("  No verified masks found yet.")
        print(f"  Please manually verify/correct masks and place them in:")
        print(f"    {VERIFIED_DIR}")
        print()
        print("  Mask filename format:  patient_NNN_AP_mask.png")
        print("                          patient_NNN_Lateral_mask.png")
        print()
        print("  Pixel values: 0=background, 1=L1, 2=L2, 3=L3, 4=L4")
        print()
        print("  Recommended annotation tools:")
        print("    - LabelMe (free, browser-based)")
        print("    - ITK-SNAP (medical imaging)")
        print("    - 3D Slicer (medical imaging)")
        print()
        print("  Once verified masks are placed, re-run this script.")
        exit(0)

    # Compute metrics
    all_metrics = defaultdict(list)

    for rec in eval_records:
        gt_mask   = cv2.imread(rec['gt_path'],   cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(rec['pred_path'],  cv2.IMREAD_GRAYSCALE)

        if gt_mask is None or pred_mask is None:
            print(f"  WARNING: Could not read masks for patient {rec['patient_id']} {rec['view']}")
            continue

        # Ensure same size
        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        metrics = compute_metrics(pred_mask, gt_mask, N_CLASSES)
        for cls, m in metrics.items():
            all_metrics[cls].append(m)

    # Aggregate
    print()
    print("  Segmentation Metrics (mean ± std over evaluated images):")
    print()
    print(f"  {'Vertebra':<12} {'Dice':<12} {'IoU':<12} {'Precision':<12} {'Recall':<12}")
    print("  " + "-" * 56)

    per_class_results = {}
    overall_dice_vals = []

    for cls in range(1, N_CLASSES):
        m_list = all_metrics[cls]
        if not m_list:
            continue
        mean_dice = np.mean([m['dice'] for m in m_list])
        std_dice  = np.std( [m['dice'] for m in m_list])
        mean_iou  = np.mean([m['iou']  for m in m_list])
        mean_prec = np.mean([m['precision'] for m in m_list])
        mean_rec  = np.mean([m['recall']    for m in m_list])

        tag = "✓" if mean_dice >= DICE_THRESHOLD else "✗ BELOW THRESHOLD"
        print(f"  {CLASS_NAMES[cls]:<12} {mean_dice:.4f}±{std_dice:.4f}  {mean_iou:.4f}       {mean_prec:.4f}       {mean_rec:.4f}  {tag}")

        per_class_results[CLASS_NAMES[cls]] = {
            'dice_mean': float(mean_dice),
            'dice_std':  float(std_dice),
            'iou_mean':  float(mean_iou),
            'precision_mean': float(mean_prec),
            'recall_mean':    float(mean_rec),
            'n_samples': len(m_list),
        }
        overall_dice_vals.extend([m['dice'] for m in m_list])

    overall_dice = float(np.mean(overall_dice_vals)) if overall_dice_vals else 0.0
    print()
    print(f"  Overall mean Dice: {overall_dice:.4f}")
    print()

    if overall_dice >= DICE_THRESHOLD:
        print(f"  ✅ Segmentation quality ACCEPTABLE (Dice >= {DICE_THRESHOLD})")
        print("  → Proceed to Phase 3: Train segmentation model")
        proceed = True
    else:
        print(f"  ❌ Segmentation quality BELOW THRESHOLD (Dice < {DICE_THRESHOLD})")
        print("  → Improve segmentation before proceeding")
        proceed = False

    # Save results
    results = {
        'n_eval_pairs': len(eval_records),
        'dice_threshold': DICE_THRESHOLD,
        'overall_dice_mean': overall_dice,
        'proceed_to_training': proceed,
        'per_class': per_class_results,
    }
    results_path = RESULTS_DIR / 'segmentation_evaluation.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")
    print()
    print("=" * 60)
    print("PHASE 2B COMPLETE")
    print("=" * 60)
