"""
PHASE 4 — Vertebral ROI Dataset Generation
============================================
Project: Anatomy-Guided Multiview Multitask Deep Learning for Lumbar BMD
         Estimation and Osteoporosis Severity Assessment from X-ray Images

After segmentation model is trained (Phase 3), this script:
  1. Loads predicted masks for all 800 patients
  2. Crops L1-L4 ROIs from each preprocessed image
  3. Saves ROIs as PNG: processed/vertebral_rois/patient_NNN/AP/L1.png ... L4.png
  4. Creates a comprehensive ROI manifest CSV

ROI structure:
  processed/
    vertebral_rois/
      patient_001/
        AP/
          L1.png  L2.png  L3.png  L4.png
        Lateral/
          L1.png  L2.png  L3.png  L4.png
      patient_002/
        ...

IMPORTANT:
  - This script requires completed Phase 3 (trained segmentation model)
  - Alternatively, if manually annotated masks exist in verified/, those can be used
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(os.environ.get("LUMOS_DIR", Path(__file__).resolve().parent))
PROCESSED_DIR = BASE_DIR / "processed"
PREPROC_DIR   = PROCESSED_DIR / "preprocessed_images"
PREDICTED_DIR = PROCESSED_DIR / "segmentation" / "predicted"
VERIFIED_DIR  = PROCESSED_DIR / "segmentation" / "verified"
PSEUDO_DIR    = PROCESSED_DIR / "segmentation" / "pseudo_labels"
ROI_DIR       = PROCESSED_DIR / "vertebral_rois"
ROI_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV    = PROCESSED_DIR / "master_dataset.csv"
ROI_MANIFEST  = PROCESSED_DIR / "roi_manifest.csv"

# ROI settings
ROI_SIZE      = 128   # Each vertebra ROI resized to 128x128
PADDING_FRAC  = 0.15  # Padding fraction around bounding box
LABEL_MAP     = {1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4'}


def get_mask_path(pid: int, view: str) -> Path:
    """Return mask path: prefer verified > predicted > pseudo_labels."""
    fname = f"patient_{pid:03d}_{view}_mask.png"
    for d in [VERIFIED_DIR, PREDICTED_DIR, PSEUDO_DIR]:
        p = d / fname
        if p.exists():
            return p
    return None


def extract_roi(image: np.ndarray, mask: np.ndarray, label_id: int,
                roi_size: int = 128, padding_frac: float = 0.15) -> np.ndarray:
    """
    Extracts a padded bounding-box ROI for a given vertebra label.
    Returns resized ROI of shape (roi_size, roi_size).
    """
    region = (mask == label_id).astype(np.uint8)
    if region.sum() == 0:
        # Vertebra not found in mask — return blank
        return np.zeros((roi_size, roi_size), dtype=np.uint8)

    # Bounding box
    rows = np.where(region.any(axis=1))[0]
    cols = np.where(region.any(axis=0))[0]
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())

    # Add padding
    h, w = image.shape[:2]
    pad_r = max(1, int((r_max - r_min) * padding_frac))
    pad_c = max(1, int((c_max - c_min) * padding_frac))

    r_min = max(0, r_min - pad_r)
    r_max = min(h - 1, r_max + pad_r)
    c_min = max(0, c_min - pad_c)
    c_max = min(w - 1, c_max + pad_c)

    roi = image[r_min:r_max+1, c_min:c_max+1]
    roi_resized = cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_LANCZOS4)
    return roi_resized


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 4: Vertebral ROI Dataset Generation")
    print("=" * 60)

    master_df = pd.read_csv(MASTER_CSV)
    usable = master_df[
        master_df['has_both_views'].astype(bool) & master_df['has_clinical'].astype(bool)
    ].copy()
    print(f"  Patients to process: {len(usable)}")
    print(f"  ROI output dir: {ROI_DIR}")
    print(f"  ROI size: {ROI_SIZE}x{ROI_SIZE} px")
    print()

    roi_records = []
    n_ok     = 0
    n_miss   = 0
    n_blank  = 0

    for _, row in usable.iterrows():
        pid = int(row['patient_id'])

        for view in ['AP', 'Lateral']:
            # Get preprocessed image
            view_key = 'ap_preprocessed_path' if view == 'AP' else 'lateral_preprocessed_path'
            img_path = row.get(view_key, '')
            if not img_path or str(img_path) == 'nan' or not os.path.exists(str(img_path)):
                img_path = str(PREPROC_DIR / f"patient_{pid:03d}" / f"{view}.png")

            # Get mask
            mask_path = get_mask_path(pid, view)

            if not os.path.exists(img_path):
                print(f"  MISSING image: {img_path}")
                n_miss += 1
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                n_miss += 1
                continue

            if mask_path is None or not mask_path.exists():
                # No mask available — save blank ROIs and flag
                mask = np.zeros_like(img)
                n_blank += 1
            else:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros_like(img)
                    n_blank += 1

            # Create output dirs
            roi_patient_dir = ROI_DIR / f"patient_{pid:03d}" / view
            roi_patient_dir.mkdir(parents=True, exist_ok=True)

            # Extract and save each vertebra ROI
            roi_rec = {
                'patient_id': pid,
                'view': view,
                'image_path': img_path,
                'mask_path': str(mask_path) if mask_path else '',
                'mask_available': mask_path is not None and mask_path.exists(),
                'bmd': float(row['bmd']),
                'bmd_L1-L2': float(row['bmd_L1-L2']),
                'bmd_L1-L3': float(row['bmd_L1-L3']),
                'bmd_L1-L4': float(row['bmd_L1-L4']),
                'bmd_L2-L3': float(row['bmd_L2-L3']),
                'bmd_L2-L4': float(row['bmd_L2-L4']),
                'bmd_L3-L4': float(row['bmd_L3-L4']),
                'Osteoporosis': int(row['Osteoporosis']),
                'split': str(row.get('split', 'unassigned')),
            }

            for label_id, label_name in LABEL_MAP.items():
                roi = extract_roi(img, mask, label_id, ROI_SIZE, PADDING_FRAC)
                out_path = str(roi_patient_dir / f"{label_name}.png")
                cv2.imwrite(out_path, roi)
                roi_rec[f'{label_name}_roi_path'] = out_path
                roi_rec[f'{label_name}_has_mask'] = int((mask == label_id).sum()) > 0

            roi_records.append(roi_rec)
            n_ok += 1

    roi_df = pd.DataFrame(roi_records)
    roi_df.to_csv(ROI_MANIFEST, index=False)

    print(f"  Patients processed: {n_ok}")
    print(f"  Missing images:     {n_miss}")
    print(f"  No mask (blank):    {n_blank}")
    print(f"  ROI manifest:       {ROI_MANIFEST}")
    print()

    # Sample verification
    print("  Sample — first 5 patients, L1 ROI stats (AP view):")
    for rec in roi_records[:5]:
        if rec['view'] == 'AP':
            path = rec.get('L1_roi_path', '')
            if path and os.path.exists(path):
                roi = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                print(f"    Patient {rec['patient_id']:03d}: L1 ROI shape={roi.shape}, "
                      f"mean={roi.mean():.1f}, has_mask={rec['L1_has_mask']}")

    print()
    print("=" * 60)
    print("PHASE 4 COMPLETE — ROI Dataset Generated")
    print("=" * 60)
    print("Next: Run phase5_train_bmd_model.py (Shared CNN + Transformer + Multitask)")
