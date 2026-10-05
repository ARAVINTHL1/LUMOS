"""
PHASE 1 — X-ray Preprocessing
================================
Project: Anatomy-Guided Multiview Multitask Deep Learning for Lumbar BMD
         Estimation and Osteoporosis Severity Assessment from X-ray Images

For each AP and Lateral DICOM image of every usable patient (800 total):
  1. Read DICOM pixel array (pydicom)
  2. Handle MONOCHROME1 / MONOCHROME2 photometric interpretation
  3. Apply window/level (if available) or robust percentile clipping
  4. Normalize to [0, 255] uint8
  5. Resize to 512x512 (preserving information)
  6. Save as PNG under processed/preprocessed_images/<patient_id>/AP.png
                                                   and Lateral.png
  7. Save a preprocessing log CSV

RULES:
  - NEVER modify original DICOM files
  - Process batches to avoid RAM exhaustion
  - Log any failures for review
"""

import os
import csv
import json
import time
import numpy as np
import pandas as pd
import pydicom
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR        = Path("d:/LUMOS")
PROCESSED_DIR   = BASE_DIR / "processed"
PREPROC_DIR     = PROCESSED_DIR / "preprocessed_images"
PREPROC_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV      = PROCESSED_DIR / "master_dataset.csv"
LOG_CSV         = PROCESSED_DIR / "preprocessing_log.csv"

# Target size for all preprocessed images
TARGET_SIZE = (512, 512)   # height x width

# ── Helper: DICOM to uint8 image ──────────────────────────────────────────────
def dicom_to_uint8(dcm_path: str) -> np.ndarray:
    """
    Reads a DICOM file and returns a normalized uint8 grayscale image.
    Handles:
      - MONOCHROME1 (inverted: bone = dark) → flip to MONOCHROME2
      - Windowing from DICOM tags if present
      - Robust percentile clipping otherwise
    """
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # Handle Photometric Interpretation
    photo = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
    if photo == 'MONOCHROME1':
        arr = arr.max() - arr  # invert so bone = bright

    # Rescale slope / intercept (Hounsfield units for CT, sometimes used in DXA)
    slope     = float(getattr(ds, 'RescaleSlope', 1) or 1)
    intercept = float(getattr(ds, 'RescaleIntercept', 0) or 0)
    arr = arr * slope + intercept

    # Window/Level from DICOM tags (DXA images often have these)
    wc = getattr(ds, 'WindowCenter', None)
    ww = getattr(ds, 'WindowWidth', None)
    if wc is not None and ww is not None:
        # May be a sequence or a single value
        wc = float(wc[0]) if hasattr(wc, '__len__') else float(wc)
        ww = float(ww[0]) if hasattr(ww, '__len__') else float(ww)
        lo = wc - ww / 2
        hi = wc + ww / 2
        arr = np.clip(arr, lo, hi)
    else:
        # Robust percentile clipping to handle outlier pixels
        lo = np.percentile(arr, 1)
        hi = np.percentile(arr, 99)
        arr = np.clip(arr, lo, hi)

    # Normalize to [0, 255]
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)

    img = arr.astype(np.uint8)
    return img


def preprocess_patient(row: dict) -> dict:
    """
    Preprocesses AP and Lateral DICOMs for one patient.
    Returns a log dict with status per view.
    """
    pid       = int(row['patient_id'])
    ap_path   = row['ap_image_path']
    lat_path  = row['lateral_image_path']

    out_dir   = PREPROC_DIR / f"patient_{pid:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'patient_id': pid,
        'ap_status': 'SKIPPED',
        'ap_output': '',
        'lat_status': 'SKIPPED',
        'lat_output': '',
        'error': '',
    }

    views = [('AP', ap_path), ('Lateral', lat_path)]
    for view_name, dcm_path in views:
        if not dcm_path or str(dcm_path) == 'nan' or str(dcm_path) == 'None':
            continue
        try:
            img = dicom_to_uint8(str(dcm_path))
            # Resize to target
            img_resized = cv2.resize(img, (TARGET_SIZE[1], TARGET_SIZE[0]),
                                     interpolation=cv2.INTER_LANCZOS4)
            # Save as PNG
            out_path = str(out_dir / f"{view_name}.png")
            cv2.imwrite(out_path, img_resized)

            if view_name == 'AP':
                result['ap_status'] = 'OK'
                result['ap_output'] = out_path
            else:
                result['lat_status'] = 'OK'
                result['lat_output'] = out_path

        except Exception as e:
            if view_name == 'AP':
                result['ap_status'] = 'ERROR'
                result['error'] += f'AP: {str(e)[:100]}; '
            else:
                result['lat_status'] = 'ERROR'
                result['error'] += f'Lat: {str(e)[:100]}; '

    return result


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 1: X-ray Preprocessing")
    print("=" * 60)

    master_df = pd.read_csv(MASTER_CSV)
    usable = master_df[
        master_df['has_both_views'] & master_df['has_clinical']
    ].copy()
    print(f"  Patients to preprocess: {len(usable)}")
    print(f"  Output directory: {PREPROC_DIR}")
    print(f"  Target image size: {TARGET_SIZE[1]}x{TARGET_SIZE[0]} px")
    print()

    rows = usable.to_dict(orient='records')

    logs = []
    ok_ap  = 0
    ok_lat = 0
    err_ap = 0
    err_lat = 0

    start_time = time.time()
    batch_size = 50

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start: batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(rows))
        print(f"  Processing patients {batch_start+1}–{batch_end} / {len(rows)} ...", end=' ', flush=True)
        batch_logs = []
        for row in batch:
            log = preprocess_patient(row)
            batch_logs.append(log)
            if log['ap_status']  == 'OK':  ok_ap  += 1
            if log['lat_status'] == 'OK':  ok_lat += 1
            if log['ap_status']  == 'ERROR': err_ap += 1
            if log['lat_status'] == 'ERROR': err_lat += 1
        logs.extend(batch_logs)
        elapsed = time.time() - start_time
        rate = (batch_end) / elapsed
        print(f"done  [rate: {rate:.1f} patients/s]")

    elapsed_total = time.time() - start_time
    print()
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  AP images:  OK={ok_ap}, ERROR={err_ap}")
    print(f"  LAT images: OK={ok_lat}, ERROR={err_lat}")
    print()

    # Update master CSV with preprocessed paths
    log_df = pd.DataFrame(logs)
    log_df.to_csv(LOG_CSV, index=False)
    print(f"  Preprocessing log saved: {LOG_CSV}")

    # Merge preprocessed paths into master_dataset
    path_map = {
        r['patient_id']: {
            'ap_preprocessed_path': r['ap_output'],
            'lateral_preprocessed_path': r['lat_output'],
        }
        for r in logs
    }

    master_df['ap_preprocessed_path'] = master_df['patient_id'].map(
        lambda pid: path_map.get(int(pid), {}).get('ap_preprocessed_path', '')
    )
    master_df['lateral_preprocessed_path'] = master_df['patient_id'].map(
        lambda pid: path_map.get(int(pid), {}).get('lateral_preprocessed_path', '')
    )
    master_df.to_csv(MASTER_CSV, index=False)
    print(f"  master_dataset.csv updated with preprocessed paths.")
    print()

    # Verify a sample: print image stats for first 5 patients
    print("  Spot-check — first 5 preprocessed AP images:")
    for r in logs[:5]:
        pid = r['patient_id']
        ap_out = r['ap_output']
        if ap_out and os.path.exists(ap_out):
            img = cv2.imread(ap_out, cv2.IMREAD_GRAYSCALE)
            print(f"    Patient {pid:03d}: shape={img.shape}, min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
        else:
            print(f"    Patient {pid:03d}: AP not found ({r['ap_status']})")

    print()
    print("=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print("Next: PHASE 2 — Vertebral Segmentation (SAM-based pseudo-labeling)")
