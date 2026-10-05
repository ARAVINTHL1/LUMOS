"""
PHASE 0 — Dataset Verification
================================
Project: Anatomy-Guided Multiview Multitask Deep Learning for Lumbar BMD
         Estimation and Osteoporosis Severity Assessment from X-ray Images

This script:
  1. Reads the clinical Excel (lumos_clinical_data.xlsx)
  2. Recursively scans lumos_x/ for all DICOM files
  3. Extracts patient IDs from folder names
  4. Reads DICOM metadata to confirm AP vs Lateral view
  5. Matches patients to clinical data
  6. Creates master_dataset.csv
  7. Prints comprehensive statistics
  8. Creates patient-level train/val/test splits (70/15/15, stratified)
  9. Saves split manifests

RULES (from project prompt):
  - Never modify original LUMOS files
  - Never split the same patient across train/val/test
  - Never treat AP and Lateral as separate patients
  - Splits must be stratified by Osteoporosis class
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from collections import defaultdict

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR        = Path("d:/LUMOS")
XRAY_DIR        = BASE_DIR / "lumos_x"
CLINICAL_FILE   = BASE_DIR / "lumos_clinical_data.xlsx"
PROCESSED_DIR   = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

OUTPUT_MASTER   = PROCESSED_DIR / "master_dataset.csv"
OUTPUT_TRAIN    = PROCESSED_DIR / "train.csv"
OUTPUT_VAL      = PROCESSED_DIR / "validation.csv"
OUTPUT_TEST     = PROCESSED_DIR / "test.csv"
OUTPUT_STATS    = PROCESSED_DIR / "dataset_statistics.json"

# ── 1. Load Clinical Excel ───────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Clinical Excel")
print("=" * 60)

clinical_df = pd.read_excel(CLINICAL_FILE)
print(f"  Clinical records loaded: {len(clinical_df)}")
print(f"  Columns: {list(clinical_df.columns)}")
print()

key_bmd_cols = ['bmd', 'bmd_L1-L2', 'bmd_L1-L3', 'bmd_L1-L4',
                'bmd_L2-L3', 'bmd_L2-L4', 'bmd_L3-L4']
print("  Key BMD columns present:", [c for c in key_bmd_cols if c in clinical_df.columns])
print()

# ── 2. Scan X-ray Directory ──────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Scanning lumos_x/ for DICOM files")
print("=" * 60)

patient_dicom_map = defaultdict(list)
dicom_extensions = {'.dcm', '.DCM', '.Dcm', '.dicom'}

all_dcm_files = []
for root, dirs, files in os.walk(XRAY_DIR):
    for fname in files:
        if any(fname.endswith(ext) for ext in dicom_extensions):
            full_path = Path(root) / fname
            all_dcm_files.append(full_path)

print(f"  Total DICOM files found: {len(all_dcm_files)}")

# Extract patient IDs from folder structure
patient_folder_pattern = re.compile(r'lumos_x_(\d+)$', re.IGNORECASE)

for dcm_path in all_dcm_files:
    parts = dcm_path.parts
    patient_id = None
    for part in parts:
        m = patient_folder_pattern.match(part)
        if m:
            patient_id = int(m.group(1))
            break
    if patient_id is not None:
        patient_dicom_map[patient_id].append(dcm_path)

print(f"  Unique patients with DICOM data: {len(patient_dicom_map)}")
print()

# ── 3. Identify AP vs Lateral ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Identifying AP vs Lateral views")
print("=" * 60)

def get_view_from_filename(dcm_path: Path) -> str:
    """Heuristic: _1 = AP, _2 = Lateral based on LUMOS naming convention."""
    stem = dcm_path.stem.lower()
    if stem.endswith('_1'):
        return 'AP'
    elif stem.endswith('_2'):
        return 'Lateral'
    else:
        return 'Unknown'

def read_dicom_view_metadata(dcm_path: Path) -> dict:
    """Try to read DICOM metadata to get view information."""
    info = {'view_from_meta': None, 'rows': None, 'cols': None, 'modality': None}
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        info['rows'] = getattr(ds, 'Rows', None)
        info['cols'] = getattr(ds, 'Columns', None)
        info['modality'] = getattr(ds, 'Modality', None)
        view_pos = getattr(ds, 'ViewPosition', None)
        if view_pos:
            vp = str(view_pos).upper().strip()
            if 'AP' in vp or 'PA' in vp:
                info['view_from_meta'] = 'AP'
            elif 'LAT' in vp:
                info['view_from_meta'] = 'Lateral'
        if info['view_from_meta'] is None:
            for attr in ['ProtocolName', 'SeriesDescription']:
                val = str(getattr(ds, attr, '') or '').upper()
                if 'AP' in val and 'LAT' not in val:
                    info['view_from_meta'] = 'AP'
                    break
                elif 'LAT' in val:
                    info['view_from_meta'] = 'Lateral'
                    break
    except Exception:
        pass
    return info

print("  Sampling first 20 patients to verify filename vs DICOM metadata agreement...")
sample_pids = sorted(patient_dicom_map.keys())[:20]
agreements, disagreements, unknown_meta = 0, 0, 0

for pid in sample_pids:
    for dcm_path in patient_dicom_map[pid]:
        view_fname = get_view_from_filename(dcm_path)
        meta = read_dicom_view_metadata(dcm_path)
        view_meta = meta['view_from_meta']
        if view_meta is None:
            unknown_meta += 1
        elif view_fname == view_meta:
            agreements += 1
        else:
            disagreements += 1
            print(f"    DISAGREEMENT: {dcm_path.name} -> filename={view_fname}, meta={view_meta}")

print(f"  Agreements: {agreements}, Disagreements: {disagreements}, Meta unknown: {unknown_meta}")
print()

# ── 4. Build Patient Records ─────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Building patient-level records")
print("=" * 60)

records = []
patients_ap_only = []
patients_lat_only = []
patients_complete = []

for pid in sorted(patient_dicom_map.keys()):
    dcm_files = patient_dicom_map[pid]
    ap_paths  = [p for p in dcm_files if get_view_from_filename(p) == 'AP']
    lat_paths = [p for p in dcm_files if get_view_from_filename(p) == 'Lateral']

    ap_path  = str(ap_paths[0])  if ap_paths  else None
    lat_path = str(lat_paths[0]) if lat_paths else None

    clin_row = clinical_df[clinical_df['patient_id'] == pid]
    has_clinical = len(clin_row) > 0

    rec = {
        'patient_id': pid,
        'ap_image_path': ap_path,
        'lateral_image_path': lat_path,
        'has_ap': ap_path is not None,
        'has_lateral': lat_path is not None,
        'has_both_views': ap_path is not None and lat_path is not None,
        'n_dicom_files': len(dcm_files),
        'has_clinical': has_clinical,
    }

    if has_clinical:
        row = clin_row.iloc[0]
        for col in ['age', 'gender', 'ethnicity', 'height', 'weight', 'BMI',
                    'Osteoporosis', 'bmd', 'bmd_L1-L2', 'bmd_L1-L3', 'bmd_L1-L4',
                    'bmd_L2-L3', 'bmd_L2-L4', 'bmd_L3-L4', 't_value']:
            rec[col] = row[col] if col in clinical_df.columns else None
    else:
        for col in ['age', 'gender', 'ethnicity', 'height', 'weight', 'BMI',
                    'Osteoporosis', 'bmd', 'bmd_L1-L2', 'bmd_L1-L3', 'bmd_L1-L4',
                    'bmd_L2-L3', 'bmd_L2-L4', 'bmd_L3-L4', 't_value']:
            rec[col] = None

    records.append(rec)

    if ap_path and lat_path:
        patients_complete.append(pid)
    elif ap_path:
        patients_ap_only.append(pid)
    elif lat_path:
        patients_lat_only.append(pid)

clinical_pids = set(clinical_df['patient_id'].tolist())
xray_pids = set(patient_dicom_map.keys())

print(f"  Patients with X-ray data:            {len(xray_pids)}")
print(f"  Patients with clinical data:          {len(clinical_pids)}")
print(f"  Matched (X-ray + clinical):           {len(xray_pids & clinical_pids)}")
print(f"  Clinical only (no X-ray folder):      {len(clinical_pids - xray_pids)}")
print(f"  X-ray only (no clinical row):         {len(xray_pids - clinical_pids)}")
print()
print(f"  Complete (AP + Lateral + Clinical):   {len([r for r in records if r['has_both_views'] and r['has_clinical']])}")
print(f"  AP only (+ clinical):                 {len(patients_ap_only)}")
print(f"  Lateral only (+ clinical):            {len(patients_lat_only)}")
print()

master_df = pd.DataFrame(records)

# ── 5. Statistics ────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Dataset Statistics")
print("=" * 60)

usable = master_df[master_df['has_both_views'] & master_df['has_clinical']].copy()
print(f"  USABLE patients (AP + Lateral + Clinical): {len(usable)}")
print()

class_counts = {}
if 'Osteoporosis' in usable.columns:
    class_counts = usable['Osteoporosis'].value_counts().sort_index()
    print("  Osteoporosis class distribution (usable patients):")
    for cls, cnt in class_counts.items():
        label = {0: 'Normal', 1: 'Osteopenia', 2: 'Osteoporosis'}.get(int(cls), str(cls))
        print(f"    Class {int(cls)} ({label}): {cnt} patients ({100*cnt/len(usable):.1f}%)")
    print()

bmd_vals = pd.Series(dtype=float)
if 'bmd' in usable.columns:
    bmd_vals = usable['bmd'].dropna()
    print(f"  BMD statistics (overall bmd column):")
    print(f"    Mean:  {bmd_vals.mean():.4f} g/cm²")
    print(f"    Std:   {bmd_vals.std():.4f} g/cm²")
    print(f"    Min:   {bmd_vals.min():.4f} g/cm²")
    print(f"    Max:   {bmd_vals.max():.4f} g/cm²")
    print(f"    Missing: {usable['bmd'].isna().sum()}")
    print()

if 'age' in usable.columns:
    age_vals = usable['age'].dropna()
    print(f"  Age statistics:")
    print(f"    Mean:  {age_vals.mean():.1f} years")
    print(f"    Std:   {age_vals.std():.1f} years")
    print(f"    Range: {age_vals.min():.0f} - {age_vals.max():.0f}")
    print()

if 'gender' in usable.columns:
    print(f"  Gender distribution:")
    print(usable['gender'].value_counts().to_string(header=False))
    print()

# ── 6. Patient-Level Split ───────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6: Patient-level stratified 70/15/15 split")
print("=" * 60)

from sklearn.model_selection import train_test_split

usable_sorted = usable.sort_values('patient_id').reset_index(drop=True)
patient_ids = usable_sorted['patient_id'].values
labels = usable_sorted['Osteoporosis'].values

try:
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        patient_ids, labels, test_size=0.30, stratify=labels, random_state=RANDOM_SEED
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, stratify=temp_labels, random_state=RANDOM_SEED
    )
    print(f"  Stratified split successful.")
except Exception as e:
    print(f"  WARNING: Stratified split failed ({e}). Using random split.")
    shuffled = list(patient_ids)
    random.shuffle(shuffled)
    n = len(shuffled)
    train_ids = shuffled[:int(0.70*n)]
    val_ids   = shuffled[int(0.70*n):int(0.85*n)]
    test_ids  = shuffled[int(0.85*n):]

train_set = set(train_ids)
val_set   = set(val_ids)
test_set  = set(test_ids)

def assign_split(pid):
    if pid in train_set: return 'train'
    if pid in val_set:   return 'validation'
    if pid in test_set:  return 'test'
    return 'unassigned'

usable_sorted['split'] = usable_sorted['patient_id'].apply(assign_split)
master_df['split'] = master_df['patient_id'].apply(assign_split)

print(f"  Train:      {len(train_ids)} patients")
print(f"  Validation: {len(val_ids)} patients")
print(f"  Test:       {len(test_ids)} patients")

for split_name in ['train', 'validation', 'test']:
    sdf = usable_sorted[usable_sorted['split'] == split_name]
    cc = sdf['Osteoporosis'].value_counts().sort_index()
    print(f"\n  {split_name.capitalize()} class breakdown:")
    for cls, cnt in cc.items():
        label = {0: 'Normal', 1: 'Osteopenia', 2: 'Osteoporosis'}.get(int(cls), str(cls))
        print(f"    {cls} ({label}): {cnt}")
print()

# ── 7. Save Outputs ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 7: Saving outputs")
print("=" * 60)

master_df.to_csv(OUTPUT_MASTER, index=False)
print(f"  {OUTPUT_MASTER}")

train_df = usable_sorted[usable_sorted['split'] == 'train']
val_df   = usable_sorted[usable_sorted['split'] == 'validation']
test_df  = usable_sorted[usable_sorted['split'] == 'test']

train_df.to_csv(OUTPUT_TRAIN, index=False)
val_df.to_csv(OUTPUT_VAL, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)
print(f"  {OUTPUT_TRAIN} ({len(train_df)} patients)")
print(f"  {OUTPUT_VAL} ({len(val_df)} patients)")
print(f"  {OUTPUT_TEST} ({len(test_df)} patients)")

stats = {
    'total_clinical_patients': int(len(clinical_pids)),
    'total_xray_patients': int(len(xray_pids)),
    'matched_patients': int(len(xray_pids & clinical_pids)),
    'usable_patients': int(len(usable)),
    'complete_both_views': int(len(patients_complete)),
    'ap_only_patients': int(len(patients_ap_only)),
    'lateral_only_patients': int(len(patients_lat_only)),
    'osteoporosis_class_distribution': {
        int(k): int(v) for k, v in class_counts.items()
    } if len(class_counts) > 0 else {},
    'bmd_mean': float(bmd_vals.mean()) if len(bmd_vals) > 0 else None,
    'bmd_std':  float(bmd_vals.std())  if len(bmd_vals) > 0 else None,
    'bmd_min':  float(bmd_vals.min())  if len(bmd_vals) > 0 else None,
    'bmd_max':  float(bmd_vals.max())  if len(bmd_vals) > 0 else None,
    'train_patients': int(len(train_df)),
    'val_patients':   int(len(val_df)),
    'test_patients':  int(len(test_df)),
    'random_seed': RANDOM_SEED,
}
with open(OUTPUT_STATS, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"  {OUTPUT_STATS}")
print()

print("=" * 60)
print("PHASE 0 COMPLETE")
print("=" * 60)
print("Next: PHASE 1 — X-ray Preprocessing (preprocess_xrays.py)")
