"""
PHASE 2 - Vertebral Segmentation: SAM-Based Pseudo-Labeling
=============================================================
Project: Anatomy-Guided Multiview Multitask Deep Learning for Lumbar BMD
         Estimation and Osteoporosis Severity Assessment from X-ray Images

This script:
  1. Selects a subset of ~200 preprocessed images for initial verified segmentation
  2. Uses SAM (Segment Anything Model) with heuristic point prompts for L1-L4
  3. Generates initial pseudo-label masks (class 0=bg, 1=L1, 2=L2, 3=L3, 4=L4)
  4. Saves masks as single-channel PNG files alongside a manifest
  5. Creates a visualization grid for human review/correction

IMPORTANT:
  - SAM pseudo-labels MUST be manually verified/corrected before training
  - Do NOT trust pseudo-labels blindly
  - This script generates a starting point for annotation, NOT ground truth

SETUP:
  pip install torch torchvision
  pip install git+https://github.com/facebookresearch/segment-anything.git
  Download SAM checkpoint:
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

RUNS ON: Google Colab (GPU recommended) or local with CUDA GPU
"""

import os
import json
import random
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
SEG_DIR       = PROCESSED_DIR / "segmentation" / "pseudo_labels"
VIZ_DIR       = PROCESSED_DIR / "segmentation" / "visualizations"
SEG_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

MASTER_CSV    = PROCESSED_DIR / "master_dataset.csv"
SEG_MANIFEST  = PROCESSED_DIR / "segmentation_manifest.csv"

RANDOM_SEED   = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Number of images to generate pseudo-labels for (None = all usable patients)
N_SUBSET = None
# SAM checkpoint path (download separately)
SAM_CANDIDATES = [
    BASE_DIR / "sam_vit_b_01ec64.pth",
    Path.home() / "sam_vit_b_01ec64.pth",
    Path("C:/Users/aravi/sam_vit_b_01ec64.pth"),
]
SAM_CHECKPOINT = next((p for p in SAM_CANDIDATES if p.exists() and p.stat().st_size > 300_000_000), BASE_DIR / "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"

# Vertebra label colors for visualization (BGR)
LABEL_COLORS = {
    0: (0,   0,   0),    # background - black
    1: (255, 80,  80),   # L1 - red
    2: (80,  255, 80),   # L2 - green
    3: (80,  80,  255),  # L3 - blue
    4: (255, 255, 80),   # L4 - yellow
}

# ── Heuristic point prompts for lumbar vertebrae ──────────────────────────────
def get_vertebra_prompts_ap(h: int, w: int) -> dict:
    """
    Heuristic AP-view point prompts for L1-L4.
    The lumbar spine in an AP X-ray typically occupies roughly the middle
    40-80% of the image height, centered horizontally.
    These are approximate starting points - SAM will refine them.
    """
    cx = w // 2
    # Approximate fractional positions (top of image = 0, bottom = 1)
    # L1 ≈ 40%, L2 ≈ 52%, L3 ≈ 64%, L4 ≈ 76% of image height
    return {
        1: (cx, int(h * 0.40)),
        2: (cx, int(h * 0.52)),
        3: (cx, int(h * 0.64)),
        4: (cx, int(h * 0.76)),
    }


def get_vertebra_prompts_lateral(h: int, w: int) -> dict:
    """
    Heuristic Lateral-view point prompts for L1-L4.
    In lateral view, the spine is along one edge; vertebrae stack vertically
    in a similar fraction of image height but may be offset horizontally.
    """
    cx = int(w * 0.50)  # spine tends to be near center in cropped lateral
    return {
        1: (cx, int(h * 0.38)),
        2: (cx, int(h * 0.50)),
        3: (cx, int(h * 0.62)),
        4: (cx, int(h * 0.74)),
    }


# ── SAM-based segmentation ────────────────────────────────────────────────────
def segment_vertebrae_sam(
    image_rgb: np.ndarray,
    view: str,
    sam_predictor,
) -> np.ndarray:
    """
    Uses SAM to generate per-vertebra masks for L1-L4.
    Returns a single-channel label mask (0=bg, 1=L1, 2=L2, 3=L3, 4=L4).
    """
    h, w = image_rgb.shape[:2]
    sam_predictor.set_image(image_rgb)

    if view == 'AP':
        prompts = get_vertebra_prompts_ap(h, w)
    else:
        prompts = get_vertebra_prompts_lateral(h, w)

    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for label_id, (px, py) in sorted(prompts.items()):
        point_coords = np.array([[px, py]])
        point_labels = np.array([1])  # foreground

        masks, scores, logits = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        # Pick smallest mask (most focused on single vertebra)
        mask_areas = [m.sum() for m in masks]
        best_idx = np.argmin(mask_areas)
        best_mask = masks[best_idx].astype(bool)

        # Only assign pixels not already labelled (prevents overlap)
        combined_mask[best_mask & (combined_mask == 0)] = label_id

    return combined_mask


def save_mask_png(mask: np.ndarray, path: str):
    """Saves single-channel label mask as PNG (values 0-4)."""
    cv2.imwrite(path, mask)


def create_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    prompts: dict,
    save_path: str,
    patient_id: int,
    view: str,
):
    """Creates a side-by-side visualization for human review."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: original image with prompt points
    axes[0].imshow(image, cmap='gray')
    for lid, (px, py) in prompts.items():
        color_rgb = tuple(c/255 for c in LABEL_COLORS[lid][::-1])  # BGR→RGB
        axes[0].plot(px, py, '*', color=color_rgb, markersize=15,
                     markeredgecolor='white', markeredgewidth=1.5)
        axes[0].text(px+10, py, f'L{lid}', color='white', fontsize=10,
                     bbox=dict(facecolor=color_rgb, alpha=0.7, pad=2))
    axes[0].set_title(f'Patient {patient_id:03d} - {view}\nOriginal + Prompt Points',
                       fontsize=12)
    axes[0].axis('off')

    # Right: image with colored mask overlay
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    for label_id, color in LABEL_COLORS.items():
        if label_id == 0:
            continue
        region = (mask == label_id)
        color_rgb = np.array(color[::-1], dtype=np.float32) / 255.0
        overlay[region] = overlay[region] * 0.5 + color_rgb * 127

    axes[1].imshow(overlay.astype(np.uint8))
    patches = [
        mpatches.Patch(color=[c/255 for c in LABEL_COLORS[i][::-1]],
                       label=f'L{i}')
        for i in range(1, 5)
    ]
    patches.insert(0, mpatches.Patch(color='black', label='Background'))
    axes[1].legend(handles=patches, loc='lower right', fontsize=9)
    axes[1].set_title(f'SAM Pseudo-Label Mask\n(REQUIRES MANUAL VERIFICATION)', fontsize=12)
    axes[1].axis('off')

    plt.suptitle(
        f'WARNING PSEUDO-LABELS - Must be verified/corrected before model training',
        color='red', fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PHASE 2: Vertebral Segmentation - SAM Pseudo-Labeling")
    parser.add_argument("--patients", type=str, default="all",
                        help="Number of patients to process (integer like 20, 200, or 'all' for all 800 usable patients)")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 2: Vertebral Segmentation - SAM Pseudo-Labeling")
    print("=" * 60)

    # Check SAM checkpoint
    if not SAM_CHECKPOINT.exists():
        print()
        print("  WARNING SAM checkpoint NOT found at:")
        print(f"    {SAM_CHECKPOINT}")
        print()
        print("  To download the SAM ViT-B checkpoint, run:")
        print("    Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile sam_vit_b_01ec64.pth")
        print()
        print("  Once downloaded, re-run this script.")
        print()
        print("  For now, generating the SEGMENTATION MANIFEST without running SAM.")
    else:
        print(f"  SAM checkpoint found: {SAM_CHECKPOINT}")

    # Load master dataset
    master_df = pd.read_csv(MASTER_CSV)
    usable = master_df[
        master_df['has_both_views'].astype(bool) & master_df['has_clinical'].astype(bool)
    ].copy()
    print(f"  Total usable patients with both views: {len(usable)}")

    # Parse patient selection
    if args.patients.lower() == 'all':
        sampled = usable.reset_index(drop=True)
        print(f"  Processing ALL {len(sampled)} patients ({len(sampled)*2} images)")
    else:
        try:
            n_target = int(args.patients)
            if n_target < len(usable):
                from sklearn.model_selection import train_test_split
                sampled, _ = train_test_split(
                    usable,
                    train_size=n_target,
                    stratify=usable['Osteoporosis'],
                    random_state=RANDOM_SEED
                )
                sampled = sampled.reset_index(drop=True)
                print(f"  Processing subset of {len(sampled)} patients (stratified)")
            else:
                sampled = usable.reset_index(drop=True)
                print(f"  Processing ALL {len(sampled)} patients ({len(sampled)*2} images)")
        except ValueError:
            sampled = usable.reset_index(drop=True)
            print(f"  Unrecognized patient count; defaulting to ALL {len(sampled)} patients")

    print(f"  Class distribution:")
    for cls, cnt in sampled['Osteoporosis'].value_counts().sort_index().items():
        label = {0: 'Normal', 1: 'Osteopenia', 2: 'Osteoporosis'}.get(int(cls), str(cls))
        print(f"    {cls} ({label}): {cnt}")
    print()

    # Build manifest
    records = []
    for _, row in sampled.iterrows():
        pid = int(row['patient_id'])
        for view in ['AP', 'Lateral']:
            view_key = 'ap_preprocessed_path' if view == 'AP' else 'lateral_preprocessed_path'
            img_path = row.get(view_key, '')
            if not img_path or str(img_path) == 'nan' or not os.path.exists(str(img_path)):
                # Construct path
                img_path = str(PREPROC_DIR / f"patient_{pid:03d}" / f"{view}.png")

            mask_path = str(SEG_DIR / f"patient_{pid:03d}_{view}_mask.png")
            viz_path  = str(VIZ_DIR  / f"patient_{pid:03d}_{view}_viz.jpg")
            records.append({
                'patient_id': pid,
                'view': view,
                'image_path': img_path,
                'mask_path': mask_path,
                'viz_path': viz_path,
                'Osteoporosis': int(row['Osteoporosis']),
                'bmd': float(row['bmd']),
                'mask_status': 'PENDING',
                'verified': False,
            })

    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(SEG_MANIFEST, index=False)
    print(f"  Segmentation manifest saved: {SEG_MANIFEST}")
    print(f"  Total images in subset: {len(manifest_df)} ({len(sampled)} patients × 2 views)")
    print()

    # Run SAM if checkpoint exists
    if SAM_CHECKPOINT.exists():
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"  Loading SAM ({SAM_MODEL_TYPE}) on {device}...")
            sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
            sam.to(device=device)
            predictor = SamPredictor(sam)
            print(f"  SAM loaded successfully.")
            print()

            ok_count  = 0
            err_count = 0

            for idx, row in manifest_df.iterrows():
                pid    = int(row['patient_id'])
                view   = row['view']
                img_path = row['image_path']

                print(f"  [{idx+1:4d}/{len(manifest_df)}] Patient {pid:03d} {view} ... ", end='', flush=True)

                # Skip if already generated
                if os.path.exists(row['mask_path']) and os.path.exists(row['viz_path']):
                    manifest_df.at[idx, 'mask_status'] = 'PSEUDO_LABEL'
                    ok_count += 1
                    print("ALREADY EXISTS (skipping)")
                    continue

                if not os.path.exists(img_path):
                    print(f"MISSING ({img_path})")
                    manifest_df.at[idx, 'mask_status'] = 'MISSING_IMAGE'
                    err_count += 1
                    continue

                try:
                    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_rgb  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                    h, w     = img_gray.shape

                    mask = segment_vertebrae_sam(img_rgb, view, predictor)
                    save_mask_png(mask, row['mask_path'])

                    if view == 'AP':
                        prompts = get_vertebra_prompts_ap(h, w)
                    else:
                        prompts = get_vertebra_prompts_lateral(h, w)

                    create_visualization(img_gray, mask, prompts,
                                         row['viz_path'], pid, view)

                    manifest_df.at[idx, 'mask_status'] = 'PSEUDO_LABEL'
                    ok_count += 1
                    print("OK")

                    # Periodically save manifest every 25 images
                    if (idx + 1) % 25 == 0:
                        manifest_df.to_csv(SEG_MANIFEST, index=False)

                except Exception as e:
                    manifest_df.at[idx, 'mask_status'] = f'ERROR: {str(e)[:80]}'
                    err_count += 1
                    print(f"ERROR: {str(e)[:60]}")

            # Save updated manifest
            manifest_df.to_csv(SEG_MANIFEST, index=False)
            print()
            print(f"  Pseudo-labels generated: {ok_count}")
            print(f"  Errors: {err_count}")
            print()
            print("  WARNING IMPORTANT: These are pseudo-labels generated by SAM.")
            print("  They MUST be manually reviewed and corrected.")
            print(f"  Visualizations saved to: {VIZ_DIR}")
            print(f"  Use an annotation tool (e.g., LabelMe, ITKSNAP) to correct masks.")

        except ImportError:
            print("  segment_anything package not installed.")
            print("  Install with:")
            print("    pip install git+https://github.com/facebookresearch/segment-anything.git")
    else:
        print("  Skipping SAM execution (checkpoint not available).")
        print("  Download checkpoint and re-run to generate pseudo-labels.")

    print()
    print("=" * 60)
    print("PHASE 2 READY")
    print("=" * 60)
    print()
    print("After running SAM and verifying masks manually:")
    print("  --> Run phase2b_evaluate_segmentation.py  (Dice, IoU metrics)")
    print("  --> Run phase3_train_segmentation_model.py (U-Net training)")
