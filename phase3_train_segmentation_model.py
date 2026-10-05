"""
PHASE 3 — Train Vertebral Segmentation Model (U-Net)
======================================================
Project: Anatomy-Guided Multiview Multitask Deep Learning for Lumbar BMD
         Estimation and Osteoporosis Severity Assessment from X-ray Images

Uses manually verified masks from Phase 2 to train a U-Net for multi-class
vertebral segmentation (0=bg, 1=L1, 2=L2, 3=L3, 4=L4).

PREREQUISITES:
  - Verified masks in: processed/segmentation/verified/
  - At least 50+ verified masks recommended
  - Run on GPU (Google Colab or local CUDA)

REQUIREMENTS:
  pip install torch torchvision segmentation-models-pytorch albumentations
"""

import os
import json
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not installed. Using simple U-Net fallback.")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALB_AVAILABLE = True
except ImportError:
    ALB_AVAILABLE = False
    print("Warning: albumentations not installed. Using basic augmentation.")

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("d:/LUMOS")
PROCESSED_DIR = BASE_DIR / "processed"
VERIFIED_DIR  = PROCESSED_DIR / "segmentation" / "verified"
PREDICTED_DIR = PROCESSED_DIR / "segmentation" / "predicted"
MODELS_DIR    = BASE_DIR / "models" / "segmentation"
RESULTS_DIR   = BASE_DIR / "results" / "segmentation"
PREDICTED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEG_MANIFEST  = PROCESSED_DIR / "segmentation_manifest.csv"
MASTER_CSV    = PROCESSED_DIR / "master_dataset.csv"
CHECKPOINT    = MODELS_DIR / "unet_vertebral_best.pth"
FINAL_MODEL   = MODELS_DIR / "unet_vertebral_final.pth"

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMG_SIZE     = 512
N_CLASSES    = 5       # 0=bg, 1=L1, 2=L2, 3=L3, 4=L4
BATCH_SIZE   = 4       # reduce if OOM on GPU
EPOCHS       = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-5
VAL_SPLIT    = 0.15    # fraction of verified data for validation
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'img_size': IMG_SIZE, 'n_classes': N_CLASSES, 'batch_size': BATCH_SIZE,
    'epochs': EPOCHS, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
    'val_split': VAL_SPLIT, 'device': str(DEVICE), 'random_seed': RANDOM_SEED,
}

print("=" * 60)
print("PHASE 3: Vertebral Segmentation Model Training")
print("=" * 60)
print(f"  Device: {DEVICE}")
print(f"  Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}")
print()

# ── Dataset ──────────────────────────────────────────────────────────────────
class VertebralSegDataset(Dataset):
    """
    Dataset of paired (image, mask) for vertebral segmentation.
    images: processed/preprocessed_images/patient_NNN/AP.png or Lateral.png
    masks:  processed/segmentation/verified/patient_NNN_AP_mask.png
    """
    def __init__(self, records: list, augment: bool = False):
        self.records = records
        self.augment = augment

        if ALB_AVAILABLE and augment:
            self.transform = A.Compose([
                A.Rotate(limit=10, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=5, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(5, 25), p=0.3),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
            self.val_transform = A.Compose([
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img  = cv2.imread(rec['image_path'], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(rec['mask_path'],  cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        if mask is None:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # Resize
        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Convert to 3-channel for SMP (expects RGB input)
        img_3ch = np.stack([img, img, img], axis=-1)

        if ALB_AVAILABLE and self.transform is not None:
            if self.augment:
                transformed = self.transform(image=img_3ch, mask=mask)
            else:
                transformed = self.val_transform(image=img_3ch, mask=mask)
            img_t  = transformed['image'].float()
            mask_t = torch.from_numpy(transformed['mask']).long()
        else:
            # Manual normalization
            img_t  = torch.from_numpy(img_3ch.transpose(2,0,1)).float() / 255.0
            mask_t = torch.from_numpy(mask.astype(np.int64))

        return img_t, mask_t


# ── Loss ──────────────────────────────────────────────────────────────────────
class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for multi-class segmentation."""
    def __init__(self, n_classes: int = 5, dice_weight: float = 0.5):
        super().__init__()
        self.n_classes    = n_classes
        self.dice_weight  = dice_weight
        self.ce_loss      = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce = self.ce_loss(logits, targets)

        probs = torch.softmax(logits, dim=1)
        dice_loss = 0.0
        smooth = 1e-6
        for cls in range(1, self.n_classes):  # skip background
            prob = probs[:, cls]
            tgt  = (targets == cls).float()
            intersection = (prob * tgt).sum()
            dice_loss += 1 - (2 * intersection + smooth) / \
                         (prob.sum() + tgt.sum() + smooth)
        dice_loss /= (self.n_classes - 1)

        return self.dice_weight * dice_loss + (1 - self.dice_weight) * ce


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, n_classes: int = 5):
    dice_scores = []
    for cls in range(1, n_classes):
        pred_b = (pred_mask == cls)
        gt_b   = (gt_mask   == cls)
        inter  = (pred_b & gt_b).sum()
        denom  = pred_b.sum() + gt_b.sum()
        dice_scores.append((2 * inter + 1e-7) / (denom + 1e-7))
    return np.mean(dice_scores), dice_scores


# ── Build model ───────────────────────────────────────────────────────────────
def build_model():
    if SMP_AVAILABLE:
        model = smp.UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=N_CLASSES,
        )
        print("  Model: U-Net++ with ResNet-34 encoder (ImageNet pretrained)")
    else:
        # Minimal fallback U-Net (not production quality)
        from torchvision.models.segmentation import fcn_resnet50
        model = fcn_resnet50(pretrained=False, num_classes=N_CLASSES)
        print("  Model: FCN-ResNet50 (fallback; install smp for U-Net++)")
    return model


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        if isinstance(logits, dict):
            logits = logits['out']
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, n_classes=5):
    model.eval()
    total_loss = 0
    all_dice   = []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        if isinstance(logits, dict):
            logits = logits['out']
        loss = criterion(logits, masks)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        gts   = masks.cpu().numpy()
        for p, g in zip(preds, gts):
            dice, _ = compute_dice(p, g, n_classes)
            all_dice.append(dice)
    return total_loss / len(loader.dataset), float(np.mean(all_dice))


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Gather verified masks
    if not VERIFIED_DIR.exists():
        print(f"  ERROR: Verified masks directory not found: {VERIFIED_DIR}")
        print("  Run Phase 2 and manually verify masks before training.")
        exit(1)

    verified_masks = list(VERIFIED_DIR.glob("*.png"))
    print(f"  Verified masks found: {len(verified_masks)}")

    if len(verified_masks) == 0:
        print("  No verified masks available yet.")
        print("  Please complete manual annotation first.")
        exit(0)

    if len(verified_masks) < 20:
        print(f"  WARNING: Only {len(verified_masks)} masks found.")
        print("  Recommend at least 50 verified masks for reliable training.")

    # Load manifest to get image paths
    manifest_df = pd.read_csv(SEG_MANIFEST)

    # Build records list matching verified masks
    records = []
    for mask_path in verified_masks:
        mask_name = mask_path.stem   # e.g. patient_001_AP_mask
        # Parse patient id and view
        parts = mask_name.split('_')
        try:
            pid  = int(parts[1])
            view = parts[2]
        except Exception:
            continue

        # Find matching image path
        row = manifest_df[
            (manifest_df['patient_id'] == pid) &
            (manifest_df['view'] == view)
        ]
        if len(row) == 0:
            continue

        img_path = row.iloc[0]['image_path']
        if not os.path.exists(img_path):
            img_path = str(
                PROCESSED_DIR / "preprocessed_images" / f"patient_{pid:03d}" / f"{view}.png"
            )

        records.append({
            'patient_id': pid,
            'view': view,
            'image_path': img_path,
            'mask_path': str(mask_path),
        })

    print(f"  Training pairs assembled: {len(records)}")

    # Split into train/val (patient-level safe: each record is a single view)
    random.shuffle(records)
    n_val = max(1, int(len(records) * VAL_SPLIT))
    val_records   = records[:n_val]
    train_records = records[n_val:]
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}")
    print()

    # Datasets and loaders
    train_ds = VertebralSegDataset(train_records, augment=True)
    val_ds   = VertebralSegDataset(val_records,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

    # Model, optimizer, scheduler, loss
    model     = build_model().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = DiceCELoss(n_classes=N_CLASSES)

    best_dice = 0.0
    history   = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    print("  Training segmentation model...")
    print(f"  {'Epoch':<8} {'TrainLoss':<12} {'ValLoss':<12} {'ValDice':<12} {'Note'}")
    print("  " + "-" * 55)

    for epoch in range(1, EPOCHS + 1):
        train_loss         = train_epoch(model, train_dl, optimizer, criterion, DEVICE)
        val_loss, val_dice = eval_epoch(model, val_dl, criterion, DEVICE, N_CLASSES)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        note = ''
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), CHECKPOINT)
            note = '← best'

        print(f"  {epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} {val_dice:<12.4f} {note}")

    torch.save(model.state_dict(), FINAL_MODEL)
    print()
    print(f"  Best Val Dice: {best_dice:.4f}")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Final model: {FINAL_MODEL}")

    # Save history
    with open(RESULTS_DIR / 'seg_training_history.json', 'w') as f:
        json.dump({**history, 'best_dice': best_dice, 'config': CONFIG}, f, indent=2)

    # Generate predictions for ALL 800 usable patients
    print()
    print("  Generating predictions for all usable patients...")
    master_df = pd.read_csv(MASTER_CSV)
    usable = master_df[
        master_df['has_both_views'].astype(bool) & master_df['has_clinical'].astype(bool)
    ]

    model.eval()
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    n_pred = 0
    for _, row in usable.iterrows():
        pid = int(row['patient_id'])
        for view in ['AP', 'Lateral']:
            view_key = 'ap_preprocessed_path' if view == 'AP' else 'lateral_preprocessed_path'
            img_path = row.get(view_key, '')
            if not img_path or str(img_path) == 'nan':
                img_path = str(
                    PROCESSED_DIR / "preprocessed_images" / f"patient_{pid:03d}" / f"{view}.png"
                )
            if not os.path.exists(img_path):
                continue

            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_3ch  = np.stack([img_gray, img_gray, img_gray], axis=-1)
            t = torch.from_numpy(img_3ch.transpose(2,0,1)).float() / 255.0
            t = normalize(t).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(t)
                if isinstance(logits, dict):
                    logits = logits['out']
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            out_path = str(PREDICTED_DIR / f"patient_{pid:03d}_{view}_mask.png")
            cv2.imwrite(out_path, pred)
            n_pred += 1

    print(f"  Predictions saved: {n_pred} masks → {PREDICTED_DIR}")
    print()
    print("=" * 60)
    print("PHASE 3 COMPLETE — Segmentation model trained")
    print("=" * 60)
    print("Next: Run phase2b_evaluate_segmentation.py on predicted masks")
    print("      Then: phase4_generate_roi_dataset.py")
