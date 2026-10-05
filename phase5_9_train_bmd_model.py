"""
PHASES 5-9 — Anatomy-Guided Multiview Multitask BMD Model
===========================================================
Project: Anatomy-Guided Multiview Multitask Deep Learning for Lumbar BMD
         Estimation and Osteoporosis Severity Assessment from X-ray Images

Architecture:
  ┌─────────────────────────────────────────────┐
  │  L1 ROI ┐                                   │
  │  L2 ROI ├──> Shared CNN Encoder ──> F1..F4  │
  │  L3 ROI │    (one encoder for all 4 verts)  │
  │  L4 ROI ┘                                   │
  │         └──> Inter-Vertebral Transformer     │
  │                    ↓                        │
  │            Vertebral Representation         │
  │                                             │
  │  AP:  vertebral rep ──┐                     │
  │                       ├──> View Fusion       │
  │  Lat: vertebral rep ──┘    (missing-view OK) │
  │                    ↓                        │
  │          Shared Spine Representation        │
  │               ┌──────┴──────┐              │
  │               ↓             ↓              │
  │      BMD Regression   Classification       │
  │   (7 outputs:         (3 classes:          │
  │    overall + 6 combs)  Normal/Osteopenia/  │
  │                         Osteoporosis)      │
  └─────────────────────────────────────────────┘

KEY DECISIONS:
  - ONE shared CNN processes all 4 vertebrae (weight sharing)
  - Transformer models INTER-VERTEBRAL relationships (not just better features)
  - Missing-view training: 60% both, 20% AP-only, 20% Lateral-only
  - Multitask: 7 BMD targets (overall + 6 vertebral-combination) + classification
  - Patient-level split (NEVER image-level)
  - Huber loss for regression, CrossEntropy for classification

RUNS ON: Google Colab (GPU required for reasonable training time)
"""

import os
import json
import random
import math
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(os.environ.get("LUMOS_DIR", Path(__file__).resolve().parent))
PROCESSED_DIR = BASE_DIR / "processed"
ROI_MANIFEST  = PROCESSED_DIR / "roi_manifest.csv"
MODELS_DIR    = BASE_DIR / "models" / "bmd_multitask"
RESULTS_DIR   = BASE_DIR / "results" / "bmd"
CLASS_RESULTS = BASE_DIR / "results" / "classification"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CLASS_RESULTS.mkdir(parents=True, exist_ok=True)

CHECKPOINT    = MODELS_DIR / "anatomy_multitask_best.pth"
FINAL_MODEL   = MODELS_DIR / "anatomy_multitask_final.pth"

# ── Hyperparameters ───────────────────────────────────────────────────────────
ROI_SIZE        = 128        # Input ROI size per vertebra
EMBED_DIM       = 256        # CNN output feature dimension
N_HEADS         = 8          # Transformer attention heads
N_TRANS_LAYERS  = 4          # Transformer encoder layers
FUSION_DIM      = 512        # View fusion output dim
N_CLASSES       = 3          # Normal / Osteopenia / Osteoporosis
N_BMD_TARGETS   = 7          # overall + 6 vertebral combinations
BATCH_SIZE      = 8
EPOCHS          = 100
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
HUBER_DELTA     = 0.1        # Huber loss delta for BMD regression

# Loss weights
LAMBDA_OVERALL_BMD = 1.0
LAMBDA_COMBO_BMD   = 0.5     # weight for each of the 6 combination targets
LAMBDA_CLASS       = 1.0

# Missing-view training probabilities
P_BOTH_VIEWS   = 0.60
P_AP_ONLY      = 0.20
P_LAT_ONLY     = 0.20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BMD target columns (in order)
BMD_COLS = ['bmd', 'bmd_L1-L2', 'bmd_L1-L3', 'bmd_L1-L4',
            'bmd_L2-L3', 'bmd_L2-L4', 'bmd_L3-L4']
BMD_NAMES = ['Overall', 'L1-L2', 'L1-L3', 'L1-L4', 'L2-L3', 'L2-L4', 'L3-L4']

print("=" * 60)
print("PHASES 5-9: Anatomy-Guided Multiview Multitask BMD Model")
print("=" * 60)
print(f"  Device: {DEVICE}")
print()


# ════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════
class LUMOSMultitaskDataset(Dataset):
    """
    Loads L1-L4 ROIs for AP and Lateral views.
    Returns:
      ap_rois:  (4, 3, ROI_SIZE, ROI_SIZE) or zero tensor if missing
      lat_rois: (4, 3, ROI_SIZE, ROI_SIZE) or zero tensor if missing
      ap_mask:  bool - True if AP view available
      lat_mask: bool - True if Lateral view available
      bmd_targets: (7,) float - [overall, L1-L2, ..., L3-L4]
      cls_target:  int - 0/1/2
    """
    def __init__(self, records: list, augment: bool = False,
                 missing_view_p: tuple = (0.6, 0.2, 0.2)):
        self.records  = records
        self.augment  = augment
        # (both, ap_only, lat_only)
        self.missing_p = missing_view_p
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def load_roi(self, path: str) -> np.ndarray:
        """Load grayscale ROI and convert to normalized 3-channel float."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.uint8)
        img = cv2.resize(img, (ROI_SIZE, ROI_SIZE))
        img_3ch = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        img_3ch = (img_3ch - self.mean) / self.std
        return img_3ch  # (ROI_SIZE, ROI_SIZE, 3)

    def augment_roi(self, img: np.ndarray) -> np.ndarray:
        """Apply mild augmentation to a single ROI."""
        # Random rotation ±10°
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
        # Random brightness/contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta  = random.uniform(-0.1, 0.1)  # brightness (in normalized space)
            img = np.clip(img * alpha + beta, -3.0, 3.0)
        return img

    def get_view_rois(self, rec: dict, view: str) -> torch.Tensor:
        """Returns (4, 3, ROI_SIZE, ROI_SIZE) tensor for a given view."""
        rois = []
        pid = int(rec.get('patient_id', 0))
        for level in ['L1', 'L2', 'L3', 'L4']:
            key = f'{level}_roi_path_{view}'
            path = rec.get(key, '')
            if not path or not os.path.exists(str(path)):
                # Fallback to current PROCESSED_DIR
                fallback_path = PROCESSED_DIR / "vertebral_rois" / f"patient_{pid:03d}" / view / f"{level}.png"
                if fallback_path.exists():
                    path = fallback_path
                else:
                    path = ''

            if not path or not os.path.exists(str(path)):
                roi = np.zeros((ROI_SIZE, ROI_SIZE, 3), dtype=np.float32)
            else:
                roi = self.load_roi(str(path))
                if self.augment:
                    roi = self.augment_roi(roi)
            rois.append(roi.transpose(2, 0, 1))  # (3, H, W)
        return torch.from_numpy(np.stack(rois, axis=0))  # (4, 3, H, W)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Determine which views to use (missing-view training)
        r = random.random()
        if r < self.missing_p[0]:
            use_ap, use_lat = True, True
        elif r < self.missing_p[0] + self.missing_p[1]:
            use_ap, use_lat = True, False
        else:
            use_ap, use_lat = False, True

        ap_rois  = self.get_view_rois(rec, 'AP')  if use_ap  else torch.zeros(4, 3, ROI_SIZE, ROI_SIZE)
        lat_rois = self.get_view_rois(rec, 'Lateral') if use_lat else torch.zeros(4, 3, ROI_SIZE, ROI_SIZE)

        ap_mask  = torch.tensor(use_ap,  dtype=torch.bool)
        lat_mask = torch.tensor(use_lat, dtype=torch.bool)

        bmd_targets = torch.tensor(
            [float(rec.get(c, 0.0) or 0.0) for c in BMD_COLS],
            dtype=torch.float32
        )
        cls_target = torch.tensor(int(rec.get('Osteoporosis', 0)), dtype=torch.long)

        return ap_rois, lat_rois, ap_mask, lat_mask, bmd_targets, cls_target


# ════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════
class SharedCNNEncoder(nn.Module):
    """
    Shared CNN encoder for vertebral ROIs.
    ONE set of weights processes all 4 vertebrae (L1-L4).
    Uses MobileNetV2 backbone (lightweight, suitable for small ROIs).
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # Use features only (drop classifier)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Project to embed_dim
        self.proj = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*4, 3, H, W)  — batch of B patients × 4 vertebrae
        returns: (B*4, embed_dim)
        """
        feat = self.features(x)          # (B*4, 1280, h, w)
        feat = self.pool(feat).flatten(1) # (B*4, 1280)
        return self.proj(feat)            # (B*4, embed_dim)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the 4 vertebral positions."""
    def __init__(self, embed_dim: int, n_pos: int = 4):
        super().__init__()
        pe = torch.zeros(n_pos, embed_dim)
        pos = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2).float() *
                        (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class InterVertebralTransformer(nn.Module):
    """
    Transformer encoder that models INTER-VERTEBRAL relationships.
    Purpose: learn how L1, L2, L3, L4 features relate to each other.
    This is NOT just 'using a Transformer for better performance' —
    it has a specific anatomical purpose.
    """
    def __init__(self, embed_dim: int = 256, n_heads: int = 8,
                 n_layers: int = 4, ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, embed_dim)  — 4 vertebral feature tokens
        returns: (B, 4, embed_dim)  — contextualized tokens
        """
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.norm(x)


class ViewFusionModule(nn.Module):
    """
    Fuses AP and Lateral vertebral representations.
    Handles missing views via a learned zero-replacement strategy:
    If a view is missing, its token is replaced by a learnable null embedding.
    """
    def __init__(self, embed_dim: int = 256, fusion_dim: int = 512):
        super().__init__()
        self.null_embed = nn.Parameter(torch.randn(1, embed_dim))
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self,
                ap_repr: torch.Tensor,   # (B, embed_dim) — mean-pooled AP
                lat_repr: torch.Tensor,  # (B, embed_dim) — mean-pooled Lateral
                ap_mask: torch.Tensor,   # (B,) bool
                lat_mask: torch.Tensor,  # (B,) bool
                ) -> torch.Tensor:
        """Returns (B, fusion_dim) fused representation."""
        B = ap_repr.shape[0]

        # Replace missing views with learned null embedding
        null = self.null_embed.expand(B, -1)
        ap_in  = torch.where(ap_mask.unsqueeze(-1),  ap_repr,  null)
        lat_in = torch.where(lat_mask.unsqueeze(-1), lat_repr, null)

        fused = torch.cat([ap_in, lat_in], dim=-1)  # (B, embed_dim*2)
        return self.fusion(fused)                     # (B, fusion_dim)


class AnatomyGuidedMultitaskModel(nn.Module):
    """
    Full anatomy-guided multiview multitask model.

    FORWARD PASS:
      ap_rois:  (B, 4, 3, H, W)
      lat_rois: (B, 4, 3, H, W)
      ap_mask:  (B,) bool
      lat_mask: (B,) bool

    OUTPUTS:
      bmd_preds:   (B, 7)  — [overall_BMD, L1-L2, L1-L3, L1-L4, L2-L3, L2-L4, L3-L4]
      class_logits:(B, 3)  — [Normal, Osteopenia, Osteoporosis]
    """
    def __init__(self, embed_dim: int = 256, n_heads: int = 8,
                 n_trans_layers: int = 4, fusion_dim: int = 512,
                 n_classes: int = 3, n_bmd: int = 7):
        super().__init__()

        # Phase 5: Shared CNN encoder
        self.shared_cnn = SharedCNNEncoder(embed_dim)

        # Phase 6: Inter-vertebral Transformer (per view)
        self.transformer = InterVertebralTransformer(embed_dim, n_heads, n_trans_layers)

        # Phase 7: Two-view fusion with missing-view handling
        self.view_fusion = ViewFusionModule(embed_dim, fusion_dim)

        # Phase 8: Multitask BMD regression heads
        # Overall BMD head
        self.bmd_head_overall = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        # Vertebral-combination BMD heads (one per combination)
        # Each uses the relevant vertebral features pooled
        self.bmd_heads_combo = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, 1),
            ) for _ in range(6)  # 6 combinations
        ])

        # Phase 9: Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def encode_view(self, rois: torch.Tensor) -> torch.Tensor:
        """
        Encodes 4 vertebral ROIs for one view.
        rois: (B, 4, 3, H, W)
        returns: (B, embed_dim) — mean-pooled transformer output
        """
        B, N_VERT, C, H, W = rois.shape
        # Flatten batch and vertebra dims for shared CNN
        rois_flat = rois.reshape(B * N_VERT, C, H, W)          # (B*4, 3, H, W)
        feats_flat = self.shared_cnn(rois_flat)                  # (B*4, embed_dim)
        feats = feats_flat.reshape(B, N_VERT, -1)               # (B, 4, embed_dim)
        # Inter-vertebral Transformer
        feats_ctx = self.transformer(feats)                      # (B, 4, embed_dim)
        # Mean-pool over vertebrae to get view-level representation
        return feats_ctx.mean(dim=1)                             # (B, embed_dim)

    def forward(self, ap_rois, lat_rois, ap_mask, lat_mask):
        # Encode each view
        ap_repr  = self.encode_view(ap_rois)   # (B, embed_dim)
        lat_repr = self.encode_view(lat_rois)  # (B, embed_dim)

        # Fuse views (with missing-view handling)
        fused = self.view_fusion(ap_repr, lat_repr, ap_mask, lat_mask)  # (B, fusion_dim)

        # BMD regression
        bmd_overall = self.bmd_head_overall(fused)  # (B, 1)
        bmd_combos  = torch.cat(
            [head(fused) for head in self.bmd_heads_combo], dim=-1
        )  # (B, 6)
        bmd_preds = torch.cat([bmd_overall, bmd_combos], dim=-1)  # (B, 7)

        # Classification
        class_logits = self.cls_head(fused)  # (B, 3)

        return bmd_preds, class_logits


# ════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ════════════════════════════════════════════════════════════
class MultitaskLoss(nn.Module):
    def __init__(self, huber_delta: float = 0.1,
                 lambda_overall: float = 1.0,
                 lambda_combo: float = 0.5,
                 lambda_cls: float = 1.0):
        super().__init__()
        self.huber        = nn.HuberLoss(delta=huber_delta, reduction='mean')
        self.ce           = nn.CrossEntropyLoss()
        self.lambda_overall = lambda_overall
        self.lambda_combo   = lambda_combo
        self.lambda_cls     = lambda_cls

    def forward(self, bmd_preds, class_logits, bmd_targets, cls_targets):
        # Primary BMD loss (overall)
        loss_overall = self.huber(bmd_preds[:, 0], bmd_targets[:, 0])

        # Auxiliary BMD losses (6 combinations)
        loss_combo = sum(
            self.huber(bmd_preds[:, i+1], bmd_targets[:, i+1])
            for i in range(6)
        ) / 6.0

        # Classification loss
        loss_cls = self.ce(class_logits, cls_targets)

        total = (self.lambda_overall * loss_overall +
                 self.lambda_combo   * loss_combo   +
                 self.lambda_cls     * loss_cls)

        return total, {
            'loss_overall_bmd': loss_overall.item(),
            'loss_combo_bmd':   loss_combo.item(),
            'loss_cls':         loss_cls.item(),
            'loss_total':       total.item(),
        }


# ════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ════════════════════════════════════════════════════════════
def compute_bmd_metrics(preds: np.ndarray, targets: np.ndarray, name: str = '') -> dict:
    mae  = float(np.abs(preds - targets).mean())
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-8))
    try:
        pearson = float(pearsonr(preds, targets)[0])
    except Exception:
        pearson = float('nan')
    try:
        spearman = float(spearmanr(preds, targets)[0])
    except Exception:
        spearman = float('nan')
    return {'name': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
            'Pearson': pearson, 'Spearman': spearman}


def compute_classification_metrics(preds: np.ndarray, targets: np.ndarray,
                                    probs: np.ndarray = None) -> dict:
    acc   = accuracy_score(targets, preds)
    prec  = precision_score(targets, preds, average='macro', zero_division=0)
    rec   = recall_score(targets, preds, average='macro', zero_division=0)
    f1    = f1_score(targets, preds, average='macro', zero_division=0)
    cm    = confusion_matrix(targets, preds).tolist()
    report = classification_report(
        targets, preds,
        target_names=['Normal', 'Osteopenia', 'Osteoporosis'],
        output_dict=True, zero_division=0
    )
    auc = None
    if probs is not None:
        try:
            auc = float(roc_auc_score(targets, probs, multi_class='ovr',
                                       average='macro'))
        except Exception:
            pass
    return {
        'accuracy': float(acc), 'precision_macro': float(prec),
        'recall_macro': float(rec), 'f1_macro': float(f1),
        'auc_macro': auc, 'confusion_matrix': cm,
        'per_class_report': report,
    }


# ════════════════════════════════════════════════════════════
# TRAINING + EVALUATION
# ════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    totals = {}
    n = 0
    for ap, lat, ap_m, lat_m, bmd_t, cls_t in loader:
        ap, lat      = ap.to(device),  lat.to(device)
        ap_m, lat_m  = ap_m.to(device), lat_m.to(device)
        bmd_t, cls_t = bmd_t.to(device), cls_t.to(device)

        optimizer.zero_grad()
        bmd_p, cls_p = model(ap, lat, ap_m, lat_m)
        loss, breakdown = criterion(bmd_p, cls_p, bmd_t, cls_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = ap.size(0)
        n += bs
        for k, v in breakdown.items():
            totals[k] = totals.get(k, 0) + v * bs

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, return_preds=False):
    model.eval()
    totals = {}
    n = 0
    all_bmd_p, all_bmd_t = [], []
    all_cls_p, all_cls_t = [], []
    all_cls_probs = []

    for ap, lat, ap_m, lat_m, bmd_t, cls_t in loader:
        ap, lat      = ap.to(device),  lat.to(device)
        ap_m, lat_m  = ap_m.to(device), lat_m.to(device)
        bmd_t_gpu    = bmd_t.to(device)
        cls_t_gpu    = cls_t.to(device)

        bmd_p, cls_p = model(ap, lat, ap_m, lat_m)
        loss, breakdown = criterion(bmd_p, cls_p, bmd_t_gpu, cls_t_gpu)

        bs = ap.size(0)
        n += bs
        for k, v in breakdown.items():
            totals[k] = totals.get(k, 0) + v * bs

        all_bmd_p.append(bmd_p.cpu().numpy())
        all_bmd_t.append(bmd_t.numpy())
        all_cls_p.append(cls_p.argmax(dim=-1).cpu().numpy())
        all_cls_probs.append(torch.softmax(cls_p, dim=-1).cpu().numpy())
        all_cls_t.append(cls_t.numpy())

    avg = {k: v / n for k, v in totals.items()}

    bmd_preds   = np.concatenate(all_bmd_p, axis=0)
    bmd_targets = np.concatenate(all_bmd_t, axis=0)
    cls_preds   = np.concatenate(all_cls_p, axis=0)
    cls_probs   = np.concatenate(all_cls_probs, axis=0)
    cls_targets = np.concatenate(all_cls_t, axis=0)

    # Overall BMD metric
    bmd_m = compute_bmd_metrics(bmd_preds[:, 0], bmd_targets[:, 0], 'Overall')

    if return_preds:
        return avg, bmd_m, bmd_preds, bmd_targets, cls_preds, cls_probs, cls_targets
    return avg, bmd_m


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── Load ROI manifest ──────────────────────────────────
    if not ROI_MANIFEST.exists():
        print("ERROR: ROI manifest not found. Run phase4_generate_roi_dataset.py first.")
        exit(1)

    roi_df = pd.read_csv(ROI_MANIFEST)
    print(f"  ROI manifest loaded: {len(roi_df)} records")

    # Reorganize manifest: one row per patient (combine AP and Lateral)
    # Pivot so each patient has both view paths in one record
    patient_records = {}
    for _, row in roi_df.iterrows():
        pid  = int(row['patient_id'])
        view = row['view']
        if pid not in patient_records:
            patient_records[pid] = {
                'patient_id': pid,
                'split': str(row['split']),
                'Osteoporosis': int(row['Osteoporosis']),
            }
            for c in BMD_COLS:
                patient_records[pid][c] = float(row.get(c, 0.0) or 0.0)

        for level in ['L1', 'L2', 'L3', 'L4']:
            key_in   = f'{level}_roi_path'
            key_out  = f'{level}_roi_path_{view}'
            patient_records[pid][key_out] = str(row.get(key_in, ''))

    records_list = list(patient_records.values())
    print(f"  Patient records assembled: {len(records_list)}")

    train_recs = [r for r in records_list if r['split'] == 'train']
    val_recs   = [r for r in records_list if r['split'] == 'validation']
    test_recs  = [r for r in records_list if r['split'] == 'test']
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}, Test: {len(test_recs)}")
    print()

    # ── Datasets + Loaders ────────────────────────────────
    train_ds = LUMOSMultitaskDataset(train_recs, augment=True,
                                      missing_view_p=(P_BOTH_VIEWS, P_AP_ONLY, P_LAT_ONLY))
    val_ds   = LUMOSMultitaskDataset(val_recs,   augment=False, missing_view_p=(1,0,0))
    test_ds  = LUMOSMultitaskDataset(test_recs,  augment=False, missing_view_p=(1,0,0))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────
    model = AnatomyGuidedMultitaskModel(
        embed_dim=EMBED_DIM, n_heads=N_HEADS, n_trans_layers=N_TRANS_LAYERS,
        fusion_dim=FUSION_DIM, n_classes=N_CLASSES, n_bmd=N_BMD_TARGETS,
    ).to(DEVICE)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = MultitaskLoss(
        huber_delta=HUBER_DELTA,
        lambda_overall=LAMBDA_OVERALL_BMD,
        lambda_combo=LAMBDA_COMBO_BMD,
        lambda_cls=LAMBDA_CLASS,
    )

    # ── Training loop ──────────────────────────────────────
    best_val_r2 = -float('inf')
    history = {'train': [], 'val': [], 'val_bmd_mae': [], 'val_bmd_r2': []}

    print("  Training...")
    print(f"  {'Ep':<5} {'TrLoss':<10} {'VlLoss':<10} "
          f"{'ValMAE':<10} {'ValR2':<10} {'Note'}")
    print("  " + "-" * 55)

    for epoch in range(1, EPOCHS + 1):
        train_losses = train_epoch(model, train_dl, optimizer, criterion, DEVICE)
        val_losses, val_bmd = eval_epoch(model, val_dl, criterion, DEVICE)
        scheduler.step()

        note = ''
        if val_bmd['R2'] > best_val_r2:
            best_val_r2 = val_bmd['R2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': best_val_r2,
            }, CHECKPOINT)
            note = '<- best'

        history['train'].append(train_losses['loss_total'])
        history['val'].append(val_losses['loss_total'])
        history['val_bmd_mae'].append(val_bmd['MAE'])
        history['val_bmd_r2'].append(val_bmd['R2'])

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:<5} {train_losses['loss_total']:<10.4f} "
                  f"{val_losses['loss_total']:<10.4f} "
                  f"{val_bmd['MAE']:<10.4f} {val_bmd['R2']:<10.4f} {note}")

    torch.save(model.state_dict(), FINAL_MODEL)
    print()
    print(f"  Best Val R2: {best_val_r2:.4f}")
    print(f"  Checkpoint:  {CHECKPOINT}")

    # ── Final Evaluation on Test Set ───────────────────────
    print()
    print("=" * 60)
    print("  FINAL EVALUATION — Test Set")
    print("=" * 60)

    # Load best checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    (test_losses, _, bmd_preds, bmd_targets,
     cls_preds, cls_probs, cls_targets) = eval_epoch(
        model, test_dl, criterion, DEVICE, return_preds=True
    )

    print()
    print("  BMD Regression Results:")
    print(f"  {'Target':<12} {'MAE':<8} {'RMSE':<8} {'R2':<8} {'Pearson':<10} {'Spearman'}")
    print("  " + "-" * 55)
    bmd_results = []
    for i, name in enumerate(BMD_NAMES):
        m = compute_bmd_metrics(bmd_preds[:, i], bmd_targets[:, i], name)
        bmd_results.append(m)
        print(f"  {name:<12} {m['MAE']:<8.4f} {m['RMSE']:<8.4f} {m['R2']:<8.4f} "
              f"{m['Pearson']:<10.4f} {m['Spearman']:.4f}")

    print()
    print("  Classification Results:")
    cls_m = compute_classification_metrics(cls_preds, cls_targets, cls_probs)
    print(f"    Accuracy:       {cls_m['accuracy']:.4f}")
    print(f"    Macro-F1:       {cls_m['f1_macro']:.4f}")
    print(f"    Macro-Precision:{cls_m['precision_macro']:.4f}")
    print(f"    Macro-Recall:   {cls_m['recall_macro']:.4f}")
    if cls_m['auc_macro']:
        print(f"    ROC-AUC (OvR):  {cls_m['auc_macro']:.4f}")
    print(f"    Confusion Matrix:")
    for row in cls_m['confusion_matrix']:
        print(f"      {row}")

    # Save results
    all_results = {
        'config': {
            'embed_dim': EMBED_DIM, 'n_heads': N_HEADS,
            'n_trans_layers': N_TRANS_LAYERS, 'fusion_dim': FUSION_DIM,
            'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'lr': LR,
            'huber_delta': HUBER_DELTA,
            'lambda_overall': LAMBDA_OVERALL_BMD,
            'lambda_combo': LAMBDA_COMBO_BMD,
            'lambda_cls': LAMBDA_CLASS,
            'missing_view_p': [P_BOTH_VIEWS, P_AP_ONLY, P_LAT_ONLY],
            'random_seed': RANDOM_SEED,
        },
        'n_test_patients': len(test_recs),
        'bmd_results': bmd_results,
        'classification_results': cls_m,
        'training_history': history,
    }

    with open(RESULTS_DIR / 'bmd_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(CLASS_RESULTS / 'classification_test_results.json', 'w') as f:
        json.dump(cls_m, f, indent=2, default=str)

    print()
    print(f"  Results saved to:")
    print(f"    {RESULTS_DIR / 'bmd_test_results.json'}")
    print(f"    {CLASS_RESULTS / 'classification_test_results.json'}")
    print()
    print("=" * 60)
    print("PHASES 5-9 COMPLETE")
    print("=" * 60)
    print("Next: Run phase10_gradcam_explainability.py")
