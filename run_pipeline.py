"""
LUMOS Pipeline Runner - Resume & Continue Training
===================================================
Runs phases sequentially:
  Phase 2  → SAM Pseudo-Labeling (resumes from checkpoint)
  Phase 4  → Vertebral ROI Dataset Generation
  Phase 5-9 → Anatomy-Guided Multiview Multitask BMD Model Training

Usage:
  python run_pipeline.py                 # Run all phases
  python run_pipeline.py --start phase4  # Start from Phase 4
  python run_pipeline.py --start phase5  # Start from Phase 5-9
  python run_pipeline.py --skip-phase2   # Skip Phase 2 (use existing masks)
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("d:/LUMOS")

LOG_FILE = BASE_DIR / "pipeline_run.log"


def log(msg: str, also_print: bool = True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_phase(script: str, args: list = None, description: str = "") -> bool:
    """Run a Python script as a subprocess and return True if successful."""
    cmd = [sys.executable, str(BASE_DIR / script)]
    if args:
        cmd.extend(args)

    log(f"{'='*60}")
    log(f"STARTING: {description}")
    log(f"Command: {' '.join(cmd)}")
    log(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            # Stream output directly to console + capture for log
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - start
        for line in result.stdout.splitlines():
            log(f"  {line}", also_print=True)

        if result.returncode == 0:
            log(f"✅ COMPLETED: {description} (elapsed: {elapsed/60:.1f} min)")
            return True
        else:
            log(f"❌ FAILED: {description} (exit code {result.returncode}, elapsed: {elapsed/60:.1f} min)")
            return False

    except Exception as e:
        elapsed = time.time() - start
        log(f"❌ EXCEPTION in {description}: {e} (elapsed: {elapsed/60:.1f} min)")
        return False


def check_phase2_status() -> tuple:
    """Returns (masks_done, total_needed)."""
    pseudo_dir = BASE_DIR / "processed" / "segmentation" / "pseudo_labels"
    masks = list(pseudo_dir.glob("*.png")) if pseudo_dir.exists() else []
    return len(masks), 1600  # 800 patients × 2 views


def check_phase4_status() -> bool:
    """Returns True if roi_manifest.csv exists."""
    return (BASE_DIR / "processed" / "roi_manifest.csv").exists()


def check_phase5_status() -> bool:
    """Returns True if final model exists."""
    return (BASE_DIR / "models" / "bmd_multitask" / "anatomy_multitask_final.pth").exists()


def main():
    parser = argparse.ArgumentParser(description="LUMOS Pipeline Runner")
    parser.add_argument("--start", choices=["phase2", "phase4", "phase5"],
                        default="phase2",
                        help="Which phase to start from (default: phase2)")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip Phase 2 (use existing masks as-is)")
    args = parser.parse_args()

    log("=" * 60)
    log("LUMOS PIPELINE RUNNER")
    log("=" * 60)

    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU (no GPU detected)"
    log(f"  Device: {device}")
    log(f"  Start phase: {args.start}")
    log(f"  Skip Phase 2: {args.skip_phase2}")
    log("")

    # ── PHASE 2: SAM Pseudo-Labeling ─────────────────────────────────────────
    run_phase2 = (args.start == "phase2") and (not args.skip_phase2)

    if run_phase2:
        masks_done, masks_total = check_phase2_status()
        log(f"Phase 2 status: {masks_done}/{masks_total} masks already generated")

        if masks_done >= masks_total:
            log("✅ Phase 2 already complete — all masks generated, skipping.")
        else:
            remaining = masks_total - masks_done
            log(f"  Resuming Phase 2: ~{remaining} images to process")
            log("  (Script will automatically skip already-completed patients)")
            ok = run_phase(
                "phase2_sam_pseudolabeling.py",
                args=["--patients", "all"],
                description="Phase 2 — SAM Pseudo-Labeling (resumed)"
            )
            if not ok:
                log("❌ Phase 2 failed. Continuing with existing masks for Phase 4...")
    else:
        log("⏭️  Skipping Phase 2.")

    # ── PHASE 4: ROI Dataset Generation ──────────────────────────────────────
    run_phase4 = args.start in ("phase2", "phase4")

    if run_phase4:
        if check_phase4_status():
            log("✅ Phase 4 already complete (roi_manifest.csv exists) — skipping.")
        else:
            masks_done, _ = check_phase2_status()
            log(f"  Running Phase 4 with {masks_done} available masks")
            ok = run_phase(
                "phase4_generate_roi_dataset.py",
                description="Phase 4 — Vertebral ROI Dataset Generation"
            )
            if not ok:
                log("❌ Phase 4 failed. Cannot continue to Phase 5-9.")
                sys.exit(1)
    else:
        log("⏭️  Skipping Phase 4.")

    # ── PHASE 5-9: BMD Model Training ────────────────────────────────────────
    if check_phase5_status():
        log("✅ Phase 5-9 already complete (final model exists).")
    else:
        if not check_phase4_status():
            log("❌ roi_manifest.csv not found — cannot start Phase 5-9. Run Phase 4 first.")
            sys.exit(1)

        ok = run_phase(
            "phase5_9_train_bmd_model.py",
            description="Phase 5-9 — Anatomy-Guided Multiview Multitask BMD Model Training"
        )
        if not ok:
            log("❌ Phase 5-9 training failed.")
            sys.exit(1)

    log("")
    log("=" * 60)
    log("🎉 LUMOS PIPELINE COMPLETE")
    log("=" * 60)
    log(f"  Results directory: {BASE_DIR / 'results'}")
    log(f"  Models directory:  {BASE_DIR / 'models'}")
    log(f"  Full log:          {LOG_FILE}")


if __name__ == "__main__":
    main()
