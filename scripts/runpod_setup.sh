#!/usr/bin/env bash
# =============================================================================
# runpod_setup.sh — environment bootstrap for training on a rented RTX 5090.
#
# Assumes a RunPod base image with PyTorch >= 2.7 + CUDA 12.8 (cu128) already
# installed (torch is NOT reinstalled here — see requirements.txt).
#
# Usage (from repo root):
#   bash scripts/runpod_setup.sh
# =============================================================================
set -euo pipefail

echo "==> nvidia-smi"
nvidia-smi || { echo "[ERROR] nvidia-smi failed — no GPU?"; exit 1; }

echo "==> Reproducibility env vars"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
echo "    PYTHONHASHSEED=$PYTHONHASHSEED  CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

echo "==> Torch / CUDA / GPU check"
python - <<'PY'
import sys
import torch

print("torch       :", torch.__version__)
print("torch.cuda  :", torch.version.cuda)
print("cuda avail  :", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("[ERROR] CUDA not available to PyTorch")

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print("device      :", name)
print("capability  :", f"sm_{cap[0]}{cap[1]}")

# RTX 5090 = Blackwell, sm_120. Needs torch built with CUDA >= 12.8.
if cap[0] >= 12 and (torch.version.cuda or "0") < "12.8":
    sys.exit(
        f"[ERROR] Blackwell GPU (sm_{cap[0]}{cap[1]}) but torch CUDA "
        f"{torch.version.cuda} < 12.8 — use a PyTorch>=2.7 / cu128 image."
    )
print("[OK] GPU/torch compatible")
PY

echo "==> pip install -r requirements.txt (non-torch deps)"
pip install --no-input -r requirements.txt

echo "==> Dataset validation"
python - <<'PY'
import pandas as pd, sys
from pathlib import Path
root = Path("dataset")
ok = True
for split in ["train", "val", "test"]:
    p = root / "manifests" / f"{split}.csv"
    if not p.exists():
        print(f"  MISSING: {p}"); ok = False; continue
    df = pd.read_csv(p)
    cities = sorted(df["city"].dropna().unique().tolist())
    print(f"  {split:5s}: {len(df):5d} rows | cities={cities}")
if not ok:
    sys.exit("[ERROR] Missing manifests — check dataset upload")
print("[OK] Manifests valid")

import random, os
random.seed(42)
all_paths = []
for split in ["train","val","test"]:
    df = pd.read_csv(root / "manifests" / f"{split}.csv")
    all_paths += df["filepath"].tolist()
sample = random.sample(all_paths, 50)
missing = [p for p in sample if not (root / p).exists()]
if missing:
    print(f"[ERROR] {len(missing)}/50 sample images missing:")
    for p in missing[:5]: print(f"  {p}")
    sys.exit(1)
print("[OK] 50/50 sample images found")
PY

echo "==> Pre-download HuggingFace models (avoids timeout during training)"
python - <<'PY'
from transformers import CLIPModel, CLIPProcessor
print("  Downloading geolocal/StreetCLIP (~1.7 GB)...")
CLIPModel.from_pretrained("geolocal/StreetCLIP")
CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
print("  [OK] StreetCLIP cached")
PY

echo "==> pytest"
# -p no:debugging: the repo's code/ package shadows stdlib `code`, which the
# pytest pdb plugin imports at configure time. Disabling it is harmless.
pytest -q -p no:debugging tests/ || { echo "[WARN] tests reported failures — review before training"; }

echo ""
echo "==> Done. Next:"
echo "    python code/train.py --config configs/baseline.yaml"
echo "    python code/train.py --config configs/streetclip.yaml"
echo "    python code/train.py --config configs/geoclip.yaml"
echo ""
echo "    Or run all 3 sequentially:"
echo "    bash scripts/runpod_train_all.sh"
