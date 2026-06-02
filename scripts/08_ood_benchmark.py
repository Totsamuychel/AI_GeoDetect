"""
08_ood_benchmark.py — Compare OOD detectors for the "is this one of our 4 cities?"
gate, on StreetCLIP embeddings.

Methods compared:
  - cosine  : max cosine sim to class-prototype (mean train emb)  [current gate]
  - maha    : Mahalanobis distance to nearest class (pooled shrinkage cov)
  - knn     : mean cosine sim to k nearest train embeddings (k=20)

In-distribution (positives) = our test split (manifests_sv/test.csv).
Out-of-distribution (negatives) = osv5m Romania (RO) street photos — definitely
none of Kyiv/Warsaw/Prague/Budapest.

Reports AUROC and TPR@FPR=5% (flag negatives while keeping 95% of real city
photos). Calibration stats come from TRAIN only.

Usage:
    python scripts/08_ood_benchmark.py --arch streetclip --per-class 400 --neg 800
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, "code")
from evaluate import load_checkpoint          # noqa: E402
from augmentations import get_norm_for, get_val_transforms  # noqa: E402
from utils import get_device                  # noqa: E402

ROOT = Path(".").resolve()
IMG_SIZE = {"streetclip": 336, "baseline": 260, "geoclip": 224}


def resolve(fp: str) -> Path:
    p = ROOT / fp
    if p.exists():
        return p
    return ROOT / "dataset" / fp


@torch.no_grad()
def embed_paths(model, arch, transform, device, paths, bs=32):
    embs, batch = [], []
    def flush():
        if not batch:
            return
        t = torch.stack(batch).to(device)
        if arch == "baseline":
            e = model.get_embeddings(t)
        else:
            e = model.encode_image(t)
        embs.append(e.float().cpu().numpy())
        batch.clear()
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        batch.append(transform(img))
        if len(batch) == bs:
            flush()
    flush()
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1))


def auroc(scores_in, scores_out):
    """AUROC with positive = in-distribution (higher score = more in-dist)."""
    y = np.r_[np.ones_like(scores_in), np.zeros_like(scores_out)]
    s = np.r_[scores_in, scores_out]
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    # trapezoid integral (np.trapz removed in numpy 2.x)
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))


def tpr_at_fpr(scores_in, scores_out, target_fpr=0.05):
    """Fraction of OOD flagged, while false-flagging <= target_fpr of in-dist.
    Threshold = target_fpr percentile of in-dist scores (lower = OOD)."""
    thr = np.percentile(scores_in, target_fpr * 100)
    tpr = float(np.mean(scores_out < thr))     # negatives correctly below thr
    fpr = float(np.mean(scores_in < thr))      # positives wrongly below thr
    return tpr, fpr, float(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="streetclip")
    ap.add_argument("--per-class", type=int, default=400, help="train emb / class")
    ap.add_argument("--pos", type=int, default=150, help="test positives / class")
    ap.add_argument("--neg", type=int, default=800, help="osv5m RO negatives")
    ap.add_argument("--knn-k", type=int, default=20)
    args = ap.parse_args()

    device = get_device()
    ckpt = f"checkpoints/{args.arch}_v2/best_model.pth"
    model, class_names, _cfg, _ = load_checkpoint(ckpt, device)
    model.eval()
    mean, std = get_norm_for(args.arch)
    tf = get_val_transforms(img_size=IMG_SIZE[args.arch], mean=mean, std=std)
    print(f"Model {args.arch}: classes={class_names}")

    # ── TRAIN embeddings (calibration) ────────────────────────────────────────
    tr = pd.read_csv("dataset/manifests_sv/train.csv")
    tr["city"] = tr["city"].str.lower()
    train_embs, train_lbl = [], []
    for ci, c in enumerate(class_names):
        sub = tr[tr.city == c].head(args.per_class)
        paths = [resolve(fp) for fp in sub.filepath]
        e = embed_paths(model, args.arch, tf, device, paths)
        train_embs.append(e); train_lbl.append(np.full(len(e), ci))
        print(f"  train {c}: {len(e)} embs (dim={e.shape[1]})")
    X = np.concatenate(train_embs); y = np.concatenate(train_lbl)
    D = X.shape[1]

    # class means + pooled shrinkage covariance
    means = np.stack([X[y == i].mean(0) for i in range(len(class_names))])  # (C,D)
    Z = X - means[y]
    cov = (Z.T @ Z) / len(X)
    alpha = 0.1
    cov = (1 - alpha) * cov + alpha * (np.trace(cov) / D) * np.eye(D)
    cov_inv = np.linalg.pinv(cov)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)        # for knn
    pa = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)  # protos

    def scores(E):
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        # cosine to prototype (higher = in-dist)
        cos = (En @ pa.T).max(1)
        # mahalanobis: -min_c distance (higher = in-dist)
        diff = E[:, None, :] - means[None, :, :]           # (N,C,D)
        md = np.einsum("ncd,de,nce->nc", diff, cov_inv, diff)
        maha = -md.min(1)
        # knn mean cosine to k nearest train
        sims = En @ Xn.T                                   # (N,Ntr)
        knn = np.sort(sims, axis=1)[:, -args.knn_k:].mean(1)
        return {"cosine": cos, "maha": maha, "knn": knn}

    # ── Positives (test) & Negatives (osv5m RO) ───────────────────────────────
    te = pd.read_csv("dataset/manifests_sv/test.csv"); te["city"] = te["city"].str.lower()
    pos_paths = []
    for c in class_names:
        pos_paths += [resolve(fp) for fp in te[te.city == c].head(args.pos).filepath]
    ro_dir = ROOT / "dataset/raw/osv5m/images/RO"
    neg_paths = sorted(ro_dir.glob("*.jpg"))[: args.neg]
    print(f"positives={len(pos_paths)} negatives(RO)={len(neg_paths)}")

    Epos = embed_paths(model, args.arch, tf, device, pos_paths)
    Eneg = embed_paths(model, args.arch, tf, device, neg_paths)
    sp, sn = scores(Epos), scores(Eneg)

    print(f"\n{'method':8s} {'AUROC':>7s} {'TPR@FPR5%':>10s} {'(FPR)':>7s}")
    print("-" * 36)
    best = None
    for m in ("cosine", "maha", "knn"):
        a = auroc(sp[m], sn[m])
        tpr, fpr, thr = tpr_at_fpr(sp[m], sn[m], 0.05)
        print(f"{m:8s} {a:7.3f} {tpr*100:9.1f}% {fpr*100:6.1f}%")
        if best is None or a > best[1]:
            best = (m, a, tpr, thr)
    print(f"\nBest by AUROC: {best[0]} (AUROC={best[1]:.3f}, TPR@5%={best[2]*100:.1f}%)")


if __name__ == "__main__":
    main()
