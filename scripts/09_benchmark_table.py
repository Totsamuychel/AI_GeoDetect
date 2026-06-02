"""
09_benchmark_table.py — Per-model confusion matrix + Acc/F1/TPR/FPR on the
v2 test split, for the webapp benchmark section (#10).

Saves results/benchmark_v2.json:
  { class_names: [...], test_size: N, models: [
      { id, label, top1, macro_f1, balanced_acc,
        confusion: [[...]],            # rows=true, cols=pred
        per_class: [{city, precision, recall, f1, tpr, fpr, support}] } ] }

Usage:
    python scripts/09_benchmark_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

sys.path.insert(0, "code")
from evaluate import load_checkpoint            # noqa: E402
from dataset import GeoDataset                  # noqa: E402
from augmentations import get_norm_for, get_val_transforms  # noqa: E402
from utils import get_device                    # noqa: E402

MODELS = [
    ("streetclip", "StreetCLIP", 336),
    ("geoclip",    "GeoCLIP",    224),
    ("baseline",   "Baseline CNN", 260),
]
COUNTRIES = ["UA", "PL", "CZ", "HU"]
TEST = "dataset/manifests_sv/test.csv"


@torch.no_grad()
def infer(arch, img_size, device):
    model, class_names, cfg, _ = load_checkpoint(f"checkpoints/{arch}_v2/best_model.pth", device)
    class_names = [c.lower() for c in class_names]
    mean, std = get_norm_for(arch)
    ds = GeoDataset(manifest_path=TEST, transform=get_val_transforms(img_size, mean=mean, std=std),
                    countries=COUNTRIES, quality_threshold=0.0, image_root=".")
    ds.df["city"] = ds.df["city"].astype(str).str.lower()
    ds.class_names = class_names
    ds._city_to_idx = {c: i for i, c in enumerate(class_names)}
    ds.num_classes = len(class_names)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8,
                        pin_memory=torch.cuda.is_available())
    preds, labels, confs, embs = [], [], [], []
    for images, lbl, _ in loader:
        images = images.to(device, non_blocking=True)
        if arch == "baseline":
            emb = model.get_embeddings(images); logits = model(images)
        elif arch == "streetclip":
            emb = model.encode_image(images); logits = model.head(emb)
        else:  # geoclip
            emb = model.encode_image(images); logits = model.classifier(emb)
        prob = F.softmax(logits, dim=1)
        preds.append(logits.argmax(1).cpu().numpy())
        confs.append(prob.max(1).values.float().cpu().numpy())
        labels.append(lbl.numpy())
        embs.append(emb.float().cpu().numpy())
    return (class_names, np.concatenate(labels), np.concatenate(preds),
            np.concatenate(confs), np.concatenate(embs))


def metrics_from_cm(cm):
    C = cm.shape[0]
    N = cm.sum()
    per_class, f1s, recalls = [], [], []
    for i in range(C):
        tp = cm[i, i]
        support = cm[i].sum()
        fp = cm[:, i].sum() - tp
        fn = support - tp
        tn = N - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support if support else 0.0          # = TPR
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1,
                          "tpr": recall, "fpr": fpr, "support": int(support)})
        f1s.append(f1); recalls.append(recall)
    top1 = float(np.trace(cm) / N)
    return top1, float(np.mean(f1s)), float(np.mean(recalls)), per_class


def reliability(confs, correct, nbins=10):
    """Калібрування: біни за впевненістю → (avg_conf, accuracy, count) + ECE."""
    bins = []
    ece = 0.0
    N = len(confs)
    for b in range(nbins):
        lo, hi = b / nbins, (b + 1) / nbins
        m = (confs >= lo) & (confs <= hi) if b == nbins - 1 else (confs >= lo) & (confs < hi)
        c = int(m.sum())
        if c == 0:
            bins.append({"conf": (lo + hi) / 2, "acc": None, "count": 0})
            continue
        acc = float(correct[m].mean())
        cf = float(confs[m].mean())
        bins.append({"conf": round(cf, 4), "acc": round(acc, 4), "count": c})
        ece += c / N * abs(acc - cf)
    return {"bins": bins, "ece": round(float(ece), 4)}


def fit_tsne(embs, labels, class_names, per_class=300):
    """Збалансований subsample → TSNE(2D), нормалізований у [-1,1]."""
    rng = np.random.default_rng(42)
    idx = []
    for ci in range(len(class_names)):
        ids = np.where(labels == ci)[0]
        if len(ids) > per_class:
            ids = rng.choice(ids, per_class, replace=False)
        idx.extend(ids.tolist())
    idx = np.array(idx)
    Y = TSNE(n_components=2, perplexity=30, init="pca",
             random_state=42).fit_transform(embs[idx])
    Y = Y - Y.min(0)
    Y = Y / (Y.max(0) + 1e-8) * 2 - 1
    return [{"x": round(float(Y[i, 0]), 3), "y": round(float(Y[i, 1]), 3),
             "city": class_names[labels[idx[i]]]} for i in range(len(idx))]


def read_geoscore(arch):
    try:
        d = json.loads(Path(f"results/eval_{arch}_v2.json").read_text(encoding="utf-8"))
        return round(float(d.get("mean_geoscore")), 1)
    except Exception:  # noqa: BLE001
        return None


def main():
    device = get_device()
    out = {"class_names": None, "test_size": None, "models": []}
    for arch, label, sz in MODELS:
        print(f">>> {arch} …")
        class_names, y, p, confs, embs = infer(arch, sz, device)
        C = len(class_names)
        cm = np.zeros((C, C), dtype=int)
        for t, pr in zip(y, p):
            cm[t, pr] += 1
        top1, macro_f1, bacc, per_class = metrics_from_cm(cm)
        rel = reliability(confs, (y == p).astype(float))
        print(f"    tsne …")
        tsne = fit_tsne(embs, y, class_names)
        out["class_names"] = class_names
        out["test_size"] = int(len(y))
        out["models"].append({
            "id": arch, "label": label,
            "top1": top1, "macro_f1": macro_f1, "balanced_acc": bacc,
            "geoscore": read_geoscore(arch),
            "confusion": cm.tolist(),
            "per_class": [dict(city=class_names[i], **per_class[i]) for i in range(C)],
            "reliability": rel,
            "tsne": tsne,
        })
        print(f"    top1={top1*100:.1f}% macroF1={macro_f1*100:.1f}% ECE={rel['ece']} "
              f"geoscore={out['models'][-1]['geoscore']} tsne_pts={len(tsne)}")
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark_v2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] results/benchmark_v2.json")


if __name__ == "__main__":
    main()
