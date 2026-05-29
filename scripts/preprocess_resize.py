"""
preprocess_resize.py — Одноразове зменшення зображень для усунення I/O-боттлнеку.

Оригінали Mapillary — 2048px JPEG (~300-700 KB). Декодування такого файлу +
ресайз до 260px на КОЖНІЙ епосі завантажує CPU й лишає GPU простоювати.

Цей скрипт зменшує всі зображення з маніфестів до max-side=384px один раз і
зберігає у дзеркальну структуру під новим коренем (dataset_fast/). Декодування
384px JPEG у ~10x швидше → GPU перестає голодувати.

Запуск:
    python scripts/preprocess_resize.py
    python scripts/preprocess_resize.py --src dataset --dst dataset_fast --max-side 384
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # великі панорами — не блокувати


def _resize_one(args: tuple[str, str, int, int]) -> tuple[str, bool, str]:
    rel, src_root, max_side, quality = args[0], args[1], args[2], args[3]
    src = Path(src_root) / rel
    dst = Path(args[4]) / rel
    if dst.exists():
        return rel, True, "skip"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = max_side / max(w, h)
            if scale < 1.0:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.BICUBIC)
            im.save(dst, "JPEG", quality=quality, optimize=True)
        return rel, True, "ok"
    except Exception as e:
        return rel, False, str(e)[:80]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset", help="корінь оригіналів (image_root)")
    ap.add_argument("--dst", default="dataset_fast", help="корінь зменшених")
    ap.add_argument("--manifests", default="dataset/manifests")
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    man_dir = Path(args.manifests)
    rels: set[str] = set()
    for split in ["train", "val", "test"]:
        p = man_dir / f"{split}.csv"
        if p.exists():
            rels.update(pd.read_csv(p)["filepath"].astype(str).tolist())
    rels = sorted(rels)
    print(f"Зображень у маніфестах: {len(rels)}")
    print(f"{args.src} → {args.dst}, max-side={args.max_side}px, workers={args.workers}")

    tasks = [(r, args.src, args.max_side, args.quality, args.dst) for r in rels]
    ok = skip = fail = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_resize_one, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rel, success, msg = f.result()
            if not success:
                fail += 1; failures.append((rel, msg))
            elif msg == "skip":
                skip += 1
            else:
                ok += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(tasks)}  (ok={ok} skip={skip} fail={fail})")

    print(f"\nГотово: ok={ok}, skip={skip}, fail={fail}")
    if failures:
        print("Помилки (перші 10):")
        for rel, msg in failures[:10]:
            print(f"  {rel}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
