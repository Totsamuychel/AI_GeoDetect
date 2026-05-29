"""Тимчасовий тест: завантаження моделі + інференс + OOD на реальному фото."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import get_registry, ROOT
from PIL import Image

reg = get_registry()
arch = sys.argv[1] if len(sys.argv) > 1 else "baseline"

cases = {
    "Budapest (in-dist)": ROOT / "dataset/raw/mapillary/budapest/images/461217298283910.jpg",
}
# Перше OOD-фото з Румунії
ro = sorted((ROOT / "dataset/raw/osv5m/images/RO").glob("*.jpg"))
if ro:
    cases["Romania (OOD)"] = ro[0]
ua = sorted((ROOT / "dataset/raw/osv5m/images/UA").glob("*.jpg"))
if ua:
    cases["Ukraine (OOD)"] = ua[0]

for name, p in cases.items():
    if not p.exists():
        print(f"skip {name}: {p} missing"); continue
    res = reg.predict(arch, Image.open(p))
    top = res["predictions"][0]
    ood = res["ood"]
    print(f"\n[{arch}] {name}: top={top['city_ua']} {top['prob']*100:.1f}% | "
          f"OOD={ood['is_ood']} sim={ood['max_similarity']} thr={ood['threshold']}")
