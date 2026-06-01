from __future__ import annotations

import sys
from pathlib import Path

# Add project root and webapp to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))

import torch
from PIL import Image
from registry import get_registry, CITY_INFO

def main():
    test_folder = ROOT / "test photo"
    if not test_folder.exists():
        print(f"Folder not found: {test_folder}")
        return

    images = list(test_folder.glob("*.jpg")) + list(test_folder.glob("*.png"))
    if not images:
        print(f"No images found in {test_folder}")
        return

    reg = get_registry()
    available_models = [m["id"] for m in reg.available() if m["available"]]
    
    if not available_models:
        print("No models available in checkpoints/ directory.")
        return

    print(f"Testing on {len(images)} images using models: {', '.join(available_models)}")
    print("=" * 80)

    for img_path in images:
        print(f"\nIMAGE: {img_path.name}")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue

        for arch in available_models:
            print(f"\n  MODEL: {arch.upper()}")
            try:
                result = reg.predict(arch, img)
                
                # Check OOD
                ood = result.get("ood", {})
                if ood.get("is_ood"):
                    print(f"  [!] OUT-OF-DISTRIBUTION DETECTED (sim={ood.get('max_similarity')}, threshold={ood.get('threshold')})")
                    print("  This photo likely doesn't belong to any of the known cities.")
                
                # Top-3 predictions
                preds = result.get("predictions", [])[:3]
                for i, p in enumerate(preds, 1):
                    bar = "█" * int(p['prob'] * 20)
                    print(f"    #{i} {p['city_ua']:10s} {p['prob']*100:5.1f}% |{bar:<20}|")
            except Exception as e:
                print(f"    Error during prediction: {e}")
        print("-" * 80)

if __name__ == "__main__":
    main()
