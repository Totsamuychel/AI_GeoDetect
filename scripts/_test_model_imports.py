"""Швидкий тест завантаження всіх 3 моделей."""
import sys, torch
sys.path.insert(0, "code")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
print(f"transformers: ", end="")
import transformers; print(transformers.__version__)
print(f"huggingface_hub: ", end="")
import huggingface_hub; print(huggingface_hub.__version__)

from models import build_model

for arch in ["baseline", "streetclip", "geoclip"]:
    print(f"\n--- {arch} ---")
    try:
        m = build_model(arch, num_classes=3, pretrained=True)
        m = m.to(device)
        x = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            if arch == "geoclip":
                coords = torch.randn(2, 2).to(device)
                out = m(x, coords=coords)
                print(f"  output keys: {list(out.keys())}")
                print(f"  logits shape: {out['logits'].shape}")
            else:
                out = m(x)
                print(f"  output shape: {out.shape}")
        print(f"  OK")
    except Exception as e:
        print(f"  ERROR: {e}")
