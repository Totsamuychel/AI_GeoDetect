"""Імпортує всі модулі проекту, ловить ImportError/SyntaxError."""
import sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

modules = [
    "augmentations", "dataset", "evaluate", "inference",
    "metrics", "models", "train", "tui_trainer", "utils", "visualize",
]
ok, bad = [], []
for m in modules:
    try:
        __import__(m)
        ok.append(m)
    except Exception as e:
        bad.append((m, type(e).__name__, str(e).split("\n")[0][:120]))

print(f"\n[OK]    {len(ok)}/{len(modules)} modules import cleanly")
for m in ok:
    print(f"  + {m}")
if bad:
    print(f"\n[FAIL]  {len(bad)}:")
    for m, kind, msg in bad:
        print(f"  - {m}: {kind}: {msg}")
sys.exit(0 if not bad else 1)
