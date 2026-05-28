"""
Порівнює requirements.txt з фактично встановленими пакетами.
Виводить таблицю: OK / VERSION MISMATCH / NOT INSTALLED
"""
import sys
import importlib.metadata as meta
from pathlib import Path
from packaging.version import Version
from packaging.requirements import Requirement

REQ_FILE = Path(__file__).parent.parent / "requirements.txt"

# Читаємо requirements.txt, пропускаємо коментарі та порожні рядки
requirements = []
for line in REQ_FILE.read_text().splitlines():
    line = line.split("#")[0].strip()
    if not line:
        continue
    try:
        requirements.append(Requirement(line))
    except Exception:
        pass

# Збираємо встановлені версії
installed = {}
for dist in meta.distributions():
    name = dist.metadata["Name"]
    if name:
        installed[name.lower().replace("-", "_")] = dist.metadata["Version"]

# Перевірка
ok = []
mismatch = []
missing = []

for req in requirements:
    key = req.name.lower().replace("-", "_")
    inst_ver = installed.get(key)

    if inst_ver is None:
        missing.append((req.name, str(req.specifier)))
        continue

    try:
        v = Version(inst_ver)
        satisfies = all(v in spec for spec in req.specifier) if req.specifier else True
    except Exception:
        satisfies = True  # невалідна версія — пропускаємо

    if satisfies:
        ok.append((req.name, inst_ver, str(req.specifier)))
    else:
        mismatch.append((req.name, inst_ver, str(req.specifier)))

# Вивід
W = 28
print(f"\n{'Package':<{W}} {'Installed':<14} {'Required':<20} Status")
print("-" * 75)

for name, ver, spec in sorted(ok):
    print(f"{'[OK]':<6} {name:<{W}} {ver:<14} {spec}")

if mismatch:
    print()
    for name, ver, spec in sorted(mismatch):
        print(f"{'[VER]':<6} {name:<{W}} {ver:<14} {spec}")

if missing:
    print()
    for name, spec in sorted(missing):
        print(f"{'[---]':<6} {name:<{W}} {'NOT INSTALLED':<14} {spec}")

print()
print(f"Summary: {len(ok)} OK | {len(mismatch)} version mismatch | {len(missing)} missing")

# Позначаємо критичні (потрібні для train.py)
CRITICAL = {
    "numpy", "pandas", "scipy", "scikit_learn",
    "transformers", "huggingface_hub", "tokenizers",
    "pillow", "torch", "torchvision",
    "tqdm", "pyyaml", "packaging", "requests",
    "python_dotenv", "rich",
}
crit_bad = [n for n, *_ in mismatch if n.lower().replace("-","_") in CRITICAL]
crit_miss = [n for n, *_ in missing if n.lower().replace("-","_") in CRITICAL]
if crit_bad or crit_miss:
    print(f"\nCRITICAL for training — version mismatch: {crit_bad or 'none'}")
    print(f"CRITICAL for training — missing:          {crit_miss or 'none'}")
else:
    print("\nAll critical training packages OK.")
