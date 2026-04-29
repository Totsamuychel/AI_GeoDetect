"""
Спільна конфігурація pytest для tests/.

Додає code/ до sys.path, щоб тести могли напряму імпортувати
модулі (models, dataset, metrics, ...), як це робиться в train.py / evaluate.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
