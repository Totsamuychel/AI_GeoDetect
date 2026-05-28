#!/usr/bin/env bash
# =============================================================================
# runpod_train_all.sh — Послідовне навчання всіх 3 архітектур
#
# Використання (з кореня проекту після runpod_setup.sh):
#   bash scripts/runpod_train_all.sh
#
# Порядок: baseline → streetclip → geoclip
# Логи: logs/baseline.log, logs/streetclip.log, logs/geoclip.log
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

mkdir -p logs

run_model() {
    local arch="$1"
    local cfg="configs/${arch}.yaml"
    local log="logs/${arch}.log"

    echo ""
    echo "============================================================"
    echo "Training: $arch"
    echo "Config  : $cfg"
    echo "Log     : $log"
    echo "Started : $(date)"
    echo "============================================================"

    python code/train.py --config "$cfg" 2>&1 | tee "$log"

    echo ""
    echo "[$(date)] $arch DONE — best checkpoint:"
    ls -lh "checkpoints/${arch}/best_model.pth" 2>/dev/null || echo "  (checkpoint not found)"
}

START=$(date +%s)

run_model baseline
run_model streetclip
run_model geoclip

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "============================================================"
echo "ALL DONE in ${ELAPSED} min — $(date)"
echo "Checkpoints:"
for arch in baseline streetclip geoclip; do
    ls -lh "checkpoints/${arch}/best_model.pth" 2>/dev/null && echo "  checkpoints/${arch}/best_model.pth OK" || true
done
echo "============================================================"
