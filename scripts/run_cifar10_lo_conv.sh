#!/usr/bin/env bash
# Train CIFAR-10 ResidualLOConv, then param-matched dense Conv2d baseline.
# Default: --no-amp (LO AMP can nan / collapse unique codes).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${PYTHON:-python3}"
fi
DATA_ROOT="${DATA_ROOT:-/home/morisan/Desktop/Projects/ConceptLearning/data}"
OUT_BASE="${OUT_BASE:-${ROOT}/runs/cifar10_lo_conv}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-128}"
MAX_WALL="${MAX_WALL_SECONDS:-5400}"  # 90 min hard cap for the pair
PAIR_T0_S="$(date +%s)"
WANDB_FLAG=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_FLAG=(--wandb)
fi

mkdir -p "$OUT_BASE"
COMMON=(
  --data-root "$DATA_ROOT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH"
  --codes 16
  --rank 16
  --affine-mode diag_low_rank
  --generator-type residual_mlp
  --hypernet-width 128
  --hypernet-depth 4
  --channels 32 64 128
  --gate-mode pool
  --no-amp
  "${WANDB_FLAG[@]}"
)

echo "=== LO-conv arm ==="
"$PY" -u experiments/cifar10_lo_conv.py \
  --model lo \
  --output-dir "${OUT_BASE}/lo" \
  --wandb-run-name "cifar10-lo-conv-K16-r16" \
  --max-wall-seconds "$MAX_WALL" \
  "${COMMON[@]}" \
  2>&1 | tee "${OUT_BASE}/lo_train.log"

GELU_MAX_WALL="$MAX_WALL"
MAX_WALL_INT="${MAX_WALL%.*}"
if [[ -n "$MAX_WALL_INT" ]] && [[ "$MAX_WALL_INT" =~ ^[0-9]+$ ]] && (( MAX_WALL_INT > 0 )); then
  PAIR_ELAPSED_S=$(( $(date +%s) - PAIR_T0_S ))
  GELU_MAX_WALL=$(( MAX_WALL_INT - PAIR_ELAPSED_S ))
  if (( GELU_MAX_WALL < 1 )); then
    GELU_MAX_WALL=1
  fi
fi

echo "=== Dense-conv arm ==="
"$PY" -u experiments/cifar10_lo_conv.py \
  --model dense \
  --output-dir "${OUT_BASE}/dense" \
  --wandb-run-name "cifar10-dense-conv-matched" \
  --max-wall-seconds "$GELU_MAX_WALL" \
  "${COMMON[@]}" \
  2>&1 | tee "${OUT_BASE}/dense_train.log"

"$PY" - <<PY
import json
from pathlib import Path
base = Path("${OUT_BASE}")
rows = {}
for name in ("lo", "dense"):
    p = base / name / "summary.json"
    if p.is_file():
        rows[name] = json.loads(p.read_text())
out = base / "compare.json"
out.write_text(json.dumps(rows, indent=2))
print("=== compare ===")
for name, s in rows.items():
    print(
        f"{name}: best_test={s.get('best_test_acc')} "
        f"params={s.get('params')} channels={s.get('channels')} "
        f"wall_s={s.get('wall_s')}"
    )
PY
