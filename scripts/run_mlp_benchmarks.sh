#!/usr/bin/env bash
# Run plain-MLP vs Discrete Bottleneck Layer GPU benchmarks (no conv / no attention).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/../DiscreteInterp/.venv/bin/python" ]]; then
    PYTHON="$ROOT/../DiscreteInterp/.venv/bin/python"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS=(--device cuda)
fi

echo "PYTHON=$PYTHON"
"$PYTHON" - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

exec "$PYTHON" "$ROOT/benchmarks/mlp_vs_residual_lo.py" "${ARGS[@]}"
