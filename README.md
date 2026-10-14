# residual-lo — Discrete Bottleneck Layers

Installable **Discrete Bottleneck Layers** for discrete interpretability.

Mainline recipe (TinyGPT FineWeb / SmolTalk):

- `BinaryGate` STE codes → one hard region per token
- shared \(W_0,b_0\) + code-conditioned residual \(\Delta(c)\) (`use_residual_base=True`)
- `diag_low_rank` residual map
- `residual_mlp` hypernet (code→θ only)
- **data-path purity:** no post-layer activation / BN on \(x\)

Transformer training (TinyGPT, BabyLM, instruct SFT, chat) lives in [`HJWoods/DiscreteInterp`](https://github.com/HJWoods/DiscreteInterp) and **imports** this package — it must not reimplement gates, hypernets, or generated affines.

Package / import name stays `residual-lo` / `residual_lo`; the layers themselves are **Discrete Bottleneck Layers** (`DiscreteBottleneckLayer`, `DiscreteBottleneckConv`).

## Install

```bash
uv add git+https://github.com/HJWoods/residual-lo.git

# Editable checkout
uv add --editable /path/to/residual-lo

uv sync --extra dev
```

## Import

```python
from residual_lo import (
    DiscreteBottleneckLayer,
    DiscreteBottleneckConv,
    DiscreteBottleneckConvBlock,
    BinaryGate,
    gate_diversity_loss,
)

layer = DiscreteBottleneckLayer(
    in_features=256,
    out_features=256,
    codes_per_head=16,
    output_initialization_scale=0.1,
    affine_mode="diag_low_rank",
    rank=64,
    generator_type="residual_mlp",
    use_residual_base=True,
)
```

## Constraint (data-path purity)

No GELU/ReLU/SiLU/BN on the data path after \(W_0\) / after a Discrete Bottleneck Layer. Activations are allowed only inside hypernetworks (code→θ).

## Tests

```bash
uv sync --extra dev
uv run pytest -q
```

## MLP benchmarks (no conv / no attention)

Compares plain GELU MLPs vs Discrete Bottleneck Layers on GPU:

- forward / forward+backward throughput for \(d\!\to\!d\) maps
- short synthetic regression (smooth + piecewise-affine targets)

```bash
./scripts/run_mlp_benchmarks.sh                  # full sweep on CUDA
./scripts/run_mlp_benchmarks.sh --quick          # smaller smoke run
# writes benchmarks/results/mlp_vs_residual_lo.json
```
