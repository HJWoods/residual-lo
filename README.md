# residual-lo

Installable discrete / residual **linear-operator** layers for interpretability experiments.

- `LinearOperatorLayer` — BinaryGate STE codes → hypernet-generated affine (mainline: shared \(W_0\) + `diag_low_rank` residual)
- `ResidualLOConv` / `ResidualLOConvBlock` — same idea as a convolution (shared \(W_0\) conv + channel residual)
- Vision experiments: CIFAR-10 LO-conv, MNIST-family LO vs MLP

**Hard rule:** all LO *layer* logic lives here. Transformer / TinyGPT / BabyLM / SFT / chat live in [`HJWoods/DiscreteInterp`](https://github.com/HJWoods/DiscreteInterp) and import this package — they must not reimplement gates, hypernets, or generated affines.

## Install

```bash
# From GitHub (once the repo exists)
uv add git+https://github.com/HJWoods/residual-lo.git

# Editable checkout
uv add --editable /path/to/residual-lo

# Library + vision experiment extras
uv sync --extra vision --extra dev
```

## Import

```python
from residual_lo import (
    LinearOperatorLayer,
    ResidualLOConv,
    ResidualLOConvBlock,
    gate_diversity_loss,
)
```

## Experiments

```bash
uv run python experiments/mnist.py --help
uv run python experiments/cifar10_lo_conv.py --help
# or: ./scripts/run_cifar10_lo_conv.sh
```

## Constraint (data-path purity)

No GELU/ReLU/SiLU/BN on the data path after \(W_0\) / after ResidualLO*. Activations are allowed only inside hypernetworks (code→θ).

## Tests

```bash
uv sync --extra dev
uv run pytest -q
```
