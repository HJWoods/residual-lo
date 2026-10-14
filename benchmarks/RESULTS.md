# MLP vs Discrete Bottleneck Layers (RTX 3070)

Run: `./scripts/run_mlp_benchmarks.sh --device cuda --batches 1024 2048 --dims 128 256 512`  
Artifact: [`results/mlp_vs_residual_lo.json`](results/mlp_vs_residual_lo.json)

Arms (no conv / no attention):

- `mlp_expand4x2` — GELU MLP with hidden `[4d, 4d]`
- `mlp_param_matched` — two-hidden GELU MLP matched to Discrete Bottleneck Layer param count via `MLP.from_parameter_budget`
- `discrete_bottleneck` — mainline `DiscreteBottleneckLayer` (`diag_low_rank`, `residual_mlp`, `use_residual_base`)

## Throughput (forward median)

| d | B | mlp_expand tok/s | matched tok/s | discrete_bottleneck tok/s | DBL slowdown vs expand |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 5.18e6 | 2.84e6 | 7.44e5 | ~7.0× |
| 128 | 2048 | 7.41e6 | 3.15e6 | 1.13e6 | ~6.5× |
| 256 | 1024 | 2.51e6 | 1.07e6 | 4.28e5 | ~5.9× |
| 256 | 2048 | 2.80e6 | 1.09e6 | 5.79e5 | ~4.8× |
| 512 | 1024 | 7.96e5 | 2.78e5 | 1.68e5 | ~4.7× |
| 512 | 2048 | 7.25e5 | 2.78e5 | 1.83e5 | ~4.0× |

## Short synthetic train (400 steps, AdamW 3e-3, B=2048) — best MSE

| d | target | expand | matched | discrete_bottleneck |
| ---: | --- | ---: | ---: | ---: |
| 128 | smooth | 0.113 | 0.117 | **0.105** |
| 128 | piecewise_affine | 0.081 | 0.096 | **0.042** |
| 256 | smooth | 0.123 | 0.548* | **0.106** |
| 256 | piecewise_affine | 0.095 | 0.834* | **0.043** |
| 512 | smooth | 0.137 | 0.557* | **0.108** |
| 512 | piecewise_affine | 0.122 | 0.865* | **0.058** |

\*Param-matched two-hidden MLP becomes very wide at Discrete Bottleneck Layer budgets and trains poorly at this LR/budget — expand MLP is the fair dense baseline for quality; matched is mainly a param-count control for speed.
