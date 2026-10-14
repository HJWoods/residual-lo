#!/usr/bin/env python3
"""GPU benchmarks: plain MLP vs Discrete Bottleneck Layers (no conv / no attention).

Measures:
  1) Forward and forward+backward wall time / throughput for d→d maps
  2) Short synthetic regression train (smooth target + piecewise-affine target)

Usage:
  python benchmarks/mlp_vs_residual_lo.py --device cuda
  python benchmarks/mlp_vs_residual_lo.py --device cuda --quick
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from residual_lo import DiscreteBottleneckLayer, MLP


@dataclass
class TimingRow:
    name: str
    d: int
    batch: int
    params: int
    fwd_ms_median: float
    fwd_ms_p90: float
    bwd_ms_median: float
    tokens_per_s_fwd: float
    tokens_per_s_bwd: float


@dataclass
class TrainRow:
    name: str
    target: str
    d: int
    params: int
    steps: int
    final_mse: float
    best_mse: float
    seconds: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--train-batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="Feature widths for d→d blocks",
    )
    p.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1024, 2048],
        help="Token/row batch sizes for timing (4096×d=512 LO may OOM on 8GB)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller sweep for a fast smoke run",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "mlp_vs_residual_lo.json",
    )
    return p.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_mlp(d: int, hidden: list[int] | None = None) -> MLP:
    if hidden is None:
        # GPT-style expand ≈4× as a plain MLP reference (still no attention).
        hidden = [4 * d, 4 * d]
    return MLP(in_features=d, out_features=d, hidden_sizes=hidden, activation=nn.GELU())


def build_lo(
    d: int,
    *,
    codes: int = 16,
    rank: int | None = None,
    unique_code_batching: bool = True,
) -> DiscreteBottleneckLayer:
    if rank is None:
        rank = max(8, d // 4)
    return DiscreteBottleneckLayer(
        in_features=d,
        out_features=d,
        codes_per_head=codes,
        output_initialization_scale=0.1,
        affine_mode="diag_low_rank",
        rank=rank,
        generator_type="residual_mlp",
        use_residual_base=True,
        unique_code_batching=unique_code_batching,
    )


def build_param_matched_mlp(lo: DiscreteBottleneckLayer) -> MLP:
    return MLP.from_parameter_budget(
        in_features=lo.in_features,
        out_features=lo.out_features,
        parameter_budget=count_params(lo),
    )


class ModelWrap(nn.Module):
    """Uniform (x → y) API so timing/train loops treat MLP and LO the same."""

    def __init__(self, kind: str, core: nn.Module):
        super().__init__()
        self.kind = kind
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            return self.core(x)
        y, _code = self.core(x)
        return y


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def time_forward_backward(
    model: nn.Module,
    *,
    d: int,
    batch: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[list[float], list[float]]:
    model.train()
    x = torch.randn(batch, d, device=device)
    # Warmup
    for _ in range(warmup):
        y = model(x)
        loss = y.square().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
    _sync(device)

    fwd_ms: list[float] = []
    bwd_ms: list[float] = []
    for _ in range(iters):
        x = torch.randn(batch, d, device=device)
        _sync(device)
        t0 = time.perf_counter()
        y = model(x)
        _sync(device)
        t1 = time.perf_counter()
        loss = y.square().mean()
        loss.backward()
        _sync(device)
        t2 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        fwd_ms.append((t1 - t0) * 1e3)
        bwd_ms.append((t2 - t0) * 1e3)  # fwd+bwd wall
    del x
    _empty_cuda_cache(device)
    return fwd_ms, bwd_ms


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[idx]


def run_timing(
    *,
    dims: list[int],
    batches: list[int],
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
) -> list[TimingRow]:
    rows: list[TimingRow] = []
    for d in dims:
        torch.manual_seed(seed + d)
        # One arm on GPU at a time (RTX 3070 8GB + other processes).
        lo_ref = build_lo(d)
        arms: list[tuple[str, nn.Module]] = [
            ("mlp_expand4x2", ModelWrap("mlp", build_mlp(d))),
            (
                "mlp_param_matched",
                ModelWrap("mlp", build_param_matched_mlp(lo_ref)),
            ),
            ("discrete_bottleneck", ModelWrap("lo", lo_ref)),
        ]
        for name, model in arms:
            model = model.to(device)
            for batch in batches:
                # Skip known-heavy cells on 8GB cards (d=512 × B=4096 LO bwd).
                if d >= 512 and batch >= 4096 and name == "discrete_bottleneck":
                    print(f"[timing] SKIP {name} d={d} B={batch} (VRAM guard)")
                    continue
                try:
                    fwd, bwd = time_forward_backward(
                        model,
                        d=d,
                        batch=batch,
                        device=device,
                        warmup=warmup,
                        iters=iters,
                    )
                except torch.OutOfMemoryError:
                    print(f"[timing] OOM  {name} d={d} B={batch} — skipping")
                    _empty_cuda_cache(device)
                    continue
                fwd_med = statistics.median(fwd)
                bwd_med = statistics.median(bwd)
                rows.append(
                    TimingRow(
                        name=name,
                        d=d,
                        batch=batch,
                        params=count_params(model),
                        fwd_ms_median=fwd_med,
                        fwd_ms_p90=percentile(fwd, 90),
                        bwd_ms_median=bwd_med,
                        tokens_per_s_fwd=(batch / (fwd_med / 1e3)) if fwd_med > 0 else 0.0,
                        tokens_per_s_bwd=(batch / (bwd_med / 1e3)) if bwd_med > 0 else 0.0,
                    )
                )
                print(
                    f"[timing] {name:18s} d={d:4d} B={batch:5d} "
                    f"params={count_params(model):8d} "
                    f"fwd_med={fwd_med:7.3f}ms "
                    f"fwd+bwd_med={bwd_med:7.3f}ms "
                    f"tok/s_fwd={batch / (fwd_med / 1e3):10.0f}"
                )
            model.cpu()
            del model
            _empty_cuda_cache(device)
    return rows


def make_smooth_target(x: torch.Tensor) -> torch.Tensor:
    """Smooth nonlinear map — friendly to plain GELU MLPs."""
    return torch.tanh(x) + 0.25 * torch.sin(2.0 * x)


def make_piecewise_affine_target(x: torch.Tensor, *, n_regions: int = 8) -> torch.Tensor:
    """Region-wise affine map — closer to Discrete Bottleneck Layer inductive bias.

    Regions keyed by sign pattern of the first ``k`` coordinates (k≈log2 regions).
    """
    d = x.shape[-1]
    k = max(1, min(d, int(n_regions).bit_length() - 1 or 1))
    bits = (x[..., :k] > 0).to(torch.int64)
    # pack bits → region id in [0, 2^k)
    region = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for i in range(k):
        region = region | (bits[:, i] << i)
    # Fixed per-region scales/biases (deterministic from region id).
    scale = 0.5 + (region % 7).to(x.dtype).unsqueeze(-1) * 0.15
    bias = ((region % 5).to(x.dtype).unsqueeze(-1) - 2.0) * 0.1
    return scale * x + bias


def train_synthetic(
    model: nn.Module,
    *,
    target_fn,
    target_name: str,
    d: int,
    steps: int,
    batch: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> TrainRow:
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float("inf")
    t0 = time.perf_counter()
    final = float("inf")
    for step in range(steps):
        x = torch.randn(batch, d, device=device)
        with torch.no_grad():
            y = target_fn(x)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        val = float(loss.detach())
        best = min(best, val)
        final = val
        if step % max(1, steps // 5) == 0 or step + 1 == steps:
            print(
                f"[train] {getattr(model, 'kind', '?')}/{target_name} "
                f"d={d} step={step:4d} mse={val:.5f}"
            )
    _sync(device)
    seconds = time.perf_counter() - t0
    # Resolve display name from wrap
    name = "discrete_bottleneck" if getattr(model, "kind", "") == "lo" else "mlp"
    if hasattr(model, "core") and isinstance(model.core, MLP):
        # distinguish expand vs matched via param count later
        name = "mlp"
    return TrainRow(
        name=name,
        target=target_name,
        d=d,
        params=count_params(model),
        steps=steps,
        final_mse=final,
        best_mse=best,
        seconds=seconds,
    )


def run_train(
    *,
    dims: list[int],
    steps: int,
    batch: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> list[TrainRow]:
    rows: list[TrainRow] = []
    targets = [
        ("smooth", make_smooth_target),
        ("piecewise_affine", make_piecewise_affine_target),
    ]
    for d in dims:
        for target_name, target_fn in targets:
            torch.manual_seed(seed + 17 * d)
            lo_ref = build_lo(d)
            arms = (
                ("mlp_expand4x2", ModelWrap("mlp", build_mlp(d))),
                (
                    "mlp_param_matched",
                    ModelWrap("mlp", build_param_matched_mlp(lo_ref)),
                ),
                ("discrete_bottleneck", ModelWrap("lo", lo_ref)),
            )
            for label, model in arms:
                model = model.to(device)
                try:
                    row = train_synthetic(
                        model,
                        target_fn=target_fn,
                        target_name=target_name,
                        d=d,
                        steps=steps,
                        batch=batch,
                        lr=lr,
                        device=device,
                        seed=seed + d,
                    )
                except torch.OutOfMemoryError:
                    print(f"[train] OOM {label} target={target_name} d={d} — skipping")
                    model.cpu()
                    del model
                    _empty_cuda_cache(device)
                    continue
                row.name = label
                rows.append(row)
                print(
                    f"[train-done] {label:18s} target={target_name:16s} "
                    f"d={d} best_mse={row.best_mse:.5f} final={row.final_mse:.5f} "
                    f"({row.seconds:.1f}s)"
                )
                model.cpu()
                del model
                _empty_cuda_cache(device)
    return rows


def main() -> None:
    args = parse_args()
    if args.quick:
        args.dims = [128, 256]
        args.batches = [1024]
        args.warmup = 10
        args.iters = 40
        args.train_steps = 150
        args.train_batch = 1024

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    print(
        f"device={device} "
        f"name={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} "
        f"torch={torch.__version__}"
    )

    timing = run_timing(
        dims=args.dims,
        batches=args.batches,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
    )
    train = run_train(
        dims=args.dims,
        steps=args.train_steps,
        batch=args.train_batch,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "config": {
            "dims": args.dims,
            "batches": args.batches,
            "warmup": args.warmup,
            "iters": args.iters,
            "train_steps": args.train_steps,
            "train_batch": args.train_batch,
            "lr": args.lr,
            "seed": args.seed,
            "quick": args.quick,
        },
        "timing": [asdict(r) for r in timing],
        "train": [asdict(r) for r in train],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
