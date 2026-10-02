"""CIFAR-10: residual LO-convolution vs fair dense-conv baseline.

Replaces ``nn.Conv2d`` stages with ``ResidualLOConv`` (shared W0 conv +
code-conditioned low-rank channel residual, BinaryGate STE, same mainline
recipe as residual_lo MLP: ``diag_low_rank`` / ``residual_mlp`` / K·rank).

Mori constraint: no activation / BN on the LO data path after W0. Stages are
``ResidualLOConv`` (+ MaxPool for spatial downsample only). Dense twin matches
that layout — Conv2d (+ MaxPool), no GELU/BN either — so the compare is
Conv vs ResidualLOConv, not "dense gets nonlinearity and LO does not."
With ``--match-params`` (default) dense channels widen so total params ≈ LO.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from residual_lo import ResidualLOConvBlock

NUM_CLASSES = 10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Default LO channel ladder (before optional dense widening).
DEFAULT_CHANNELS = (32, 64, 128)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CIFAR-10 ResidualLOConv vs dense Conv2d baseline"
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/morisan/Desktop/Projects/ConceptLearning/data"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./runs/cifar10_lo_conv"),
    )
    p.add_argument("--model", choices=["lo", "dense"], required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--channels",
        type=int,
        nargs=3,
        default=list(DEFAULT_CHANNELS),
        metavar=("C1", "C2", "C3"),
        help="Channel widths after each conv stage (default 32 64 128).",
    )
    p.add_argument(
        "--match-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Widen dense channels to match LO param count (default on).",
    )
    p.add_argument("--codes", type=int, default=16)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument(
        "--affine-mode",
        choices=["low_rank", "diag_low_rank"],
        default="diag_low_rank",
    )
    p.add_argument(
        "--generator-type",
        choices=["mlp", "residual_mlp"],
        default="residual_mlp",
    )
    p.add_argument("--hypernet-width", type=int, default=128)
    p.add_argument("--hypernet-depth", type=int, default=4)
    p.add_argument(
        "--gate-mode",
        choices=["pool", "spatial"],
        default="pool",
        help="pool=GAP→one code/image; spatial=per-location codes.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--no-amp",
        action="store_true",
        default=True,
        help="Disable AMP (default on; LO AMP can collapse unique codes).",
    )
    p.add_argument(
        "--amp",
        action="store_false",
        dest="no_amp",
        help="Enable AMP (not recommended for LO).",
    )
    p.add_argument("--max-wall-seconds", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="discrete-interp")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    p.add_argument("--wandb-run-id", type=str, default="")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_cifar_root(data_root: Path) -> Path:
    data_root = data_root.expanduser().resolve()
    if (data_root / "cifar-10-batches-py").is_dir():
        return data_root
    if data_root.name == "cifar-10-batches-py" and data_root.is_dir():
        return data_root.parent
    for candidate in (
        Path("./data"),
        Path("/home/morisan/Desktop/Projects/DiscreteInterp/data"),
    ):
        if (candidate / "cifar-10-batches-py").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No local CIFAR-10 at {data_root}. Expected cifar-10-batches-py/"
    )


def loaders(data_root: Path, batch_size: int, num_workers: int):
    root = resolve_cifar_root(data_root)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=str(root), train=True, download=False, transform=train_tf
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(root), train=False, download=False, transform=test_tf
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return trainloader, testloader, root


class DenseConvBlock(nn.Module):
    """Conv2d → optional MaxPool (fair twin of no-act ResidualLOConvBlock).

    No BN/GELU: matches the LO stage constraint so the research question is
    Conv2d vs ResidualLOConv, not post-conv nonlinearity differences.
    Bias=True to mirror ResidualLOConv's biased W0 Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        pool: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.pool = nn.MaxPool2d(2) if pool else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        y = self.conv(x)
        if self.pool is not None:
            y = self.pool(y)
        return y, None


class CIFAR10LOConv(nn.Module):
    """Three-stage CIFAR CNN: LO-conv or dense-conv + linear head."""

    def __init__(
        self,
        *,
        model: str,
        channels: tuple[int, int, int],
        codes: int = 16,
        rank: int = 16,
        affine_mode: str = "diag_low_rank",
        generator_type: str = "residual_mlp",
        hypernet_width: int = 128,
        hypernet_depth: int = 4,
        gate_mode: str = "pool",
    ):
        super().__init__()
        self.model_kind = model
        self.channels = tuple(channels)
        c1, c2, c3 = self.channels
        if model == "lo":
            kw = dict(
                codes=codes,
                rank=rank,
                affine_mode=affine_mode,  # type: ignore[arg-type]
                generator_type=generator_type,  # type: ignore[arg-type]
                hypernet_width=hypernet_width,
                hypernet_depth=hypernet_depth,
                gate_mode=gate_mode,  # type: ignore[arg-type]
            )
            self.stage1 = ResidualLOConvBlock(3, c1, pool=True, **kw)
            self.stage2 = ResidualLOConvBlock(c1, c2, pool=True, **kw)
            self.stage3 = ResidualLOConvBlock(c2, c3, pool=False, **kw)
        elif model == "dense":
            self.stage1 = DenseConvBlock(3, c1, pool=True)
            self.stage2 = DenseConvBlock(c1, c2, pool=True)
            self.stage3 = DenseConvBlock(c2, c3, pool=False)
        else:
            raise ValueError(f"unknown model={model!r}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c3, NUM_CLASSES)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        codes: list[torch.Tensor] = []
        x, c1 = self.stage1(x)
        x, c2 = self.stage2(x)
        x, c3 = self.stage3(x)
        for c in (c1, c2, c3):
            if c is not None:
                codes.append(c)
        x = self.pool(x).flatten(1)
        logits = self.head(x)
        return logits, codes if codes else None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def dense_channels_for_budget(
    budget: int,
    *,
    base: tuple[int, int, int] = DEFAULT_CHANNELS,
    kernel_size: int = 3,
) -> tuple[int, int, int]:
    """Scale the dense channel ladder so total params ≈ ``budget``.

    Uses a uniform multiplier on (c1,c2,c3) and picks the integer scale whose
    param count is closest to the LO budget. Head included; no BN (no-act twin).
    """
    k2 = kernel_size * kernel_size

    def count(c1: int, c2: int, c3: int) -> int:
        # Conv (with bias, matching DenseConvBlock) + head Linear(c3, 10)
        conv = (
            3 * c1 * k2
            + c1
            + c1 * c2 * k2
            + c2
            + c2 * c3 * k2
            + c3
        )
        head = c3 * NUM_CLASSES + NUM_CLASSES
        return conv + head

    base_n = count(*base)
    if base_n <= 0:
        return base
    # Continuous scale from params ~ scale^2 (conv dominate).
    scale0 = math.sqrt(max(budget, 1) / base_n)
    best = base
    best_err = abs(base_n - budget)
    for scale in (
        max(1.0, scale0 * f) for f in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2)
    ):
        c1 = max(8, int(round(base[0] * scale)))
        c2 = max(8, int(round(base[1] * scale)))
        c3 = max(8, int(round(base[2] * scale)))
        # Keep monotonic ladder.
        c2 = max(c2, c1)
        c3 = max(c3, c2)
        n = count(c1, c2, c3)
        err = abs(n - budget)
        if err < best_err or (err == best_err and n >= budget):
            best_err = err
            best = (c1, c2, c3)
    # Fine search around best scale.
    for delta in range(-16, 17):
        c1 = max(8, best[0] + delta)
        c2 = max(c1, best[1] + 2 * delta)
        c3 = max(c2, best[2] + 4 * delta)
        n = count(c1, c2, c3)
        err = abs(n - budget)
        if err < best_err:
            best_err = err
            best = (c1, c2, c3)
    return best


def unique_codes_mean(codes_list: list[torch.Tensor] | None) -> float | None:
    if not codes_list:
        return None
    totals = []
    for codes in codes_list:
        hard = (codes > 0.5).float()
        # Cap unique scan for spatial mode (large BHW).
        if hard.shape[0] > 4096:
            hard = hard[:4096]
        totals.append(float(torch.unique(hard, dim=0).shape[0]))
    return sum(totals) / len(totals)


@torch.no_grad()
def evaluate(
    model: CIFAR10LOConv,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float, float | None]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    unique_sum = 0.0
    unique_batches = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        x = images.to(device, non_blocking=True)
        y = labels.to(device, non_blocking=True)
        logits, codes = model(x)
        loss_sum += float(criterion(logits, y))
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.numel()
        u = unique_codes_mean(codes)
        if u is not None:
            unique_sum += u
            unique_batches += 1
    n = max(total, 1)
    acc = correct / n
    ce = loss_sum / n
    unique = unique_sum / unique_batches if unique_batches else None
    model.train()
    return acc, ce, unique


def maybe_wandb(args: argparse.Namespace, config: dict):
    if not args.wandb:
        return None
    import wandb

    init_kwargs: dict = {
        "project": args.wandb_project,
        "name": args.wandb_run_name or f"cifar10-lo-conv-{args.model}",
        "config": config,
        "resume": "allow",
        "dir": str(args.output_dir),
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
    run = wandb.init(**init_kwargs)
    (args.output_dir / "wandb_run_id.txt").write_text(run.id)
    (args.output_dir / "wandb_url.txt").write_text(run.url)
    return run


def save_ckpt(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    best_test: float,
    args: argparse.Namespace,
    meta: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_test": best_test,
            "args": {
                k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
            },
            "meta": meta,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wall_t0 = time.perf_counter()
    deadline = (
        wall_t0 + args.max_wall_seconds if args.max_wall_seconds > 0 else float("inf")
    )

    trainloader, testloader, resolved_root = loaders(
        args.data_root, args.batch_size, args.num_workers
    )
    print(
        f"device={device} data_root={resolved_root} model={args.model} "
        f"no_amp={args.no_amp} gate_mode={args.gate_mode}"
    )

    base_channels = tuple(int(c) for c in args.channels)
    channels = base_channels
    lo_param_budget: int | None = None
    if args.model == "dense" and args.match_params:
        lo_ref = CIFAR10LOConv(
            model="lo",
            channels=base_channels,
            codes=args.codes,
            rank=args.rank,
            affine_mode=args.affine_mode,
            generator_type=args.generator_type,
            hypernet_width=args.hypernet_width,
            hypernet_depth=args.hypernet_depth,
            gate_mode=args.gate_mode,
        )
        lo_param_budget = lo_ref.count_parameters()
        channels = dense_channels_for_budget(lo_param_budget, base=base_channels)
        del lo_ref

    model = CIFAR10LOConv(
        model=args.model,
        channels=channels,
        codes=args.codes,
        rank=args.rank,
        affine_mode=args.affine_mode,
        generator_type=args.generator_type,
        hypernet_width=args.hypernet_width,
        hypernet_depth=args.hypernet_depth,
        gate_mode=args.gate_mode,
    ).to(device)
    param_count = model.count_parameters()
    print(
        f"params={param_count:,} channels={channels}"
        + (
            f" (matched to LO budget {lo_param_budget:,})"
            if lo_param_budget is not None
            else ""
        )
    )
    if args.model == "lo":
        print(
            f"LO-conv recipe: W0 Conv2d + {args.affine_mode} residual / "
            f"{args.generator_type} K={args.codes} r={args.rank} "
            f"hypernet=[{args.hypernet_width}]×{args.hypernet_depth} "
            f"gate={args.gate_mode}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    best_test = -1.0
    history: list[dict] = []

    ckpt_path = args.output_dir / "checkpoint.pt"
    if args.resume and ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_test = float(ckpt.get("best_test", -1.0))
        print(f"resumed epoch={start_epoch} best_test={best_test:.4f}")

    config = {
        "model": args.model,
        "channels": list(channels),
        "base_channels": list(base_channels),
        "match_params": args.match_params,
        "lo_param_budget": lo_param_budget,
        "params": param_count,
        "codes": args.codes,
        "rank": args.rank,
        "affine_mode": args.affine_mode,
        "generator_type": args.generator_type,
        "hypernet_width": args.hypernet_width,
        "hypernet_depth": args.hypernet_depth,
        "gate_mode": args.gate_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "no_amp": args.no_amp,
        "seed": args.seed,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    wb = maybe_wandb(args, config)

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopped_early = False

    for epoch in range(start_epoch, args.epochs):
        if time.perf_counter() >= deadline:
            stopped_early = True
            print(f"hit max_wall_seconds before epoch {epoch}")
            break
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        unique_sum = 0.0
        unique_batches = 0
        pbar = tqdm(trainloader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for images, labels in pbar:
            if time.perf_counter() >= deadline:
                stopped_early = True
                break
            x = images.to(device, non_blocking=True)
            y = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, codes = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * y.size(0)
            running_correct += int((logits.argmax(dim=1) == y).sum())
            running_total += y.size(0)
            u = unique_codes_mean(codes)
            if u is not None:
                unique_sum += u
                unique_batches += 1
            pbar.set_postfix(
                loss=f"{float(loss.detach()):.3f}",
                acc=f"{running_correct / max(running_total, 1):.3f}",
            )
        if stopped_early:
            print(f"hit max_wall_seconds mid-epoch {epoch}")
            break
        scheduler.step()
        train_acc = running_correct / max(running_total, 1)
        train_ce = running_loss / max(running_total, 1)
        train_unique = unique_sum / unique_batches if unique_batches else None
        test_acc, test_ce, test_unique = evaluate(model, testloader, device)
        if test_acc > best_test:
            best_test = test_acc
            save_ckpt(
                args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_test=best_test,
                args=args,
                meta={"test_acc": test_acc, "test_ce": test_ce},
            )
        save_ckpt(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_test=best_test,
            args=args,
            meta={},
        )
        row = {
            "epoch": epoch,
            "train_acc": train_acc,
            "train_ce": train_ce,
            "train_unique": train_unique,
            "test_acc": test_acc,
            "test_ce": test_ce,
            "test_unique": test_unique,
            "lr": float(scheduler.get_last_lr()[0]),
            "wall_s": time.perf_counter() - wall_t0,
        }
        history.append(row)
        uniq_s = (
            f" uniq≈{train_unique:.1f}" if train_unique is not None else ""
        )
        print(
            f"epoch {epoch+1:03d} train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
            f"best={best_test:.4f} ce={train_ce:.3f}/{test_ce:.3f}{uniq_s}"
        )
        if wb is not None:
            import wandb

            wandb.log({k: v for k, v in row.items() if v is not None}, step=epoch)

    peak_vram = None
    if device.type == "cuda":
        peak_vram = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    summary = {
        "model": args.model,
        "params": param_count,
        "channels": list(channels),
        "base_channels": list(base_channels),
        "lo_param_budget": lo_param_budget,
        "best_test_acc": best_test,
        "final_test_acc": history[-1]["test_acc"] if history else None,
        "final_train_acc": history[-1]["train_acc"] if history else None,
        "final_train_ce": history[-1]["train_ce"] if history else None,
        "final_test_ce": history[-1]["test_ce"] if history else None,
        "final_train_unique": history[-1].get("train_unique") if history else None,
        "epochs_ran": len(history),
        "stopped_early": stopped_early,
        "wall_s": time.perf_counter() - wall_t0,
        "peak_vram_mib": peak_vram,
        "config": config,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    if wb is not None:
        import wandb

        wandb.summary.update(
            {k: v for k, v in summary.items() if k not in ("history", "config")}
        )
        wb.finish()


if __name__ == "__main__":
    main()
