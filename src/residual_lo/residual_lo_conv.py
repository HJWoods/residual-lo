"""Residual LO applied as a convolution: shared W0 conv + code-conditioned Δ.

Same spirit as ``LinearOperatorLayer`` / ``residual_lo``:

- ``BinaryGate`` STE hard codes (K bits)
- shared base map ``W0`` (here a ``Conv2d``)
- hypernet ``code → θ`` generating a low-rank residual affine on channels
- ``y = W0⋆x + α · Δ_θ(c)(x)`` with the usual ``diag_low_rank`` / ``low_rank``
  K·rank bottleneck (residual is pointwise / 1×1 on channels)

Gate modes
----------
- ``pool`` (default): global-average-pool → one code per image (CIFAR-friendly)
- ``spatial``: gate each spatial location's channel vector (token analogue;
  unique-code batching groups locations by hard region)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from residual_lo.linear_operator import (
    AffineMode,
    BinaryGate,
    GeneratorType,
    ResidualHypernet,
    apply_generated_operator,
    generated_parameter_count,
    unique_code_indices,
)
from residual_lo.mlp import MLP

GateMode = Literal["pool", "spatial"]


def apply_generated_operator_nchw(
    x: torch.Tensor,
    parameters: torch.Tensor,
    in_channels: int,
    out_channels: int,
    *,
    affine_mode: AffineMode = "diag_low_rank",
    rank: int | None = None,
    bias: bool = True,
) -> torch.Tensor:
    """Apply a generated channel affine at every spatial location.

    ``x`` is ``(B, Cin, H, W)``, ``parameters`` is ``(B, g)`` (one row per
    image) or ``(B*H*W, g)`` when codes are already spatial-expanded.
    """
    if x.ndim != 4:
        raise ValueError(f"expected NCHW input, got shape {tuple(x.shape)}")
    b, c_in, h, w = x.shape
    if c_in != in_channels:
        raise ValueError(f"in_channels={in_channels} but x has {c_in}")
    if parameters.shape[0] == b:
        # One θ per image → expand across spatial so apply_* sees (BHW, g).
        flat = x.permute(0, 2, 3, 1).reshape(b * h * w, in_channels)
        params = (
            parameters.unsqueeze(1)
            .expand(b, h * w, parameters.shape[-1])
            .reshape(b * h * w, parameters.shape[-1])
        )
        out = apply_generated_operator(
            flat,
            params,
            in_channels,
            out_channels,
            affine_mode=affine_mode,
            rank=rank,
            bias=bias,
        )
        return out.reshape(b, h, w, out_channels).permute(0, 3, 1, 2).contiguous()
    if parameters.shape[0] == b * h * w:
        flat = x.permute(0, 2, 3, 1).reshape(b * h * w, in_channels)
        out = apply_generated_operator(
            flat,
            parameters,
            in_channels,
            out_channels,
            affine_mode=affine_mode,
            rank=rank,
            bias=bias,
        )
        return out.reshape(b, h, w, out_channels).permute(0, 3, 1, 2).contiguous()
    raise ValueError(
        f"parameters batch {parameters.shape[0]} matches neither B={b} "
        f"nor B·H·W={b * h * w}"
    )


class ResidualLOConv(nn.Module):
    """Binary-gated residual convolution with a K·rank channel residual.

    Mainline: ``use_residual_base=True``, ``affine_mode=diag_low_rank``,
    ``generator_type=residual_mlp``, ``gate_mode=pool``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        codes_per_head: int = 16,
        rank: int = 16,
        affine_mode: AffineMode = "diag_low_rank",
        generator_type: GeneratorType = "residual_mlp",
        hypernet_hidden_sizes: list[int] | None = None,
        output_initialization_scale: float = 0.05,
        bias: bool = True,
        use_residual_base: bool = True,
        gate_mode: GateMode = "pool",
        unique_code_batching: bool = True,
        unique_code_batching_max_frac: float = 0.85,
        gate_temperature: float = 1.0,
        residual_highway: bool = True,
        hypernet_final_ln: bool = False,
    ):
        super().__init__()
        if generator_type not in ("mlp", "residual_mlp"):
            raise ValueError(f"unknown generator_type={generator_type!r}")
        if gate_mode not in ("pool", "spatial"):
            raise ValueError(f"unknown gate_mode={gate_mode!r}")
        if affine_mode != "full" and (rank is None or rank <= 0):
            raise ValueError(f"rank must be positive for affine_mode={affine_mode!r}")
        if affine_mode == "full":
            rank = None
        if generator_type == "residual_mlp" and hypernet_hidden_sizes is None:
            hypernet_hidden_sizes = [128, 128, 128, 128]
        if generator_type == "mlp" and hypernet_hidden_sizes is None:
            hypernet_hidden_sizes = [128, 128]
        if padding is None:
            padding = kernel_size // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.affine_mode: AffineMode = affine_mode
        self.rank = rank
        self.generator_type: GeneratorType = generator_type
        self.use_residual_base = use_residual_base
        self.gate_mode: GateMode = gate_mode
        self.unique_code_batching = unique_code_batching
        self.unique_code_batching_max_frac = float(unique_code_batching_max_frac)
        self.codes = codes_per_head
        self.hypernet_hidden_sizes = list(hypernet_hidden_sizes or [])
        self.residual_highway = residual_highway
        self.hypernet_final_ln = bool(hypernet_final_ln)

        self.gate = BinaryGate(
            in_channels, codes_per_head, temperature=gate_temperature
        )
        generated_parameters = generated_parameter_count(
            in_channels,
            out_channels,
            affine_mode=affine_mode,
            rank=rank,
            bias=bias,
        )
        self.generated_parameters = generated_parameters

        self.hypernet: nn.Module | None
        self.parameter_head: nn.Linear | None
        self.arch_encoder: nn.Module | None = None
        if generator_type == "mlp":
            self.hypernet = MLP(
                in_features=codes_per_head,
                out_features=self.hypernet_hidden_sizes[-1],
                hidden_sizes=self.hypernet_hidden_sizes,
                activation=nn.GELU(),
            )
            self.parameter_head = nn.Linear(
                self.hypernet_hidden_sizes[-1], generated_parameters, bias=False
            )
            with torch.no_grad():
                self.parameter_head.weight.mul_(output_initialization_scale)
            self.arch_encoder = None
        else:
            self.hypernet = None
            self.parameter_head = None
            self.arch_encoder = ResidualHypernet(
                codes_per_head,
                generated_parameters,
                hidden_sizes=list(self.hypernet_hidden_sizes),
                highway=residual_highway,
                final_ln=self.hypernet_final_ln,
                output_initialization_scale=output_initialization_scale,
            )

        if use_residual_base:
            self.base_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            self.base_conv = None
            # Stride/padding still needed when residual-only; apply as 1×1 then
            # optional downsample via avg-pool if stride>1 is unusual — require
            # residual base for stride≠1.
            if stride != 1:
                raise ValueError(
                    "stride!=1 requires use_residual_base=True "
                    "(residual Δ is pointwise 1×1 on channels)"
                )

    def generate_parameters(self, code: torch.Tensor) -> torch.Tensor:
        if self.generator_type == "mlp":
            assert self.hypernet is not None and self.parameter_head is not None
            return self.parameter_head(self.hypernet(code))
        assert self.arch_encoder is not None
        return self.arch_encoder(code)

    def _gate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Features fed to BinaryGate: pooled (B, C) or spatial (BHW, C)."""
        if self.gate_mode == "pool":
            return x.mean(dim=(2, 3))
        # spatial: treat each location as a token
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b * h * w, c)

    def _generate_for_codes(self, code: torch.Tensor) -> torch.Tensor:
        """Generate θ, optionally once per unique hard code then scatter."""
        if not self.unique_code_batching or code.shape[0] <= 1:
            return self.generate_parameters(code)
        hard = (code.detach() > 0.5).to(torch.int64)
        first, inverse = unique_code_indices(hard)
        n_unique = first.shape[0]
        batch = code.shape[0]
        if n_unique >= max(1, int(self.unique_code_batching_max_frac * batch)):
            return self.generate_parameters(code)
        return self.generate_parameters(code[first])[inverse]

    def apply_residual(
        self, x: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        """Pointwise channel residual Δ_θ(c)(x); handles pool vs spatial θ."""
        if self.gate_mode == "pool":
            return apply_generated_operator_nchw(
                x,
                parameters,
                self.in_channels,
                self.out_channels,
                affine_mode=self.affine_mode,
                rank=self.rank,
                bias=self.use_bias,
            )
        # spatial: parameters already (BHW, g); still need NCHW layout apply
        b, _, h, w = x.shape
        if parameters.shape[0] != b * h * w:
            raise ValueError(
                f"spatial gate expects BHW params, got {parameters.shape[0]} "
                f"vs {b * h * w}"
            )
        return apply_generated_operator_nchw(
            x,
            parameters,
            self.in_channels,
            self.out_channels,
            affine_mode=self.affine_mode,
            rank=self.rank,
            bias=self.use_bias,
        )

    def forward_from_codes(
        self, x: torch.Tensor, code: torch.Tensor
    ) -> torch.Tensor:
        """Apply Δ then add W0⋆x, matching residual spatial size to the base.

        Pointwise Δ is computed at input resolution. When ``stride`` / ``padding``
        make ``base_conv`` change H×W (including odd maps and non-``k//2``
        padding), resize Δ with adaptive average pooling so the add always
        aligns — never assume ``avg_pool2d(kernel=stride)`` matches Conv2d.
        """
        parameters = self._generate_for_codes(code)
        residual = self.apply_residual(x, parameters)
        if self.use_residual_base:
            assert self.base_conv is not None
            base = self.base_conv(x)
            if residual.shape[-2:] != base.shape[-2:]:
                residual = F.adaptive_avg_pool2d(residual, base.shape[-2:])
            return base + residual
        # Residual-only: __init__ already rejects stride≠1; Δ stays at input H×W.
        return residual

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self._gate_features(x)
        code, _soft = self.gate(feats)
        output = self.forward_from_codes(x, code)
        # Return codes in (B, K) for pool mode; spatial returns (BHW, K).
        return output, code

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ResidualLOConvBlock(nn.Module):
    """ResidualLOConv → optional MaxPool; returns ``(y, code)``.

    No BN / GELU / other activation on the data path after LO. Nonlinearity
    lives only inside the LO hypernet (code→θ). MaxPool is spatial downsample
    only (not an activation).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        codes: int = 16,
        rank: int = 16,
        affine_mode: AffineMode = "diag_low_rank",
        generator_type: GeneratorType = "residual_mlp",
        hypernet_width: int = 128,
        hypernet_depth: int = 4,
        gate_mode: GateMode = "pool",
        pool: bool = True,
    ):
        super().__init__()
        self.lo = ResidualLOConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            codes_per_head=codes,
            rank=rank,
            affine_mode=affine_mode,
            generator_type=generator_type,
            hypernet_hidden_sizes=[hypernet_width] * max(1, hypernet_depth),
            gate_mode=gate_mode,
            use_residual_base=True,
            unique_code_batching=True,
            output_initialization_scale=0.05,
        )
        self.pool = nn.MaxPool2d(2) if pool else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, code = self.lo(x)
        if self.pool is not None:
            y = self.pool(y)
        return y, code
