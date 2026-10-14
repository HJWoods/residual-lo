import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from residual_lo.mlp import MLP

AffineMode = Literal["full", "low_rank", "diag_low_rank"]
GeneratorType = Literal["mlp", "residual_mlp"]


def generated_parameter_count(
    in_features: int,
    out_features: int,
    *,
    affine_mode: AffineMode = "full",
    rank: int | None = None,
    bias: bool = True,
) -> int:
    """Number of scalars emitted by the hypernet parameter head for one affine."""
    if affine_mode == "full":
        count = out_features * in_features
    elif affine_mode in ("low_rank", "diag_low_rank"):
        if rank is None or rank <= 0:
            raise ValueError(f"rank must be a positive int for affine_mode={affine_mode!r}")
        count = out_features * rank + rank * in_features
        if affine_mode == "diag_low_rank":
            count += min(in_features, out_features)
    else:
        raise ValueError(
            f"unknown affine_mode={affine_mode!r}; expected full, low_rank, or diag_low_rank"
        )
    if bias:
        count += out_features
    return count


def _batched_affine_mv(
    matrix: torch.Tensor, vectors: torch.Tensor
) -> torch.Tensor:
    """Batched y[b] = M[b] @ x[b] without an explicit unsqueeze/squeeze temp.

    ``matrix`` is (B, P, Q), ``vectors`` is (B, Q) → (B, P).
    """
    return torch.einsum("bpq,bq->bp", matrix, vectors)


def apply_generated_affine(
    x: torch.Tensor,
    parameters: torch.Tensor,
    in_features: int,
    out_features: int,
    bias: bool = True,
) -> torch.Tensor:
    """Apply a dense generated affine without sharing weights across the batch."""
    split = in_features * out_features
    weights = parameters[:, :split].reshape(-1, out_features, in_features)
    output = _batched_affine_mv(weights, x)
    if bias:
        output = output + parameters[:, split:]
    return output


def apply_generated_low_rank_affine(
    x: torch.Tensor,
    parameters: torch.Tensor,
    in_features: int,
    out_features: int,
    rank: int,
    bias: bool = True,
) -> torch.Tensor:
    """Apply y = U(Vx) + b without materializing W = UV."""
    u_size = out_features * rank
    v_size = rank * in_features
    u = parameters[:, :u_size].reshape(-1, out_features, rank)
    v = parameters[:, u_size : u_size + v_size].reshape(-1, rank, in_features)
    mid = _batched_affine_mv(v, x)
    output = _batched_affine_mv(u, mid)
    if bias:
        output = output + parameters[:, u_size + v_size :]
    return output


def apply_generated_diag_low_rank_affine(
    x: torch.Tensor,
    parameters: torch.Tensor,
    in_features: int,
    out_features: int,
    rank: int,
    bias: bool = True,
) -> torch.Tensor:
    """Apply y = U(Vx) + d⊙x_overlap + b without materializing W.

    The diagonal term adds an elementwise contribution on the leading
    ``min(in, out)`` coordinates (the main diagonal of a rectangular matrix).

    Square maps (``in_features == out_features``, the TinyGPT LO slot) take a
    fast path that avoids padding the diagonal contribution.
    """
    u_size = out_features * rank
    v_size = rank * in_features
    diag_size = min(in_features, out_features)
    u = parameters[:, :u_size].reshape(-1, out_features, rank)
    v = parameters[:, u_size : u_size + v_size].reshape(-1, rank, in_features)
    offset = u_size + v_size
    diagonal = parameters[:, offset : offset + diag_size]
    mid = _batched_affine_mv(v, x)
    output = _batched_affine_mv(u, mid)
    if diag_size == out_features:
        # Hot path for DiscreteBottleneckMLP (d→d): d⊙x with no pad/cat.
        output = output + diagonal * x[:, :diag_size]
    else:
        # Rectangular: pad the trailing out-diag_size coords with zeros.
        # Build via cat so grads flow (in-place index-assign into zeros drops them).
        diag_contrib = diagonal * x[:, :diag_size]
        pad = diag_contrib.new_zeros(diag_contrib.shape[0], out_features - diag_size)
        output = output + torch.cat([diag_contrib, pad], dim=1)
    if bias:
        output = output + parameters[:, offset + diag_size :]
    return output


def apply_diag_low_rank_from_unique(
    x: torch.Tensor,
    unique_parameters: torch.Tensor,
    inverse: torch.Tensor,
    in_features: int,
    out_features: int,
    rank: int,
    bias: bool = True,
) -> torch.Tensor:
    """diag_low_rank apply with params generated once per unique code.

    Parses ``unique_parameters`` (U, g) once, then indexes the factored U/V/diag/bias
    by ``inverse`` (N,) — same FLOPs as expanding the flat (N, g) vector first,
    but avoids one full-width gather of g (= 2·d·r + 2·d) before reshape.
    """
    u_size = out_features * rank
    v_size = rank * in_features
    diag_size = min(in_features, out_features)
    u = unique_parameters[:, :u_size].reshape(-1, out_features, rank)
    v = unique_parameters[:, u_size : u_size + v_size].reshape(-1, rank, in_features)
    offset = u_size + v_size
    diagonal = unique_parameters[:, offset : offset + diag_size]
    u_rows = u[inverse]
    v_rows = v[inverse]
    mid = _batched_affine_mv(v_rows, x)
    output = _batched_affine_mv(u_rows, mid)
    diag_rows = diagonal[inverse]
    if diag_size == out_features:
        output = output + diag_rows * x[:, :diag_size]
    else:
        diag_contrib = diag_rows * x[:, :diag_size]
        pad = diag_contrib.new_zeros(diag_contrib.shape[0], out_features - diag_size)
        output = output + torch.cat([diag_contrib, pad], dim=1)
    if bias:
        output = output + unique_parameters[:, offset + diag_size :][inverse]
    return output


def apply_generated_operator(
    x: torch.Tensor,
    parameters: torch.Tensor,
    in_features: int,
    out_features: int,
    *,
    affine_mode: AffineMode = "full",
    rank: int | None = None,
    bias: bool = True,
) -> torch.Tensor:
    if affine_mode == "full":
        return apply_generated_affine(
            x, parameters, in_features, out_features, bias=bias
        )
    if affine_mode == "low_rank":
        assert rank is not None
        return apply_generated_low_rank_affine(
            x, parameters, in_features, out_features, rank, bias=bias
        )
    if affine_mode == "diag_low_rank":
        assert rank is not None
        return apply_generated_diag_low_rank_affine(
            x, parameters, in_features, out_features, rank, bias=bias
        )
    raise ValueError(f"unknown affine_mode={affine_mode!r}")


# Public synonym (same implementation as apply_generated_operator).
apply_generated_parameters = apply_generated_operator


def pack_binary_codes(hard: torch.Tensor) -> torch.Tensor:
    """Pack binary bit rows (B, K) with K <= 63 into int64 codes."""
    if hard.ndim != 2:
        raise ValueError(f"expected hard codes of shape (B, K), got {tuple(hard.shape)}")
    if hard.shape[1] > 63:
        raise ValueError("pack_binary_codes supports at most 63 bits")
    # Bit-shift powers stay exact in int64 (avoid float 2**k for large K).
    powers = 1 << torch.arange(hard.shape[1], device=hard.device, dtype=torch.int64)
    return (hard.to(torch.int64) * powers).sum(dim=1)


def unique_code_indices(hard: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (first_occurrence_indices, inverse) for packed hard codes."""
    packed = pack_binary_codes(hard)
    unique, inverse = torch.unique(packed, return_inverse=True)
    # First occurrence of each unique packed value (stable for scatter).
    first = packed.new_full((unique.numel(),), packed.numel())
    positions = torch.arange(packed.numel(), device=packed.device)
    first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
    return first, inverse


class BinaryGate(nn.Module):
    def __init__(
        self,
        in_features: int,
        code_size: int,
        *,
        temperature: float = 1.0,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.linear = nn.Linear(in_features, code_size)
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear(x) / self.temperature
        # Straight-through estimator for the binary step function.
        soft = torch.sigmoid(logits)
        hard = (logits > 0.0).to(logits.dtype)
        code = hard + soft - soft.detach()
        return code, soft


def gate_diversity_loss(soft: torch.Tensor) -> torch.Tensor:
    """Encourage healthy unique-code utilization (binary soft codes).

    Per-bit Bernoulli entropy / balance / variance. Used as
    ``recon - coeff * gate_diversity_loss(soft)`` (full_lo control).
    """
    p = soft.clamp(1e-4, 1.0 - 1e-4)
    bit_entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log()).mean()
    bit_balance = (soft.mean(dim=0) - 0.5).pow(2).mean()
    soft_variance = soft.var(dim=0, unbiased=False).mean()
    return bit_entropy + soft_variance - bit_balance


class ResidualBlock(nn.Module):
    """Pre-norm residual or highway block over code features."""

    def __init__(self, width: int, *, highway: bool = True):
        super().__init__()
        self.highway = highway
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.GELU()
        if highway:
            self.gate_proj = nn.Linear(width, width)
        else:
            self.gate_proj = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        y = self.fc2(self.act(self.fc1(x)))
        if self.highway and self.gate_proj is not None:
            g = torch.sigmoid(self.gate_proj(x))
            return g * y + (1.0 - g) * h
        return h + y


class ResidualHypernet(nn.Module):
    """Deep residual/highway MLP: code → hidden → parameter head → θ.

    Still one affine per code; depth is only in the code→θ map.
    """

    def __init__(
        self,
        codes_per_head: int,
        generated_parameters: int,
        *,
        hidden_sizes: list[int] | None = None,
        highway: bool = True,
        final_ln: bool = False,
        output_initialization_scale: float = 1.0,
    ):
        super().__init__()
        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [128, 128, 128, 128]
        width = hidden_sizes[0]
        if any(h != width for h in hidden_sizes):
            # Force constant width for residual connections.
            hidden_sizes = [width] * len(hidden_sizes)
        self.input_proj = nn.Linear(codes_per_head, width)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, highway=highway) for _ in hidden_sizes]
        )
        self.final_ln = nn.LayerNorm(width) if final_ln else None
        self.parameter_head = nn.Linear(width, generated_parameters, bias=False)
        with torch.no_grad():
            self.parameter_head.weight.mul_(output_initialization_scale)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(code)
        for block in self.blocks:
            h = block(h)
        if self.final_ln is not None:
            h = self.final_ln(h)
        return self.parameter_head(h)


class DiscreteBottleneckLayer(nn.Module):
    """Discrete Bottleneck Layer: BinaryGate STE → one affine map per hard code.

    Mainline: ``use_residual_base=True``, ``affine_mode=diag_low_rank``,
    ``generator_type=residual_mlp``. Flat ``mlp`` is kept as a small control
    (tests / A/B). Optional ``use_residual_base`` learns shared ``W_0, b_0`` and
    generates residual ``ΔW_c`` so ``W_c = W_0 + ΔW_c``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        codes_per_head: int,
        output_initialization_scale: float,
        hypernet_hidden_sizes: list[int] | None = None,
        bias: bool = True,
        affine_mode: AffineMode = "diag_low_rank",
        rank: int | None = None,
        generator_type: GeneratorType = "residual_mlp",
        use_residual_base: bool = False,
        unique_code_batching: bool = False,
        unique_code_batching_max_frac: float = 0.85,
        gate_temperature: float = 1.0,
        residual_highway: bool = True,
        hypernet_final_ln: bool = False,
        residual_alpha_mode: Literal["none", "global", "code"] = "none",
        residual_alpha_init: float = 0.1,
    ):
        super().__init__()
        if generator_type not in ("mlp", "residual_mlp"):
            raise ValueError(
                f"unknown generator_type={generator_type!r}; "
                "mainline supports mlp and residual_mlp"
            )
        if not 0.0 < unique_code_batching_max_frac <= 1.0:
            raise ValueError(
                "unique_code_batching_max_frac must be in (0, 1]; "
                f"got {unique_code_batching_max_frac}"
            )
        if generator_type == "residual_mlp" and hypernet_hidden_sizes is None:
            hypernet_hidden_sizes = [128, 128, 128, 128]
        if generator_type == "mlp" and hypernet_hidden_sizes is None:
            hypernet_hidden_sizes = [128, 128]
        if generator_type == "mlp" and (
            hypernet_hidden_sizes is None or len(hypernet_hidden_sizes) == 0
        ):
            raise ValueError("mlp generator requires non-empty hypernet_hidden_sizes")
        if affine_mode != "full" and (rank is None or rank <= 0):
            raise ValueError(f"rank must be positive for affine_mode={affine_mode!r}")
        if affine_mode == "full":
            rank = None
        self.hypernet_hidden_sizes = (
            list(hypernet_hidden_sizes) if hypernet_hidden_sizes is not None else []
        )
        self.use_bias = bias
        self.affine_mode: AffineMode = affine_mode
        self.rank = rank
        self.generator_type: GeneratorType = generator_type
        self.gate_type = "binary"
        self.use_residual_base = use_residual_base
        self.unique_code_batching = unique_code_batching
        # When unique/batch exceeds this fraction, unique-code batching costs more
        # than it saves (hypernet already runs ~once per token; expand is pure overhead).
        self.unique_code_batching_max_frac = float(unique_code_batching_max_frac)
        self.gate_temperature = gate_temperature
        self.in_features = in_features
        self.out_features = out_features
        self.residual_highway = residual_highway
        self.hypernet_final_ln = bool(hypernet_final_ln)
        if residual_alpha_mode not in ("none", "global", "code"):
            raise ValueError(
                f"unknown residual_alpha_mode={residual_alpha_mode!r}"
            )
        self.residual_alpha_mode: Literal["none", "global", "code"] = residual_alpha_mode
        self.residual_alpha_init = float(residual_alpha_init)
        # Optional override used by curriculum ramps (multiplies learned α).
        self._residual_alpha_schedule: float = 1.0

        self.gate = BinaryGate(
            in_features, codes_per_head, temperature=gate_temperature
        )
        self.codes = codes_per_head
        generated_parameters = generated_parameter_count(
            in_features,
            out_features,
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
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            if bias:
                self.base_bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("base_bias", None)
        else:
            self.register_parameter("base_weight", None)
            self.register_parameter("base_bias", None)

        # Residual scale α: y = W0 x + α · Δ_θ(c)(x). softplus keeps α > 0.
        if residual_alpha_mode == "global":
            init = math.log(math.expm1(max(residual_alpha_init, 1e-6)))
            self.residual_alpha = nn.Parameter(torch.tensor(init))
            self.residual_alpha_proj = None
        elif residual_alpha_mode == "code":
            self.register_parameter("residual_alpha", None)
            self.residual_alpha_proj = nn.Linear(codes_per_head, 1, bias=True)
            with torch.no_grad():
                self.residual_alpha_proj.weight.zero_()
                init = math.log(math.expm1(max(residual_alpha_init, 1e-6)))
                self.residual_alpha_proj.bias.fill_(init)
        else:
            self.register_parameter("residual_alpha", None)
            self.residual_alpha_proj = None

    def init_residual_base_from_linear(self, linear: nn.Linear) -> None:
        """Diagnostic: copy a teacher Linear into shared W_0, b_0."""
        if not self.use_residual_base:
            raise RuntimeError("init_residual_base_from_linear requires use_residual_base")
        if linear.in_features != self.in_features or linear.out_features != self.out_features:
            raise ValueError(
                f"linear shape {(linear.out_features, linear.in_features)} "
                f"!= residual base {(self.out_features, self.in_features)}"
            )
        with torch.no_grad():
            self.base_weight.copy_(linear.weight)
            if self.base_bias is not None:
                if linear.bias is None:
                    self.base_bias.zero_()
                else:
                    self.base_bias.copy_(linear.bias)

    def init_residual_from_affine(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None
    ) -> None:
        """Copy a teacher / LS affine into shared W0,b0 (residual models only)."""
        if not self.use_residual_base or self.base_weight is None:
            raise RuntimeError("init_residual_from_affine requires use_residual_base")
        if weight.shape != self.base_weight.shape:
            raise ValueError(
                f"weight shape {tuple(weight.shape)} != "
                f"{tuple(self.base_weight.shape)}"
            )
        with torch.no_grad():
            self.base_weight.copy_(weight)
            if self.base_bias is not None:
                if bias is None:
                    self.base_bias.zero_()
                else:
                    self.base_bias.copy_(bias)

    def generate_parameters(self, code: torch.Tensor) -> torch.Tensor:
        """Map codes (N, K) → generated affine parameters (N, g)."""
        if self.generator_type == "mlp":
            assert self.hypernet is not None and self.parameter_head is not None
            return self.parameter_head(self.hypernet(code))
        assert self.arch_encoder is not None
        return self.arch_encoder(code)

    def generate_parameters_for_batch(self, code: torch.Tensor) -> torch.Tensor:
        """Generate params, optionally once per unique hard code then scatter."""
        if not self.unique_code_batching or code.shape[0] <= 1:
            return self.generate_parameters(code)
        hard = (code.detach() > 0.5).to(torch.int64)
        first, inverse = unique_code_indices(hard)
        unique_parameters = self.generate_parameters(code[first])
        return unique_parameters[inverse]

    def residual_alpha_value(
        self, code: torch.Tensor | None = None
    ) -> torch.Tensor | float:
        """Positive residual scale α (optionally per-row when mode=code)."""
        sched = float(self._residual_alpha_schedule)
        if self.residual_alpha_mode == "none":
            return sched
        if self.residual_alpha_mode == "global":
            assert self.residual_alpha is not None
            return F.softplus(self.residual_alpha) * sched
        assert self.residual_alpha_proj is not None
        if code is None:
            raise ValueError("code-conditioned residual α requires code")
        return F.softplus(self.residual_alpha_proj(code)).squeeze(-1) * sched

    def _scale_residual(
        self,
        residual: torch.Tensor,
        code: torch.Tensor | None = None,
        alpha: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if alpha is None:
            alpha = self.residual_alpha_value(code)
        if isinstance(alpha, float):
            if alpha == 1.0:
                return residual
            return residual * alpha
        if alpha.ndim == 0:
            return residual * alpha
        return residual * alpha.unsqueeze(-1)

    def apply_affine(
        self,
        x: torch.Tensor,
        parameters: torch.Tensor,
        *,
        code: torch.Tensor | None = None,
        alpha: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        residual = apply_generated_operator(
            x,
            parameters,
            self.in_features,
            self.out_features,
            affine_mode=self.affine_mode,
            rank=self.rank,
            bias=self.use_bias,
        )
        residual = self._scale_residual(residual, code=code, alpha=alpha)
        if self.use_residual_base:
            assert self.base_weight is not None
            base = F.linear(x, self.base_weight, self.base_bias)
            return base + residual
        return residual

    def apply_parameters(self, x: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """Alias for ``apply_affine`` (tests / older call sites)."""
        return self.apply_affine(x, parameters)

    def _apply_residual_from_unique(
        self,
        x: torch.Tensor,
        unique_parameters: torch.Tensor,
        inverse: torch.Tensor,
        *,
        code: torch.Tensor | None = None,
        alpha: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply W0 + Δ using params generated once per unique hard code."""
        if self.affine_mode == "diag_low_rank":
            assert self.rank is not None
            residual = apply_diag_low_rank_from_unique(
                x,
                unique_parameters,
                inverse,
                self.in_features,
                self.out_features,
                self.rank,
                bias=self.use_bias,
            )
        else:
            # low_rank / full: expand flat params (same as before).
            residual = apply_generated_operator(
                x,
                unique_parameters[inverse],
                self.in_features,
                self.out_features,
                affine_mode=self.affine_mode,
                rank=self.rank,
                bias=self.use_bias,
            )
        residual = self._scale_residual(residual, code=code, alpha=alpha)
        if self.use_residual_base:
            assert self.base_weight is not None
            base = F.linear(x, self.base_weight, self.base_bias)
            return base + residual
        return residual

    def _unique_code_inverse(self, hard: torch.Tensor) -> torch.Tensor:
        """Map each row's hard code to a dense unique-id in `[0, U)`."""
        k = hard.shape[-1]
        if k <= 63:
            bits = (hard > 0.5).to(torch.int64)
            powers = 1 << torch.arange(k, device=hard.device, dtype=torch.int64)
            ids = (bits * powers).sum(dim=-1)
            _unique_ids, inverse = torch.unique(ids, return_inverse=True)
            return inverse
        _unique_hard, inverse = torch.unique(hard, dim=0, return_inverse=True)
        return inverse

    def forward_from_codes(
        self,
        x: torch.Tensor,
        code: torch.Tensor,
        *,
        unique_code_batching: bool | None = None,
    ) -> torch.Tensor:
        use_unique = (
            self.unique_code_batching
            if unique_code_batching is None
            else unique_code_batching
        )
        if not use_unique or x.shape[0] <= 1:
            return self.apply_affine(
                x, self.generate_parameters(code), code=code
            )

        # Generate once per distinct hard region when collisions are common enough
        # that the hypernet savings beat unique-find + expand overhead. No
        # `.item()` sync: ``first.shape[0]`` is known after ``torch.unique``.
        hard = (code.detach() > 0.5).to(torch.int64)
        first_index, inverse = unique_code_indices(hard)
        n_unique = first_index.shape[0]
        batch = x.shape[0]
        if n_unique >= max(1, int(self.unique_code_batching_max_frac * batch)):
            # Near-unique batch: skip unique path (same hypernet work, less overhead).
            return self.apply_affine(
                x, self.generate_parameters(code), code=code
            )

        rep_params = self.generate_parameters(code[first_index])
        return self._apply_residual_from_unique(
            x, rep_params, inverse, code=code
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        unique_code_batching: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        code, _ = self.gate(x)
        output = self.forward_from_codes(
            x, code, unique_code_batching=unique_code_batching
        )
        return output, code

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
