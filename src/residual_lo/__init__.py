"""Residual / discrete linear-operator layers.

Public surface for the LO bottleneck (BinaryGate + hypernet-generated affine)
and ResidualLOConv. Transformer / TinyGPT code lives in DiscreteInterp and
imports from here — do not reimplement layer logic there.
"""

from residual_lo.linear_operator import (
    AffineMode,
    BinaryGate,
    GeneratorType,
    LinearOperatorLayer,
    ResidualBlock,
    ResidualHypernet,
    apply_diag_low_rank_from_unique,
    apply_generated_affine,
    apply_generated_diag_low_rank_affine,
    apply_generated_low_rank_affine,
    apply_generated_operator,
    apply_generated_parameters,
    gate_diversity_loss,
    generated_parameter_count,
    pack_binary_codes,
    unique_code_indices,
)
from residual_lo.mlp import MLP
from residual_lo.residual_lo_conv import (
    GateMode,
    ResidualLOConv,
    ResidualLOConvBlock,
    apply_generated_operator_nchw,
)

__version__ = "0.1.0"

__all__ = [
    "AffineMode",
    "BinaryGate",
    "GateMode",
    "GeneratorType",
    "LinearOperatorLayer",
    "MLP",
    "ResidualBlock",
    "ResidualHypernet",
    "ResidualLOConv",
    "ResidualLOConvBlock",
    "apply_diag_low_rank_from_unique",
    "apply_generated_affine",
    "apply_generated_diag_low_rank_affine",
    "apply_generated_low_rank_affine",
    "apply_generated_operator",
    "apply_generated_operator_nchw",
    "apply_generated_parameters",
    "gate_diversity_loss",
    "generated_parameter_count",
    "pack_binary_codes",
    "unique_code_indices",
    "__version__",
]
