"""Tests for DiscreteBottleneckConv: Discrete Bottleneck Layer gate/apply invariants in conv form."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from residual_lo import (
    DiscreteBottleneckConv,
    DiscreteBottleneckConvBlock,
    apply_generated_operator,
    apply_generated_operator_nchw,
)


def _lo_conv(**kwargs) -> DiscreteBottleneckConv:
    defaults = dict(
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        codes_per_head=4,
        rank=2,
        affine_mode="diag_low_rank",
        generator_type="mlp",
        hypernet_hidden_sizes=[32, 32],
        gate_mode="pool",
        use_residual_base=True,
        unique_code_batching=True,
        output_initialization_scale=0.5,
    )
    defaults.update(kwargs)
    return DiscreteBottleneckConv(**defaults)


@torch.no_grad()
def test_forward_shapes_pool_and_spatial():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 7, 7)
    pool = _lo_conv(gate_mode="pool")
    y, code = pool(x)
    assert y.shape == (2, 8, 7, 7)
    assert code.shape == (2, 4)

    spatial = _lo_conv(gate_mode="spatial")
    y2, code2 = spatial(x)
    assert y2.shape == (2, 8, 7, 7)
    assert code2.shape == (2 * 7 * 7, 4)


@torch.no_grad()
def test_nchw_apply_matches_flat_apply():
    torch.manual_seed(1)
    b, c_in, h, w, c_out, r = 3, 6, 4, 5, 5, 2
    x = torch.randn(b, c_in, h, w)
    # Fake diag_low_rank params: U + V + diag + bias
    g = c_out * r + r * c_in + min(c_in, c_out) + c_out
    params = torch.randn(b, g) * 0.1
    out_nchw = apply_generated_operator_nchw(
        x, params, c_in, c_out, affine_mode="diag_low_rank", rank=r, bias=True
    )
    flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c_in)
    params_exp = params.unsqueeze(1).expand(b, h * w, g).reshape(b * h * w, g)
    out_flat = apply_generated_operator(
        flat, params_exp, c_in, c_out, affine_mode="diag_low_rank", rank=r, bias=True
    )
    out_ref = out_flat.reshape(b, h, w, c_out).permute(0, 3, 1, 2)
    assert torch.allclose(out_nchw, out_ref, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_fixed_code_residual_is_affine_in_channels():
    """For fixed code, channel residual Δ is affine in the channel vector."""
    torch.manual_seed(2)
    layer = _lo_conv(in_channels=6, out_channels=6, gate_mode="pool")
    layer.eval()
    code = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    params = layer.generate_parameters(code)

    a, b = 0.6, -0.4
    # Single-pixel maps so spatial geometry is irrelevant.
    x = torch.randn(1, 6, 1, 1)
    y = torch.randn(1, 6, 1, 1)
    fx = layer.apply_residual(x, params)
    fy = layer.apply_residual(y, params)
    fcombo = layer.apply_residual(a * x + b * y, params)
    f0 = layer.apply_residual(torch.zeros_like(x), params)
    expected = a * fx + b * fy - (a + b - 1.0) * f0
    assert torch.allclose(fcombo, expected, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_forward_equals_base_plus_residual():
    torch.manual_seed(3)
    layer = _lo_conv(in_channels=4, out_channels=5, gate_mode="pool")
    layer.eval()
    x = torch.randn(2, 4, 6, 6)
    feats = layer._gate_features(x)
    code, _ = layer.gate(feats)
    y = layer.forward_from_codes(x, code)
    params = layer._generate_for_codes(code)
    residual = layer.apply_residual(x, params)
    base = layer.base_conv(x)
    assert torch.allclose(y, base + residual, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_same_code_same_delta_across_batch_rows():
    torch.manual_seed(4)
    layer = _lo_conv(gate_mode="pool", unique_code_batching=True)
    layer.eval()
    code = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    x = torch.randn(2, 8, 3, 3)
    y = layer.forward_from_codes(x, code)
    # Swap inputs, keep codes → Δ identical so swapping x swaps y - base.
    base = layer.base_conv(x)
    delta = y - base
    y_swap = layer.forward_from_codes(x.flip(0), code)
    delta_swap = y_swap - layer.base_conv(x.flip(0))
    assert torch.allclose(delta.flip(0), delta_swap, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_strided_odd_map_matches_base_spatial():
    """Odd H/W + stride=2: y must match Conv2d spatial size (Bugbot medium)."""
    torch.manual_seed(5)
    layer = _lo_conv(stride=2, padding=1, kernel_size=3)
    layer.eval()
    x = torch.randn(2, 8, 7, 7)  # odd → Conv2d out 4×4; naive pool(stride) is 3×3
    y, _ = layer(x)
    base = layer.base_conv(x)
    assert y.shape == base.shape == (2, 8, 4, 4)


@torch.no_grad()
def test_nonzero_asymmetric_padding_matches_base_spatial():
    """padding != k//2 changes H×W; residual must still add onto base."""
    torch.manual_seed(6)
    layer = _lo_conv(stride=1, padding=0, kernel_size=3)  # 9×9 → 7×7
    layer.eval()
    x = torch.randn(1, 8, 9, 9)
    y, _ = layer(x)
    base = layer.base_conv(x)
    assert y.shape == base.shape == (1, 8, 7, 7)


@torch.no_grad()
def test_block_no_bn_or_activation_after_lo():
    """DiscreteBottleneckConvBlock must not wrap LO with BN/GELU (Mori constraint)."""
    torch.manual_seed(7)
    block = DiscreteBottleneckConvBlock(
        3,
        8,
        codes=4,
        rank=2,
        hypernet_width=32,
        hypernet_depth=2,
        pool=True,
    )
    assert not hasattr(block, "bn")
    assert not hasattr(block, "act")
    assert block.pool is not None
    x = torch.randn(2, 3, 16, 16)
    y, code = block(x)
    assert y.shape == (2, 8, 8, 8)  # MaxPool2d(2)
    assert code.shape == (2, 4)
    # Block output equals pooled LO output (no extra nonlinearity).
    y_lo, _ = block.lo(x)
    assert torch.allclose(y, block.pool(y_lo), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    test_forward_shapes_pool_and_spatial()
    test_nchw_apply_matches_flat_apply()
    test_fixed_code_residual_is_affine_in_channels()
    test_forward_equals_base_plus_residual()
    test_same_code_same_delta_across_batch_rows()
    test_strided_odd_map_matches_base_spatial()
    test_nonzero_asymmetric_padding_matches_base_spatial()
    test_block_no_bn_or_activation_after_lo()
    print("all discrete-bottleneck-conv tests passed")
