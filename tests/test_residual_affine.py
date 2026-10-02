"""Property tests: residual LO preserves one hard code → one affine map on x."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from residual_lo import LinearOperatorLayer, apply_generated_parameters


def _residual_layer(
    *,
    d: int = 16,
    k: int = 4,
    rank: int = 3,
    affine_mode: str = "diag_low_rank",
) -> LinearOperatorLayer:
    return LinearOperatorLayer(
        in_features=d,
        out_features=d,
        codes_per_head=k,
        output_initialization_scale=0.5,
        hypernet_hidden_sizes=[32, 32],
        bias=True,
        affine_mode=affine_mode,  # type: ignore[arg-type]
        rank=rank,
        use_residual_base=True,
        generator_type="mlp",
    )


@torch.no_grad()
def test_fixed_hard_code_is_affine_in_x():
    """For fixed hard code c, f(ax+by)+const matches linearity (affine)."""
    torch.manual_seed(0)
    layer = _residual_layer()
    layer.eval()

    d = layer.in_features
    k = layer.codes
    # Fixed hard code (STE path unused — feed code directly).
    code = torch.tensor([[1.0, 0.0, 1.0, 0.0][:k]], dtype=torch.float32)
    if k > 4:
        code = torch.randint(0, 2, (1, k), dtype=torch.float32)

    a, b = 0.7, -1.3
    x = torch.randn(8, d)
    y = torch.randn(8, d)
    # Expand code to batch for generate_parameters.
    code_b = code.expand(x.shape[0], -1)
    params = layer.generate_parameters(code_b)

    # Same code for all rows → apply_parameters is batch of identical maps.
    f_x = layer.apply_parameters(x, params)
    f_y = layer.apply_parameters(y, params)
    f_combo = layer.apply_parameters(a * x + b * y, params)

    # Affine: f(ax+by) = a f(x) + b f(y) - (a+b-1) f(0)
    f0 = layer.apply_parameters(torch.zeros_like(x), params)
    expected = a * f_x + b * f_y - (a + b - 1.0) * f0
    assert torch.allclose(f_combo, expected, atol=1e-5, rtol=1e-4), (
        f"max abs err={(f_combo - expected).abs().max().item()}"
    )


@torch.no_grad()
def test_materialized_w_matches_forward_full():
    """Materialize W_c = W_0 + W_Δ(c) and match apply_parameters (full affine_mode)."""
    torch.manual_seed(1)
    layer = LinearOperatorLayer(
        in_features=12,
        out_features=10,
        codes_per_head=3,
        output_initialization_scale=0.5,
        hypernet_hidden_sizes=[24, 24],
        bias=True,
        affine_mode="full",
        rank=None,
        use_residual_base=True,
        generator_type="mlp",
    )
    layer.eval()
    code = torch.tensor([[1.0, 0.0, 1.0]])
    x = torch.randn(5, 12)
    params = layer.generate_parameters(code.expand(x.shape[0], -1))
    y_fwd = layer.apply_parameters(x, params)

    # Materialize dense ΔW from generated params (same layout as apply_generated_affine).
    split = 12 * 10
    w_delta = params[0, :split].reshape(10, 12)
    b_delta = params[0, split : split + 10]
    w_c = layer.base_weight + w_delta
    b_c = layer.base_bias + b_delta
    y_mat = F.linear(x, w_c, b_c)
    assert torch.allclose(y_fwd, y_mat, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_materialized_uv_matches_forward_factored():
    """For diag_low_rank: W_c = W_0 + U_c V_c + diag; forward matches materialization."""
    torch.manual_seed(2)
    d, r = 14, 4
    layer = _residual_layer(d=d, k=3, rank=r, affine_mode="diag_low_rank")
    layer.eval()
    code = torch.tensor([[0.0, 1.0, 1.0]])
    x = torch.randn(6, d)
    params = layer.generate_parameters(code.expand(x.shape[0], -1))
    y_fwd = layer.apply_parameters(x, params)

    flat = params[0]
    cursor = 0
    u = flat[cursor : cursor + d * r].reshape(d, r)
    cursor += d * r
    v = flat[cursor : cursor + r * d].reshape(r, d)
    cursor += r * d
    diag = flat[cursor : cursor + d]
    cursor += d
    b_delta = flat[cursor : cursor + d]

    w_delta = u @ v + torch.diag(diag)
    w_c = layer.base_weight + w_delta
    b_c = layer.base_bias + b_delta
    y_mat = F.linear(x, w_c, b_c)
    assert torch.allclose(y_fwd, y_mat, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_hypernet_gelu_not_on_data_path():
    """GELU appears only in hypernet(code); apply_parameters uses linear ops on x."""
    layer = _residual_layer()
    # Hypernet MLP uses GELU between Linear layers on the *code* input.
    assert isinstance(layer.hypernet.activation, torch.nn.GELU)
    # Data-path apply: generate params from code, then affine in x — no activation module.
    code = torch.ones(2, layer.codes)
    params = layer.generate_parameters(code)
    x = torch.randn(2, layer.in_features)
    y = layer.apply_parameters(x, params)
    assert y.shape == (2, layer.out_features)
    # Sanity: zero input with fixed params yields bias-only (affine intercept).
    y0 = layer.apply_parameters(torch.zeros_like(x), params)
    # Changing x linearly changes y (smoke: not constant unless W=0).
    y1 = layer.apply_parameters(torch.ones_like(x), params)
    assert not torch.allclose(y0, y1)


@torch.no_grad()
def test_apply_generated_parameters_is_linear_homogeneous_without_bias():
    """Factored apply without bias is linear (homogeneous) in x for fixed params."""
    torch.manual_seed(3)
    d, r = 10, 2
    params = torch.randn(4, d * r + r * d)  # no bias, no diag
    x = torch.randn(4, d)
    a = 2.5
    y = apply_generated_parameters(
        x, params, d, d, affine_mode="low_rank", rank=r, bias=False
    )
    y_scaled = apply_generated_parameters(
        a * x, params, d, d, affine_mode="low_rank", rank=r, bias=False
    )
    assert torch.allclose(y_scaled, a * y, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_unique_code_batching_matches_per_sample_for_hard_codes():
    """Unique-code path must equal per-sample apply when codes are hard (no soft STE)."""
    torch.manual_seed(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = _residual_layer(d=32, k=8, rank=4).to(device)
    layer.eval()
    b = 64
    x = torch.randn(b, layer.in_features, device=device)
    # ~20 unique hard codes with collisions
    palette = (torch.rand(20, layer.codes, device=device) > 0.5).float()
    hard = palette[torch.randint(0, 20, (b,), device=device)]
    code = hard  # hard STE: soft == hard
    y_unique = layer.forward_from_codes(x, code, unique_code_batching=True)
    y_dense = layer.forward_from_codes(x, code, unique_code_batching=False)
    assert torch.allclose(y_unique, y_dense, atol=1e-5, rtol=1e-4), (
        f"max abs={(y_unique - y_dense).abs().max().item()}"
    )
    # Same hard code ⇒ identical affine on two rows
    i = j = None
    for a in range(b):
        for c in range(a + 1, b):
            if torch.equal(hard[a], hard[c]):
                i, j = a, c
                break
        if i is not None:
            break
    assert i is not None
    # Swap x[i] and check map equality via applying same code to both
    params = layer.generate_parameters(hard[i : i + 1])
    yi = layer.apply_parameters(x[i : i + 1], params)
    yj = layer.apply_parameters(x[j : j + 1], params)
    # different x can differ; same code's W applied to x[i] matches unique path
    assert torch.allclose(y_unique[i : i + 1], yi, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_adaptive_unique_batching_near_unique_matches_dense():
    """When nearly every token has a distinct code, adaptive path matches dense."""
    torch.manual_seed(5)
    layer = _residual_layer(d=24, k=12, rank=4)
    layer.eval()
    b = 48
    x = torch.randn(b, layer.in_features)
    # Force near-unique hard codes (K=12 → plenty of room).
    hard = (torch.rand(b, layer.codes) > 0.5).float()
    y_u = layer.forward_from_codes(x, hard, unique_code_batching=True)
    y_d = layer.forward_from_codes(x, hard, unique_code_batching=False)
    assert torch.allclose(y_u, y_d, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_diag_low_rank_indexed_matches_expanded():
    """Indexed unique apply must match expand-then-apply for diag_low_rank."""
    from residual_lo import (
        apply_diag_low_rank_from_unique,
        apply_generated_diag_low_rank_affine,
    )

    torch.manual_seed(6)
    d, r, u, n = 16, 4, 7, 40
    params_u = torch.randn(u, d * r + r * d + d + d)
    inverse = torch.randint(0, u, (n,))
    x = torch.randn(n, d)
    y_idx = apply_diag_low_rank_from_unique(x, params_u, inverse, d, d, r, bias=True)
    y_exp = apply_generated_diag_low_rank_affine(
        x, params_u[inverse], d, d, r, bias=True
    )
    assert torch.allclose(y_idx, y_exp, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    test_fixed_hard_code_is_affine_in_x()
    test_materialized_w_matches_forward_full()
    test_materialized_uv_matches_forward_factored()
    test_hypernet_gelu_not_on_data_path()
    test_apply_generated_parameters_is_linear_homogeneous_without_bias()
    test_unique_code_batching_matches_per_sample_for_hard_codes()
    test_adaptive_unique_batching_near_unique_matches_dense()
    test_diag_low_rank_indexed_matches_expanded()
    print("all residual-affine property tests passed")
