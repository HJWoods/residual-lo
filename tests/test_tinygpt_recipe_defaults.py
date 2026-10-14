"""Assert LinearOperatorLayer defaults match the TinyGPT FineWeb/SmolTalk recipe."""

from residual_lo import LinearOperatorLayer


def test_mainline_defaults_match_tinygpt_recipe():
    layer = LinearOperatorLayer(
        in_features=32,
        out_features=32,
        codes_per_head=8,
        output_initialization_scale=0.1,
        rank=4,
        use_residual_base=True,
    )
    assert layer.affine_mode == "diag_low_rank"
    assert layer.generator_type == "residual_mlp"
    assert layer.use_residual_base is True
    assert layer.base_weight is not None
    # Data-path purity: no activation module wrapping the apply path.
    assert not hasattr(layer, "act")
    assert not hasattr(layer, "bn")
