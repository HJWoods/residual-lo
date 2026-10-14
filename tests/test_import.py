"""Smoke tests: package import and public API surface."""

import residual_lo


def test_version():
    assert residual_lo.__version__


def test_public_symbols_importable():
    from residual_lo import (
        BinaryGate,
        DiscreteBottleneckConv,
        DiscreteBottleneckConvBlock,
        DiscreteBottleneckLayer,
        MLP,
        gate_diversity_loss,
    )

    assert callable(DiscreteBottleneckLayer)
    assert callable(BinaryGate)
    assert callable(DiscreteBottleneckConv)
    assert callable(DiscreteBottleneckConvBlock)
    assert callable(gate_diversity_loss)
    assert callable(MLP)


def test_legacy_names_removed():
    import residual_lo as pkg

    for name in ("LinearOperatorLayer", "ResidualLOConv", "ResidualLOConvBlock"):
        assert not hasattr(pkg, name), f"legacy alias still exported: {name}"
