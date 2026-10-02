"""Smoke tests: package import and public API surface."""

import residual_lo


def test_version():
    assert residual_lo.__version__


def test_public_symbols_importable():
    from residual_lo import (
        BinaryGate,
        LinearOperatorLayer,
        MLP,
        ResidualLOConv,
        ResidualLOConvBlock,
        gate_diversity_loss,
    )

    assert callable(LinearOperatorLayer)
    assert callable(BinaryGate)
    assert callable(ResidualLOConv)
    assert callable(ResidualLOConvBlock)
    assert callable(gate_diversity_loss)
    assert callable(MLP)
