import pytest
import numpy as np
import torch
from typing import Tuple

from philtorch.lpv import lfilter


def _generate_time_varying_coeffs(
    B: int,
    T: int,
    num_order: int,
    den_order: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate time-varying filter coefficients"""
    # Generate smooth time-varying coefficients
    b = torch.randn(B, T, num_order + 1)
    a = torch.randn(B, T, den_order)

    # Ensure stability by keeping denominator coefficients small
    # a = torch.clamp(a, -0.5, 0.5)
    a = a / a.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)

    return b, a


def _generate_test_signal(
    B: int, T: int, signal_type: str = "white_noise"
) -> torch.Tensor:
    """Generate different types of test signals"""
    if signal_type == "white_noise":
        return torch.randn(B, T)
    elif signal_type == "impulse":
        x = torch.zeros(B, T)
        x[:, 0] = 1.0
        return x
    elif signal_type == "step":
        return torch.ones(B, T)
    elif signal_type == "sine":
        t = torch.linspace(0, 4 * np.pi, T).unsqueeze(0).expand(B, -1)
        return torch.sin(t)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


@pytest.mark.parametrize("B", [1, 8, 16])
@pytest.mark.parametrize("T", [32, 128])
@pytest.mark.parametrize("num_order", [1, 2, 4])
@pytest.mark.parametrize("den_order", [1, 3, 5])
@pytest.mark.parametrize("form", ["df1", "df2"])
def test_time_varying_basic_functionality(
    B: int, T: int, num_order: int, den_order: int, form: str
):
    """Test basic functionality with time-varying coefficients"""
    b, a = _generate_time_varying_coeffs(B, T, num_order, den_order)
    x = _generate_test_signal(B, T, "white_noise")

    y = lfilter(b, a, x, form=form)

    if isinstance(y, tuple):
        y, zf = y
        assert zf.shape[0] == B  # Check final state shape

    assert y.shape == (B, T)
    assert not torch.isnan(y).any(), f"NaN values found in output for form {form}"
    assert torch.isfinite(y).all(), f"Non-finite values found in output for form {form}"


@pytest.mark.parametrize("signal_type", ["impulse", "step", "sine", "white_noise"])
def test_different_input_signals(signal_type: str):
    """Test filter response to different input signal types"""
    B, T = 2, 100
    b, a = _generate_time_varying_coeffs(B, T, 3, 2)
    x = _generate_test_signal(B, T, signal_type)

    y = lfilter(b, a, x, form="df1")

    if isinstance(y, tuple):
        y = y[0]

    assert y.shape == (B, T)
    assert not torch.isnan(y).any()
    assert torch.isfinite(y).all()

    # For impulse response, check that it's not all zeros
    if signal_type == "impulse":
        assert not torch.allclose(y, torch.zeros_like(y))


def test_linearity_property():
    """Test linearity: filter(a*x1 + b*x2) = a*filter(x1) + b*filter(x2)"""
    B, T = 2, 50
    b, a = _generate_time_varying_coeffs(B, T, 2, 1)

    x1 = _generate_test_signal(B, T, "white_noise")
    x2 = _generate_test_signal(B, T, "white_noise")

    alpha, beta = 0.7, 1.3
    x_combined = alpha * x1 + beta * x2

    y1 = lfilter(b, a, x1, form="df1")
    y2 = lfilter(b, a, x2, form="df1")
    y_combined = lfilter(b, a, x_combined, form="df1")

    # Extract tensor from tuple if necessary
    if isinstance(y1, tuple):
        y1 = y1[0]
    if isinstance(y2, tuple):
        y2 = y2[0]
    if isinstance(y_combined, tuple):
        y_combined = y_combined[0]

    y_expected = alpha * y1 + beta * y2

    # Check linearity within reasonable tolerance
    assert torch.allclose(y_combined, y_expected, atol=1e-6)


def test_zero_input_zero_output():
    """Test that zero input produces zero output (assuming zero initial conditions)"""
    B, T = 2, 50
    b, a = _generate_time_varying_coeffs(B, T, 2, 1)
    x = torch.zeros(B, T)

    y = lfilter(b, a, x, form="df1")

    if isinstance(y, tuple):
        y = y[0]

    assert torch.allclose(y, torch.zeros_like(y))


def test_batch_independence():
    """Test that different batches are processed independently"""
    B, T = 3, 50

    # Create different coefficients for each batch
    b = torch.randn(B, T, 3) * 0.1
    a = torch.randn(B, T, 2) * 0.1

    # Create different inputs for each batch
    x = torch.randn(B, T)

    # Process all batches together
    y_batch = lfilter(b, a, x, form="df1")

    if isinstance(y_batch, tuple):
        y_batch = y_batch[0]

    # Process each batch individually
    y_individual = []
    for i in range(B):
        y_i = lfilter(b[i : i + 1], a[i : i + 1], x[i : i + 1], form="df1")
        if isinstance(y_i, tuple):
            y_i = y_i[0]
        y_individual.append(y_i)

    y_individual = torch.cat(y_individual, dim=0)

    # Results should be identical
    assert torch.allclose(y_batch, y_individual)


def test_initial_conditions_df2():
    """Test filter with initial conditions in DF2 form"""
    B, T = 2, 50
    order = 3
    b, a = _generate_time_varying_coeffs(B, T, order, order)
    x = _generate_test_signal(B, T, "white_noise")

    # Test with random initial conditions
    zi = torch.randn(B, order) * 0.1

    y, zf = lfilter(b, a, x, zi=zi, form="df2")

    print(y.shape, zf.shape, b.shape, a.shape, x.shape)
    assert y.shape == (B, T)
    assert zf.shape == (B, order)
    assert not torch.isnan(y).any()
    assert not torch.isnan(zf).any()
    assert torch.isfinite(y).all()
    assert torch.isfinite(zf).all()


def test_performance_large_signals():
    """Test performance with larger signals"""
    B, T = 8, 1000
    b, a = _generate_time_varying_coeffs(B, T, 4, 3)
    x = _generate_test_signal(B, T, "white_noise")

    # This should complete without errors
    y = lfilter(b, a, x, form="df1")

    if isinstance(y, tuple):
        y = y[0]

    assert y.shape == (B, T)
    assert not torch.isnan(y).any()
    assert torch.isfinite(y).all()
