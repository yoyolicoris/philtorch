import pytest
import torch

from philtorch.core import lpv_fir, lpv_allpole


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("T", [64])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_allpole_inverse(B: int, T: int, order: int):
    """Test all-pole filter inverse operation"""
    a = torch.randn(B, T, order)
    a = a / a.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)  # Ensure stability
    x = torch.randn(B, T)
    zi = torch.randn(B, order)

    # Apply all-pole filter
    y, _ = lpv_allpole(a, x, zi=zi)

    # Inverse all-pole filter
    b = torch.cat([torch.ones(B, T, 1), a], dim=-1)  # FIR coefficients from all-pole
    x_reconstructed, _ = lpv_fir(b, y, zi=zi)

    assert x.shape == x_reconstructed.shape, "Reconstructed signal shape mismatch"
    assert torch.allclose(
        x, x_reconstructed, atol=1e-6
    ), "Reconstructed signal mismatch"


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("T", [64])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_fir_inverse(B: int, T: int, order: int):
    """Test FIR filter inverse operation"""
    a = torch.randn(B, T, order)
    a = a / a.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
    g0 = torch.randn(B, T, 1)
    b = torch.cat([g0, a * g0], dim=-1)
    x = torch.randn(B, T)
    zi = torch.zeros(B, order)

    # Apply FIR filter
    y, _ = lpv_fir(b, x, zi=zi)

    # Inverse FIR filter
    x_reconstructed, _ = lpv_allpole(a, y / g0.squeeze(2), zi=zi)
    assert x.shape == x_reconstructed.shape, "Reconstructed signal shape mismatch"
    assert torch.allclose(
        x, x_reconstructed, atol=5e-6
    ), "Reconstructed signal mismatch"
