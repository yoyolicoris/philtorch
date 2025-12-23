import pytest
import torch
from scipy.interpolate import CubicSpline

from philtorch.lti.interp import cubic_spline


@pytest.mark.parametrize("n", [20, 50, 100])
@pytest.mark.parametrize("m", [2, 3, 7])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_cubic_spline(n, m, device):
    x = torch.randn(n, device=device).double()
    x = torch.cat([x, x.flip(0)[1:]])  # make it periodic

    y_philtorch = cubic_spline(x.unsqueeze(0), m, parallel_form=False).squeeze().cpu()

    t = torch.arange(len(x)).double()
    cs = CubicSpline(t.cpu().numpy(), x.cpu().numpy(), bc_type="periodic")
    t_new = torch.arange(0, len(x) - 1 + 1 / m / 2, 1 / m, dtype=torch.double)
    y_scipy = cs(t_new.cpu().numpy())

    assert torch.allclose(
        y_philtorch, torch.from_numpy(y_scipy)
    ), f"Max abs diff: {(y_philtorch - torch.from_numpy(y_scipy)).abs().max()}"
