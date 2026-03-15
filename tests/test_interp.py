import pytest
import torch
from scipy.interpolate import CubicSpline
from scipy.signal import cspline1d, cspline1d_eval

from philtorch.lti.interp import cubic_spline, cspline


@pytest.mark.parametrize("n", [20, 100])
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
@pytest.mark.parametrize("pfe", [True, False])
def test_cubic_spline(n, m, device, pfe):
    x = torch.randn(n, device=device).double()
    x = torch.cat([x, x.flip(0)[1:]])  # make it periodic

    y_philtorch = cubic_spline(x.unsqueeze(0), m, parallel_form=pfe).squeeze().cpu()

    t = torch.arange(len(x)).double()
    # cs = CubicSpline(t.cpu().numpy(), x.cpu().numpy(), bc_type="periodic")
    t_new = torch.arange(0, len(x) - 1 + 1 / m / 2, 1 / m, dtype=torch.double)
    # y_scipy = cs(t_new.cpu().numpy())
    y_scipy = cspline1d_eval(cspline1d(x.cpu().numpy()), t_new.cpu().numpy())

    assert torch.allclose(
        y_philtorch, torch.from_numpy(y_scipy)
    ), f"Max abs diff: {(y_philtorch - torch.from_numpy(y_scipy)).abs().max()}"


@pytest.mark.parametrize("n", [20])
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
@pytest.mark.parametrize("pfe", [True, False])
def test_cspline(n, device, pfe):
    # x = torch.randn(n, device=device).double()
    x = torch.tensor([0.0, 1.0] * (n // 2), device=device).double()  # make it periodic
    y_philtorch = cspline(x.unsqueeze(0), parallel_form=pfe).squeeze().cpu()
    y_scipy = cspline1d(x.cpu().numpy())

    print("Philtorch:", y_philtorch)
    print("SciPy:", y_scipy)
    print(x)
    assert torch.allclose(
        y_philtorch, torch.from_numpy(y_scipy)
    ), f"Max abs diff: {(y_philtorch - torch.from_numpy(y_scipy)).abs().max()}"
