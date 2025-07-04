import pytest
import numpy as np
import torch
from scipy import signal
from typing import Tuple, Optional

from philtorch.lti import lfilter, fir
from philtorch.mat import a2companion, vandermonde


def _generate_random_filter_coeffs(
    num_order: int, den_order: int, B: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random filter coefficients"""

    # Time-invariant coefficients
    b = np.random.randn(B, num_order + 1)
    a = np.random.randn(B, den_order)
    a = a / np.abs(a).sum(axis=-1, keepdims=True)

    return b, a


def _generate_random_signal(B: int, T: int) -> np.ndarray:
    """Generate random input signal"""
    return np.random.randn(B, T)


def test_df_fir():
    """Test df2 filter with FIR coefficients"""

    B = 3
    T = 100
    num_order = 4

    b = np.random.randn(B, num_order + 1)
    x = _generate_random_signal(B, T)

    # Convert to torch tensors
    b_torch = torch.from_numpy(b)
    x_torch = torch.from_numpy(x)

    # Apply philtorch filter
    y_torch = fir(b_torch, x_torch, tranpose=False)
    # Apply scipy filter
    y_scipy = np.stack([signal.lfilter(b[i], [1.0], x[i]) for i in range(B)], axis=0)

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(
        np.abs(y_torch.numpy() - y_scipy)
    )


@pytest.mark.parametrize("include_zi", [True, False])
def test_tdf_fir(include_zi: bool):
    """Test df2 filter with FIR coefficients"""

    B = 3
    T = 100
    num_order = 4

    b = np.random.randn(B, num_order + 1)
    x = _generate_random_signal(B, T)
    if include_zi:
        # Generate random initial conditions
        zi = np.random.randn(B, num_order)
    else:
        zi = None

    # Convert to torch tensors
    b_torch = torch.from_numpy(b)
    x_torch = torch.from_numpy(x)
    if zi is not None:
        zi_torch = torch.from_numpy(zi)
    else:
        zi_torch = None

    # Apply philtorch filter
    torch_results = fir(b_torch, x_torch, zi=zi_torch, tranpose=True)
    # Apply scipy filter
    scipy_results = [
        signal.lfilter(b[i], [1.0], x[i], zi=zi[i] if zi is not None else None)
        for i in range(B)
    ]

    if include_zi:
        y_scipy, zf_scipy = zip(*scipy_results)
        y_scipy = np.stack(y_scipy, axis=0)
        zf_scipy = np.stack(zf_scipy, axis=0)
        y_torch, zf_torch = torch_results

        assert np.allclose(zf_torch.numpy(), zf_scipy), np.max(
            np.abs(zf_torch.numpy() - zf_scipy)
        )
    else:
        y_scipy = np.vstack(scipy_results)
        y_torch = torch_results

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(
        np.abs(y_torch.numpy() - y_scipy)
    )


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("T", [32, 128])
@pytest.mark.parametrize("num_order", [1, 2, 4])
@pytest.mark.parametrize("den_order", [1, 3, 5])
@pytest.mark.parametrize("form", ["df2", "tdf2", "df1", "tdf1"])
def test_time_invariant_filter(
    B: int, T: int, num_order: int, den_order: int, form: str
):
    """Test time-invariant filters against scipy.signal.lfilter"""

    # Generate test data
    b, a = _generate_random_filter_coeffs(num_order, den_order, B)
    x = _generate_random_signal(B, T)

    # Convert to torch tensors
    b_torch = torch.from_numpy(b)
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)

    # Apply philtorch filter
    y_torch = lfilter(b_torch, a_torch, x_torch, form=form)

    # Apply scipy filter
    y_scipy = np.stack(
        [signal.lfilter(b[i], [1.0] + a[i].tolist(), x[i]) for i in range(B)], axis=0
    )

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(
        np.abs(y_torch.numpy() - y_scipy)
    )


@pytest.mark.parametrize("num_order", [1, 3, 5])
@pytest.mark.parametrize("den_order", [1, 2, 4])
def test_tdf2_zi(num_order: int, den_order: int):
    B = 3
    T = 100
    # Generate test data
    b, a = _generate_random_filter_coeffs(num_order, den_order, B)
    x = _generate_random_signal(B, T)
    zi = np.random.randn(B, max(num_order, den_order))

    # Convert to torch tensors
    b_torch = torch.from_numpy(b)
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)
    zi_torch = torch.from_numpy(zi)

    # Apply philtorch filter
    y_torch, zf_torch = lfilter(b_torch, a_torch, x_torch, zi=zi_torch, form="tdf2")

    # Apply scipy filter
    y_scipy, zf_scipy = zip(
        *[signal.lfilter(b[i], [1.0] + a[i].tolist(), x[i], zi=zi[i]) for i in range(B)]
    )

    y_scipy = np.stack(y_scipy, axis=0)
    zf_scipy = np.stack(zf_scipy, axis=0)
    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(
        np.abs(y_torch.numpy() - y_scipy)
    )
    assert np.allclose(zf_torch.numpy(), zf_scipy), np.max(
        np.abs(zf_torch.numpy() - zf_scipy)
    )


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("T", [101])
@pytest.mark.parametrize(
    ("num_order", "den_order", "delayed_form"),
    [
        (1, 1, False),
        (3, 3, True),
        (4, 5, False),
        (4, 6, True),
        (3, 2, False),
        (5, 3, True),
    ],
)
@pytest.mark.parametrize("form", ["df2", "tdf2"])
@pytest.mark.parametrize("enable_L", [True, False])
@pytest.mark.parametrize("enable_V", [True, False])
def test_diag_ssm_backend(
    B: int,
    T: int,
    num_order: int,
    den_order: int,
    form: str,
    enable_L: bool,
    enable_V: bool,
    delayed_form: bool,
):
    """Test time-invariant filters against scipy.signal.lfilter"""

    # Generate test data
    # b, a = _generate_random_filter_coeffs(num_order, den_order, B)
    b = np.random.randn(B, num_order + 1)
    num_cmplx_poles = den_order // 2
    num_real_poles = den_order - 2 * num_cmplx_poles

    cmplx_poles = np.random.rand(num_cmplx_poles) ** 0.5 * np.exp(
        1j * np.random.rand(num_cmplx_poles) * 2 * np.pi
    )
    real_poles = np.random.rand(num_real_poles) ** 0.5 + 0j
    roots = np.concatenate([cmplx_poles, cmplx_poles.conj(), real_poles])
    a = np.polynomial.Polynomial.fromroots(roots).coef.real[-2::-1].copy()

    x = _generate_random_signal(B, T)

    # Convert to torch tensors
    b_torch = torch.from_numpy(b)
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)
    roots_torch = torch.from_numpy(roots)
    L = roots_torch if enable_L else None
    V = vandermonde(roots_torch) if enable_V else None

    if form == "tdf2" and enable_V:
        Vinv = V.T
        V = None
    else:
        Vinv = None

    # Apply philtorch filter
    y_torch = lfilter(
        b_torch,
        a_torch,
        x_torch,
        form=form,
        backend="diag_ssm",
        L=L,
        V=V,
        Vinv=Vinv,
        delayed_form=delayed_form,
    )

    # Apply scipy filter
    y_scipy = np.stack(
        [signal.lfilter(b[i], [1.0] + a.tolist(), x[i]) for i in range(B)], axis=0
    )

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(
        np.abs(y_torch.numpy() - y_scipy)
    )
