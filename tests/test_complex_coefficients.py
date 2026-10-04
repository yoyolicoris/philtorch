"""Filters with complex coefficients, in every form, compute what SciPy does.

The transposed forms used to filter with the conjugated coefficients, which
only matters once the coefficients are complex.
"""

import numpy as np
import pytest
import torch
from scipy import signal

import philtorch.lpv as lpv
import philtorch.lti as lti

N = 24


@pytest.fixture
def coeffs():
    rng = np.random.default_rng(0)
    b = rng.standard_normal(3) * 0.3 + 1j * rng.standard_normal(3) * 0.3
    a = np.array([-0.5 + 0.1j, 0.2 - 0.05j])
    x = rng.standard_normal((2, N)) + 1j * rng.standard_normal((2, N))
    return b, a, x


def _lpv_constant(c):
    """Repeat constant coefficients over the batch and time steps."""
    return torch.from_numpy(np.broadcast_to(c, (2, N, c.size)).copy())


def _holomorphic(f, *args, h=1e-6):
    """Return whether f has the same derivative along d and along i d.

    A filter without conjugation is holomorphic in its coefficients and input,
    so stepping them by h d or by i h d gives the same complex derivative.
    """
    gen = torch.Generator().manual_seed(0)
    d = [torch.randn(arg.shape, dtype=arg.dtype, generator=gen) for arg in args]

    def step(s):
        return f(*[arg + s * di for arg, di in zip(args, d)])

    along_real = (step(h) - step(-h)) / (2 * h)
    along_imag = (step(1j * h) - step(-1j * h)) / (2j * h)
    return torch.allclose(along_real, along_imag, atol=1e-6)


@pytest.mark.parametrize(
    ("form", "backend"),
    [
        ("df2", "ssm"),
        ("tdf2", "ssm"),
        ("df1", "ssm"),
        ("tdf1", "ssm"),
        ("df2", "diag_ssm"),
        ("tdf2", "diag_ssm"),
    ],
)
def test_lti_lfilter(coeffs, form, backend):
    b, a, x = coeffs
    y = lti.lfilter(
        torch.from_numpy(np.stack([b, b])),
        torch.from_numpy(a),
        torch.from_numpy(x),
        form=form,
        backend=backend,
    )
    assert np.allclose(y.numpy(), signal.lfilter(b, np.r_[1, a], x))


def test_lti_lfilter_zi(coeffs):
    b, a, x = coeffs
    zi = np.ones((2, 2)) * (0.3 - 0.2j)
    y, zf = lti.lfilter(
        torch.from_numpy(b), torch.from_numpy(a), torch.from_numpy(x), zi=torch.from_numpy(zi)
    )
    y_ref, zf_ref = signal.lfilter(b, np.r_[1, a], x, zi=zi)
    assert np.allclose(y.numpy(), y_ref)
    assert np.allclose(zf.numpy(), zf_ref)
    assert np.allclose(
        lti.lfilter_zi(torch.from_numpy(a), torch.from_numpy(b)).numpy(),
        signal.lfilter_zi(b, np.r_[1, a]),
    )


def test_lti_filtfilt(coeffs):
    b, a, x = coeffs
    y = lti.filtfilt(
        torch.from_numpy(b), torch.from_numpy(a), torch.from_numpy(x), padmode="reflect"
    )
    assert np.allclose(y.numpy(), signal.filtfilt(b, np.r_[1, a], x, padtype="even"))


@pytest.mark.parametrize("transpose", [True, False])
def test_lti_fir(coeffs, transpose):
    b, _, x = coeffs
    y = lti.fir(torch.from_numpy(np.stack([b, b])), torch.from_numpy(x), transpose=transpose)
    assert np.allclose(y.numpy(), signal.lfilter(b, [1], x))


@pytest.mark.parametrize(
    ("form", "backend"),
    [
        ("df2", "ssm"),
        ("tdf2", "ssm"),
        ("df1", "ssm"),
        ("tdf1", "ssm"),
        ("df2", "torchlpc"),
        ("df1", "torchlpc"),
        ("tdf1", "torchlpc"),
    ],
)
def test_lpv_lfilter_constant(coeffs, form, backend):
    b, a, x = coeffs
    y = lpv.lfilter(
        _lpv_constant(b), _lpv_constant(a), torch.from_numpy(x), form=form, backend=backend
    )
    assert np.allclose(y.numpy(), signal.lfilter(b, np.r_[1, a], x))


@pytest.mark.parametrize("transpose", [True, False])
def test_lpv_fir_and_allpole_constant(coeffs, transpose):
    b, a, x = coeffs
    y = lpv.fir(_lpv_constant(b), torch.from_numpy(x), transpose=transpose)
    assert np.allclose(y.numpy(), signal.lfilter(b, [1], x))
    y = lpv.allpole(_lpv_constant(a), torch.from_numpy(x), transpose=transpose)
    assert np.allclose(y.numpy(), signal.lfilter([1], np.r_[1, a], x))


# Time-varying transposed forms are different filters from the direct forms,
# so check them for conjugation directly.
@pytest.mark.parametrize(
    ("form", "backend"),
    [
        ("df2", "ssm"),
        ("tdf2", "ssm"),
        ("df1", "ssm"),
        ("tdf1", "ssm"),
        ("df2", "torchlpc"),
        ("df1", "torchlpc"),
        ("tdf1", "torchlpc"),
    ],
)
def test_lpv_lfilter_time_varying_is_holomorphic(form, backend):
    gen = torch.Generator().manual_seed(0)
    b = torch.randn(2, N, 3, dtype=torch.complex128, generator=gen) * 0.3
    a = torch.randn(2, N, 2, dtype=torch.complex128, generator=gen) * 0.2
    x = torch.randn(2, N, dtype=torch.complex128, generator=gen)
    assert _holomorphic(lambda b, a, x: lpv.lfilter(b, a, x, form=form, backend=backend), b, a, x)


@pytest.mark.parametrize("transpose", [True, False])
def test_lpv_fir_and_allpole_time_varying_are_holomorphic(transpose):
    gen = torch.Generator().manual_seed(0)
    b = torch.randn(2, N, 3, dtype=torch.complex128, generator=gen) * 0.3
    a = torch.randn(2, N, 2, dtype=torch.complex128, generator=gen) * 0.2
    x = torch.randn(2, N, dtype=torch.complex128, generator=gen)
    assert _holomorphic(lambda b, x: lpv.fir(b, x, transpose=transpose), b, x)
    assert _holomorphic(lambda a, x: lpv.allpole(a, x, transpose=transpose), a, x)
