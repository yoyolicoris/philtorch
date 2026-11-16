import numpy as np
import pytest
import torch
from scipy import signal

from philtorch.lti import pfefilt, sos2pfe

from .test_lti_sosfilt import _random_sos

DOUBLE_REAL = [1.0, 0.5, -0.3, -1.0, 0.25]  # (1 - 0.5 z^-1)^2
OTHER_REAL = [0.7, 0.1, 0.2, -0.3, 0.02]  # poles 0.2 and 0.1
PAIR = np.delete(signal.butter(2, 0.3, output="sos"), 3, 1)[0]


def _expected(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """SciPy's cascade, one filter per signal when sos is batched."""
    full = np.insert(sos, 3, 1.0, axis=-1)
    if sos.ndim == 2:
        return signal.sosfilt(full, x)
    return np.stack([signal.sosfilt(s, xi) for s, xi in zip(full, x)])


def _check(sos: np.ndarray, x: np.ndarray):
    r, p, g, batch_index, power = sos2pfe(torch.from_numpy(sos))
    y = pfefilt(r, p, g, torch.from_numpy(x), batch_index=batch_index, power=power)
    assert y.dtype == torch.from_numpy(x).dtype
    torch.testing.assert_close(y, torch.from_numpy(_expected(sos, x)), rtol=1e-7, atol=1e-9)


def _random_real_sos() -> np.ndarray:
    np.random.seed(0)
    sos = _random_sos(K=4)
    sos[1, 3:] = [-0.8, 0.15]  # real poles 0.5 and 0.3
    return sos


@pytest.mark.parametrize(
    "sos",
    [
        _random_real_sos(),
        np.delete(signal.butter(5, 0.2, output="sos"), 3, 1),  # a pole at 0
        np.stack([PAIR, PAIR, PAIR]),
        np.stack([DOUBLE_REAL, DOUBLE_REAL]),  # a real pole of multiplicity 4
        np.array([[1.0, 2.0, 3.0, 0.0, 0.0]]),  # FIR: a double pole at 0
    ],
    ids=["simple", "odd-order", "repeated-pair", "repeated-real", "fir"],
)
def test_pfefilt_matches_scipy(sos):
    _check(sos, np.random.default_rng(0).standard_normal(64))


def test_pfefilt_complex_filter_and_input():
    np.random.seed(1)
    sos = _random_sos(K=3) + 0.2j * _random_sos(K=3)
    x = np.random.randn(64) + 1j * np.random.randn(64)
    _check(sos, x)
    # a complex filter with a real input
    y = pfefilt(*sos2pfe(torch.from_numpy(sos))[:3], torch.from_numpy(x.real.copy()))
    torch.testing.assert_close(y, torch.from_numpy(_expected(sos, x.real.copy())))


def test_pfefilt_float16_with_a_conjugate_pair():
    # A real float16 filter whose complex section runs in complex32.
    np.random.seed(2)
    x = np.random.randn(2, 500)
    sos = PAIR[None]
    r, p, g = sos2pfe(torch.from_numpy(sos))[:3]
    x16 = torch.from_numpy(x).half().requires_grad_()
    y = pfefilt(r.to(torch.complex32), p.to(torch.complex32), g.half(), x16)
    assert y.dtype == torch.float16
    expected = torch.from_numpy(_expected(sos, x))
    assert (y.double() - expected).abs().max() / expected.abs().max() < 2e-3
    y.sum().backward()
    x64 = x16.detach().double().requires_grad_()
    pfefilt(
        r.to(torch.complex32).cdouble(), p.to(torch.complex32).cdouble(), g.half().double(), x64
    ).sum().backward()
    assert x16.grad.dtype == torch.float16
    assert (x16.grad.double() - x64.grad).abs().max() / x64.grad.abs().max() < 2e-3


def test_pfefilt_batched_multiplicities():
    # one cascade per signal, with different multiplicities and numbers of real poles
    sos = np.stack(
        [
            [DOUBLE_REAL, DOUBLE_REAL],
            [DOUBLE_REAL, OTHER_REAL],
            [PAIR, PAIR],
            [PAIR, OTHER_REAL],
        ]
    )
    _check(sos, np.random.default_rng(0).standard_normal((4, 64)))


@pytest.mark.parametrize(
    "sos",
    [
        np.stack([PAIR, PAIR, DOUBLE_REAL, OTHER_REAL]),
        _random_sos(K=2) + 0.2j * _random_sos(K=2),
        np.stack([[PAIR, DOUBLE_REAL], [OTHER_REAL, PAIR]]),
    ],
    ids=["real", "complex", "batched"],
)
def test_pfefilt_in_chunks_equals_whole(sos):
    r, p, g, batch_index, power = sos2pfe(torch.from_numpy(sos))
    x = torch.randn(2, 60, dtype=r.dtype if sos.dtype.kind == "c" else torch.float64)
    # one state per term with batch_index, else per term and row
    zi = torch.zeros((2,) * (batch_index is None) + (p.numel(),), dtype=p.dtype)
    whole, zf_whole = pfefilt(r, p, g, x, zi=zi, batch_index=batch_index, power=power)
    ys, zf = [], zi
    for chunk in x.split([20, 0, 25, 15], dim=-1):
        y, zf = pfefilt(r, p, g, chunk, zi=zf, batch_index=batch_index, power=power)
        ys.append(y)
    torch.testing.assert_close(torch.cat(ys, dim=-1), whole)
    torch.testing.assert_close(zf, zf_whole)


def test_pfefilt_without_poles():
    x = torch.randn(2, 8, dtype=torch.float64)
    empty = torch.zeros(0, dtype=torch.complex128)
    y = pfefilt(empty, empty, torch.tensor(2.0, dtype=torch.float64), x)
    torch.testing.assert_close(y, 2 * x)
