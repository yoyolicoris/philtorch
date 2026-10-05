import numpy as np
import pytest
import torch
from scipy import signal

from philtorch.lti import filtfilt, fir, lfilter, lfilter_zi, lfiltic
from philtorch.mat import vandermonde


def _generate_random_filter_coeffs(
    num_order: int, den_order: int, B: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random filter coefficients"""

    # Time-invariant coefficients
    b = np.random.randn(B, num_order + 1)
    a = np.random.randn(B, den_order)
    a = a / np.abs(a).sum(axis=-1, keepdims=True)

    return b, a


def _generate_random_signal(B: int, T: int) -> np.ndarray:
    """Generate random input signal"""
    return np.random.randn(B, T)


def _generate_a(den_order):
    num_cmplx_poles = den_order // 2
    num_real_poles = den_order - 2 * num_cmplx_poles

    cmplx_poles = np.random.rand(num_cmplx_poles) ** 0.5 * np.exp(
        1j * np.random.rand(num_cmplx_poles) * 2 * np.pi
    )
    real_poles = np.random.rand(num_real_poles) ** 0.5 + 0j
    roots = np.concatenate([cmplx_poles, cmplx_poles.conj(), real_poles])
    a = np.polynomial.Polynomial.fromroots(roots).coef.real[-2::-1].copy()
    return a, roots


@pytest.mark.parametrize("b_shape", [(3, 5), (5,)])
@pytest.mark.parametrize("a_shape", [(3, 4), (4,)])
@pytest.mark.parametrize("padmode", ["reflect", "replicate", None])
@pytest.mark.parametrize("padlen", [None, 0, 21])
def test_filtfilt(b_shape, a_shape, padmode, padlen):
    x = np.random.randn(3, 100)
    b = np.random.randn(*b_shape)
    if len(a_shape) == 1:
        a = _generate_a(a_shape[0])[0]
    else:
        a = np.stack([_generate_a(a_shape[1])[0] for _ in range(a_shape[0])], axis=0)

    # Convert to torch tensors
    b_torch = torch.from_numpy(b)
    a_torch = torch.from_numpy(a)
    x_torch = torch.from_numpy(x)

    b = np.broadcast_to(b, (x.shape[0], b.shape[-1]))
    a = np.broadcast_to(a, (x.shape[0], a.shape[-1]))
    match padmode:
        case "reflect":
            padtype = "even"
        case "replicate":
            padtype = "constant"
        case None:
            padtype = None
        case _:
            raise ValueError(f"Unsupported padmode: {padmode}")

    # Apply scipy filtfilt
    y_scipy = np.stack(
        [
            signal.filtfilt(b[i], [1.0] + a[i].tolist(), x[i], padtype=padtype, padlen=padlen)
            for i in range(b.shape[0])
        ],
        axis=0,
    )

    # Apply philtorch filtfilt
    y_torch = filtfilt(b_torch, a_torch, x_torch, padmode=padmode, padlen=padlen)

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(np.abs(y_torch.numpy() - y_scipy))


@pytest.mark.parametrize("b_shape", [(3, 5), (5,)])
@pytest.mark.parametrize("a_shape", [(3, 2), (2,)])
@pytest.mark.parametrize("x_shape", [None, (4,), (3, 4)])
@pytest.mark.parametrize("y_shape", [(2,), (3, 2)])
def test_lfiltic(b_shape, a_shape, y_shape, x_shape):
    """Test lfiltic function"""

    # Generate random filter coefficients
    b = np.random.randn(*b_shape)
    if len(a_shape) == 1:
        a = _generate_a(a_shape[0])[0]
    else:
        a = np.stack([_generate_a(a_shape[1])[0] for _ in range(a_shape[0])], axis=0)
    y = np.random.randn(*y_shape)
    x = np.random.randn(*x_shape) if x_shape is not None else None

    # Convert to torch tensors
    b_torch = torch.from_numpy(b)
    a_torch = torch.from_numpy(a)
    y_torch = torch.from_numpy(y)
    x_torch = torch.from_numpy(x) if x is not None else None

    if b.ndim > 1:
        batch_size = b.shape[0]
    elif a.ndim > 1:
        batch_size = a.shape[0]
    elif y.ndim > 1:
        batch_size = y.shape[0]
    elif x is not None and x.ndim > 1:
        batch_size = x.shape[0]
    else:
        batch_size = 1
    b = np.broadcast_to(b, (batch_size, b.shape[-1]))
    a = np.broadcast_to(a, (batch_size, a.shape[-1]))
    y = np.broadcast_to(y, (batch_size, y.shape[-1]))
    if x is not None:
        x = np.broadcast_to(x, (batch_size, x.shape[-1]))

    # Apply scipy lfiltic
    zi_scipy = np.stack(
        [
            signal.lfiltic(b[i], [1.0] + a[i].tolist(), y[i], x[i] if x is not None else None)
            for i in range(b.shape[0])
        ],
        axis=0,
    )
    if b.ndim == 1:
        zi_scipy = zi_scipy.flatten()

    # Apply philtorch lfiltic
    zi_torch = lfiltic(b_torch, a_torch, y_torch, x_torch)

    # Compare outputs
    assert np.allclose(zi_torch.numpy(), zi_scipy), np.max(np.abs(zi_torch.numpy() - zi_scipy))


@pytest.mark.parametrize("b_shape", [(3, 5), (4,)])
@pytest.mark.parametrize("a_shape", [(3, 2), (5,)])
def test_lfilter_zi(b_shape, a_shape):
    """Test lfilter_zi function"""

    # Generate random filter coefficients
    b = np.random.randn(*b_shape)
    if len(a_shape) == 1:
        a = _generate_a(a_shape[0])[0]
    else:
        a = np.stack([_generate_a(a_shape[1])[0] for _ in range(a_shape[0])], axis=0)

    # Convert to torch tensors
    b_torch = torch.from_numpy(b)
    a_torch = torch.from_numpy(a)

    if b.ndim > 1:
        batch_size = b.shape[0]
    elif a.ndim > 1:
        batch_size = a.shape[0]
    else:
        batch_size = 1
    b = np.broadcast_to(b, (batch_size, b.shape[-1]))
    a = np.broadcast_to(a, (batch_size, a.shape[-1]))
    # Apply scipy lfilter_zi
    zi_scipy = np.stack(
        [signal.lfilter_zi(b[i], [1.0] + a[i].tolist()) for i in range(b.shape[0])],
        axis=0,
    )
    if b.ndim == 1:
        zi_scipy = zi_scipy.flatten()

    # Apply philtorch lfilter_zi
    zi_torch = lfilter_zi(b=b_torch, a=a_torch)

    # Compare outputs
    assert np.allclose(zi_torch.numpy(), zi_scipy), np.max(np.abs(zi_torch.numpy() - zi_scipy))


@pytest.mark.parametrize("transpose", [True, False])
def test_zi_constant_response(transpose: bool):
    b, a = signal.butter(5, 0.25)
    a0 = a[0]
    b = b / a0
    a = a[1:] / a0

    b = torch.from_numpy(b).float()
    a = torch.from_numpy(a).float()

    zi = lfilter_zi(a, b if transpose else None, transpose=transpose)
    y, _ = lfilter(b, a, torch.ones(3, 10), zi=zi, form="df2" if not transpose else "tdf2")

    assert torch.all(torch.diff(y).abs() < 1e-5), y.diff().abs().max()


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
    y_torch = fir(b_torch, x_torch, transpose=False)
    # Apply scipy filter
    y_scipy = np.stack([signal.lfilter(b[i], [1.0], x[i]) for i in range(B)], axis=0)

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(np.abs(y_torch.numpy() - y_scipy))


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
    torch_results = fir(b_torch, x_torch, zi=zi_torch, transpose=True)
    # Apply scipy filter
    scipy_results = [
        signal.lfilter(b[i], [1.0], x[i], zi=zi[i] if zi is not None else None) for i in range(B)
    ]

    if include_zi:
        y_scipy, zf_scipy = zip(*scipy_results)
        y_scipy = np.stack(y_scipy, axis=0)
        zf_scipy = np.stack(zf_scipy, axis=0)
        y_torch, zf_torch = torch_results

        assert np.allclose(zf_torch.numpy(), zf_scipy), np.max(np.abs(zf_torch.numpy() - zf_scipy))
    else:
        y_scipy = np.vstack(scipy_results)
        y_torch = torch_results

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(np.abs(y_torch.numpy() - y_scipy))


@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("T", [32, 128])
@pytest.mark.parametrize("num_order", [1, 2, 4])
@pytest.mark.parametrize("den_order", [1, 3, 5])
@pytest.mark.parametrize("form", ["df2", "tdf2", "df1", "tdf1"])
def test_time_invariant_filter(B: int, T: int, num_order: int, den_order: int, form: str):
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
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(np.abs(y_torch.numpy() - y_scipy))


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
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(np.abs(y_torch.numpy() - y_scipy))
    assert np.allclose(zf_torch.numpy(), zf_scipy), np.max(np.abs(zf_torch.numpy() - zf_scipy))


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

    # Seeded: about 3 in 10,000 draws place two poles almost on top of each
    # other, which makes diagonalizing the filter ill-conditioned and misses
    # np.allclose's tolerance.
    np.random.seed(0)

    # Generate test data
    # b, a = _generate_random_filter_coeffs(num_order, den_order, B)
    b = np.random.randn(B, num_order + 1)
    a, roots = _generate_a(den_order)

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
    y_scipy = np.stack([signal.lfilter(b[i], [1.0] + a.tolist(), x[i]) for i in range(B)], axis=0)

    # Compare outputs
    assert np.allclose(y_torch.numpy(), y_scipy), np.max(np.abs(y_torch.numpy() - y_scipy))


# 1 - z^-1 + 0.5 z^-2 has complex poles, so the diag_ssm eigendecomposition is
# complex even though the filter is real.
_COMPLEX_POLES = np.array([-1.0, 0.5])


@pytest.mark.parametrize("form", ["df2", "tdf2"])
@pytest.mark.parametrize("delayed_form", [False, True])
# delayed_form delays the recursive part by b.size(-1) - a.size(-1) =
# M_b - M_a + 1 = 7 samples, more than the shortest signal.
@pytest.mark.parametrize("T", [4, 40])
def test_diag_ssm_unbatched_long_numerator(form: str, delayed_form: bool, T: int):
    rng = np.random.default_rng(0)
    b = rng.standard_normal(9)  # M_b = 8 > M_a = 2, split into an FIR part
    x = rng.standard_normal((3, T))
    y = lfilter(
        torch.from_numpy(b),
        torch.from_numpy(_COMPLEX_POLES),
        torch.from_numpy(x),
        form=form,
        backend="diag_ssm",
        delayed_form=delayed_form,
    )
    assert np.allclose(y.numpy(), signal.lfilter(b, np.r_[1, _COMPLEX_POLES], x))


@pytest.mark.parametrize("form", ["df2", "tdf2"])
def test_diag_ssm_real_filter_gives_real_output(form: str):
    rng = np.random.default_rng(0)
    b = rng.standard_normal(3)
    x = rng.standard_normal((3, 40))
    zi = rng.standard_normal((3, 2))
    a = torch.from_numpy(_COMPLEX_POLES)

    y = lfilter(torch.from_numpy(b), a, torch.from_numpy(x), form=form, backend="diag_ssm")
    assert not y.is_complex()
    assert np.allclose(y.numpy(), signal.lfilter(b, np.r_[1, _COMPLEX_POLES], x))

    y, zf = lfilter(
        torch.from_numpy(b),
        a,
        torch.from_numpy(x),
        zi=torch.from_numpy(zi),
        form=form,
        backend="diag_ssm",
    )
    assert not y.is_complex() and not zf.is_complex()
    if form == "tdf2":
        y_ref, zf_ref = signal.lfilter(b, np.r_[1, _COMPLEX_POLES], x, zi=zi)
        assert np.allclose(y.numpy(), y_ref) and np.allclose(zf.numpy(), zf_ref)


@pytest.mark.parametrize(
    ("form", "backend", "num_taps"),
    [("df1", "ssm", 3), ("tdf1", "ssm", 3), ("df2", "diag_ssm", 5), ("tdf2", "diag_ssm", 5)],
)
def test_zi_rejected_where_unsupported(form: str, backend: str, num_taps: int):
    b = torch.ones(num_taps, dtype=torch.float64)
    a = torch.from_numpy(_COMPLEX_POLES)
    x = torch.ones(2, 10, dtype=torch.float64)
    zi = torch.zeros(2, max(num_taps - 1, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="does not take zi"):
        lfilter(b, a, x, zi=zi, form=form, backend=backend)


@pytest.mark.parametrize("transpose", [True, False])
def test_fir_one_tap(transpose: bool):
    x = torch.randn(2, 10, dtype=torch.float64)
    b = torch.full((2, 1), 2.0, dtype=torch.float64)
    assert torch.allclose(fir(b, x, transpose=transpose), 2 * x)
    y, zf = fir(b, x, zi=x.new_zeros(2, 0), transpose=transpose)
    assert torch.allclose(y, 2 * x)
    assert zf.shape == (2, 0)


@pytest.mark.parametrize(
    ("num_taps", "y_len", "x_len"),
    [
        (1, 2, None),  # one-tap b
        (4, 1, 2),  # short histories are padded with zeros
        (4, 3, 4),  # long histories are truncated
        (4, 2, 3),  # exact
    ],
)
def test_lfiltic_matches_scipy(num_taps: int, y_len: int, x_len: int | None):
    rng = np.random.default_rng(0)
    b = rng.standard_normal(num_taps)
    y = rng.standard_normal(y_len)
    x = None if x_len is None else rng.standard_normal(x_len)
    zi = lfiltic(
        torch.from_numpy(b),
        torch.from_numpy(_COMPLEX_POLES),
        torch.from_numpy(y),
        None if x is None else torch.from_numpy(x),
    )
    assert np.allclose(zi.numpy(), signal.lfiltic(b, np.r_[1, _COMPLEX_POLES], y, x))


@pytest.mark.parametrize("form", ["df1", "tdf1"])
def test_filtfilt_rejects_direct_form_one(form: str):
    with pytest.raises(ValueError, match="filtfilt needs form"):
        filtfilt(torch.ones(3), torch.tensor([-0.5]), torch.randn(50), form=form)


@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("N", [2, 5, 9])
def test_fir_state_with_signals_shorter_than_the_order(transpose: bool, N: int):
    # With N < M, part of zi is still in the final state.
    rng = np.random.default_rng(0)
    b = rng.standard_normal(6)  # M = 5
    x = rng.standard_normal((2, N))
    zi = rng.standard_normal((2, 5))
    y, zf = fir(
        torch.from_numpy(np.stack([b, b])),
        torch.from_numpy(x),
        zi=torch.from_numpy(zi),
        transpose=transpose,
    )
    if transpose:
        y_ref, zf_ref = signal.lfilter(b, [1], x, zi=zi)
    else:  # zi holds the past inputs, newest first
        history = np.concatenate([zi[:, ::-1], x], axis=1)
        y_ref, zf_ref = signal.lfilter(b, [1], history)[:, 5:], history[:, ::-1][:, :5]
    assert np.allclose(y.numpy(), y_ref)
    assert np.allclose(zf.numpy(), zf_ref)
