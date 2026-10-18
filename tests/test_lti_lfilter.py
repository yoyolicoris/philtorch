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


def _empty_a_coeffs(num_taps: int, batched: bool, B: int = 3):
    """Random NumPy b and an empty tensor a, batched or shared."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal((B, num_taps) if batched else (num_taps,))
    a = torch.zeros(B, 0, dtype=torch.float64) if batched else torch.zeros(0, dtype=torch.float64)
    return b, a


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize(
    ("form", "backend", "use_zi"),
    [
        (form, backend, use_zi)
        for form, backend in [
            ("df2", "ssm"),
            ("tdf2", "ssm"),
            ("df1", "ssm"),
            ("tdf1", "ssm"),
            ("df2", "diag_ssm"),
            ("tdf2", "diag_ssm"),
        ]
        for use_zi in (False, True)
        # df1 and tdf1 do not take zi.
        if not (use_zi and form in ("df1", "tdf1"))
    ],
)
def test_lfilter_pure_gain(form: str, backend: str, use_zi: bool, batched: bool):
    B, T = 3, 20
    b, a = _empty_a_coeffs(1, batched, B)
    x = np.random.default_rng(1).standard_normal((B, T))
    # Unbatched everything: the filter is run on one signal.
    if not batched:
        x = x[0]
    kwargs = {}
    if use_zi:
        kwargs["zi"] = torch.zeros(*(() if not batched else (B,)), 0, dtype=torch.float64)
    out = lfilter(torch.from_numpy(b), a, torch.from_numpy(x), form=form, backend=backend, **kwargs)
    y, zf = out if use_zi else (out, None)
    assert y.shape == x.shape
    if batched:
        y_ref = np.stack([signal.lfilter(b[i], [1.0], x[i]) for i in range(B)])
    else:
        y_ref = signal.lfilter(b, [1.0], x)
    assert np.allclose(y.numpy(), y_ref)
    if use_zi:
        assert zf.shape == ((B, 0) if batched else (0,))


@pytest.mark.parametrize(
    ("form", "backend", "use_zi"),
    [
        (form, backend, use_zi)
        for form, backend in [
            ("df2", "ssm"),
            ("tdf2", "ssm"),
            ("df1", "ssm"),
            ("tdf1", "ssm"),
            ("df2", "diag_ssm"),
            ("tdf2", "diag_ssm"),
        ]
        for use_zi in (False, True)
        # df1 and tdf1 do not take zi.
        if not (use_zi and form in ("df1", "tdf1"))
    ],
)
def test_lfilter_pure_gain_shared_coefficients_batched_signal(
    form: str, backend: str, use_zi: bool
):
    # Shared b (1,) and a (0,) on a batch of signals; a shared zi has shape (0,).
    B, T = 3, 20
    b, a = _empty_a_coeffs(1, False, B)
    x = np.random.default_rng(1).standard_normal((B, T))
    kwargs = {"zi": torch.zeros(0, dtype=torch.float64)} if use_zi else {}
    out = lfilter(torch.from_numpy(b), a, torch.from_numpy(x), form=form, backend=backend, **kwargs)
    y, zf = out if use_zi else (out, None)
    y_ref = np.stack([signal.lfilter(b, [1.0], x[i]) for i in range(B)])
    assert y.shape == (B, T)
    assert np.allclose(y.numpy(), y_ref)
    if use_zi:
        assert zf.shape == (B, 0)


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize(
    ("form", "backend", "delayed_form"),
    [
        ("df2", "ssm", False),
        ("tdf2", "ssm", False),
        ("df1", "ssm", False),
        ("tdf1", "ssm", False),
        ("df2", "diag_ssm", False),
        ("df2", "diag_ssm", True),
        ("tdf2", "diag_ssm", False),
        ("tdf2", "diag_ssm", True),
    ],
)
def test_lfilter_empty_a_fir(form: str, backend: str, delayed_form: bool, batched: bool):
    # On "ssm", 3 taps pad a to length 2, so the empty-a shortcut is not reached:
    # these cases check the FIR result of an empty a through the general path.
    B, T = 3, 20
    b, a = _empty_a_coeffs(3, batched, B)
    x = np.random.default_rng(1).standard_normal((B, T))
    kwargs = {"delayed_form": delayed_form} if backend == "diag_ssm" else {}
    y = lfilter(
        torch.from_numpy(b),
        a,
        torch.from_numpy(x),
        form=form,
        backend=backend,
        **kwargs,
    )
    b_ref = np.broadcast_to(b, (B, 3))
    y_ref = np.stack([signal.lfilter(b_ref[i], [1.0], x[i]) for i in range(B)])
    assert np.allclose(y.numpy(), y_ref)


@pytest.mark.parametrize("batched", [True, False])
def test_lfilter_empty_a_fir_tdf2_zi(batched: bool):
    B, T = 3, 20
    rng = np.random.default_rng(2)
    b, a = _empty_a_coeffs(3, batched, B)
    x = rng.standard_normal((B, T))
    zi = rng.standard_normal((B, 2))
    b_ref = np.broadcast_to(b, (B, 3))
    refs = [signal.lfilter(b_ref[i], [1.0], x[i], zi=zi[i]) for i in range(B)]
    y_ref = np.stack([r[0] for r in refs])
    zf_ref = np.stack([r[1] for r in refs])
    y, zf = lfilter(
        torch.from_numpy(b), a, torch.from_numpy(x), zi=torch.from_numpy(zi), form="tdf2"
    )
    assert np.allclose(y.numpy(), y_ref)
    assert np.allclose(zf.numpy(), zf_ref)


@pytest.mark.parametrize("backend", ["ssm", "diag_ssm"])
@pytest.mark.parametrize("num_taps", [1, 3])
def test_lfilter_empty_a_gradients(backend: str, num_taps: int):
    rng = np.random.default_rng(3)
    b = torch.from_numpy(rng.standard_normal((2, num_taps))).requires_grad_()
    x = torch.from_numpy(rng.standard_normal((2, 12))).requires_grad_()
    a = torch.zeros(2, 0, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda b, x: lfilter(b, a, x, form="tdf2", backend=backend), (b, x)
    )


@pytest.mark.parametrize("backend", ["ssm", "diag_ssm"])
@pytest.mark.parametrize("a", [torch.zeros(0), torch.zeros(1)], ids=["empty-a", "order-1"])
def test_lfilter_rejects_0d_signal(backend: str, a: torch.Tensor):
    with pytest.raises(ValueError, match="x must be 1D or 2D"):
        lfilter(torch.ones(1), a, torch.tensor(2.0), backend=backend)


@pytest.mark.parametrize("backend", ["ssm", "diag_ssm"])
@pytest.mark.parametrize("zi_shape", [(0,), (2, 0)])
def test_lfilter_pure_gain_final_state_keeps_zi_in_the_graph(backend: str, zi_shape):
    # An empty final state has a zero gradient for zi, not none.
    zi = torch.zeros(zi_shape, dtype=torch.float64, requires_grad=True)
    b = torch.ones(1, dtype=torch.float64)
    x = torch.randn(2, 4, dtype=torch.float64)
    _, zf = lfilter(b, torch.zeros(0, dtype=torch.float64), x, zi=zi, backend=backend)
    (grad,) = torch.autograd.grad(zf.sum(), zi)
    assert grad.shape == zi.shape


@pytest.mark.parametrize(
    ("match", "num_taps", "zi_shape", "kwargs"),
    [
        ("0 states", 1, (2, 1), {"form": "tdf2"}),
        ("zi must be", 1, (2, 1, 0), {"backend": "ssm"}),
        ("zi must be", 1, (2, 1, 0), {"backend": "diag_ssm"}),
        ("Unknown filter form", 3, None, {"form": "df1", "backend": "diag_ssm"}),
        ("does not take zi", 3, (2, 2), {"backend": "diag_ssm"}),
    ],
    ids=["zi-with-states", "zi-3d-ssm", "zi-3d-diag_ssm", "diag_ssm-df1", "diag_ssm-zi-long-b"],
)
def test_lfilter_empty_a_errors(
    match: str, num_taps: int, zi_shape: tuple[int, ...] | None, kwargs: dict
):
    f64 = {"dtype": torch.float64}
    b = torch.ones(2, num_taps, **f64)
    a = torch.zeros(2, 0, **f64)
    x = torch.ones(2, 10, **f64)
    if zi_shape is not None:
        kwargs = {**kwargs, "zi": torch.zeros(*zi_shape, **f64)}
    with pytest.raises(ValueError, match=match):
        lfilter(b, a, x, **kwargs)


@pytest.mark.parametrize("backend", ["ssm", "diag_ssm"])
@pytest.mark.parametrize("zi_shape", [None, (0,), (3, 0)])
def test_lfilter_no_taps_is_zero(backend: str, zi_shape: tuple[int, ...] | None):
    # b with no coefficients is the zero filter, with or without poles.
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((3, 10)))
    b = torch.zeros(0, dtype=torch.float64)
    a = torch.zeros(0, dtype=torch.float64)
    kwargs = {} if zi_shape is None else {"zi": torch.zeros(*zi_shape, dtype=torch.float64)}
    out = lfilter(b, a, x, backend=backend, **kwargs)
    y, zf = (out, None) if zi_shape is None else out
    assert y.shape == x.shape
    assert torch.equal(y, torch.zeros_like(x))
    if zf is not None:
        assert zf.shape == (3, 0)


_EMPTY_A_UNROLL_CASES = [(form, "ssm", 1) for form in ("df2", "tdf2", "df1", "tdf1")] + [
    (form, "diag_ssm", taps) for form in ("df2", "tdf2") for taps in (1, 3)
]


@pytest.mark.parametrize(("form", "backend", "num_taps"), _EMPTY_A_UNROLL_CASES)
@pytest.mark.parametrize(
    ("unroll_factor", "error"),
    [
        (0, ValueError),
        (-3, ValueError),
        (1.5, TypeError),
        (np.int64(2), None),
        (torch.tensor(3), None),
        (True, None),
        (None, None),
    ],
    ids=["0", "-3", "1.5", "np.int64", "tensor", "True", "None"],
)
def test_lfilter_empty_a_unroll_factor(
    form: str, backend: str, num_taps: int, unroll_factor, error
):
    # The empty-a shortcuts check unroll_factor as the general path does.
    rng = np.random.default_rng(0)
    b = rng.standard_normal(num_taps)
    x = rng.standard_normal((3, 12))
    a = torch.zeros(0, dtype=torch.float64)
    args = (torch.from_numpy(b), a, torch.from_numpy(x))
    kwargs = {"form": form, "backend": backend, "unroll_factor": unroll_factor}
    if error is not None:
        match = (
            "must be None or at least 1" if error is ValueError else "must be None or an integer"
        )
        with pytest.raises(error, match=match):
            lfilter(*args, **kwargs)
        return
    y = lfilter(*args, **kwargs)
    assert np.allclose(y.numpy(), signal.lfilter(b, [1.0], x, axis=-1))


@pytest.mark.parametrize(
    ("form", "case"),
    [
        (form, case)
        for form in ("df2", "tdf2", "df1", "tdf1")
        for case in ("b32", "zi64", "b-complex", "b-meta")
        # df1 and tdf1 do not take zi.
        if not (case == "zi64" and form in ("df1", "tdf1"))
    ],
)
def test_lfilter_pure_gain_rejects_dtype_and_device(form: str, case: str):
    x_dtype = torch.float32 if case == "zi64" else torch.float64
    x = torch.ones(3, 10, dtype=x_dtype)
    a = torch.zeros(0, dtype=torch.float64)
    b = torch.ones(1, dtype=x_dtype)
    kwargs = {}
    match case:
        case "b32":
            b = b.float()
        case "zi64":
            kwargs["zi"] = torch.zeros(0, dtype=torch.float64)
        case "b-complex":
            b = b.to(torch.complex128)
        case "b-meta":
            b = b.to("meta")
    with pytest.raises(ValueError, match="share a dtype and a device"):
        lfilter(b, a, x, form=form, backend="ssm", **kwargs)


@pytest.mark.parametrize("form", ["df2", "tdf2", "df1", "tdf1"])
def test_lfilter_pure_gain_accepts_empty_a_of_another_dtype(form: str):
    # torch.tensor([]) is float32, but an empty a holds no values.
    rng = np.random.default_rng(0)
    b = torch.from_numpy(rng.standard_normal(1))
    x = torch.from_numpy(rng.standard_normal((3, 10)))
    y = lfilter(b, torch.tensor([]), x, form=form, backend="ssm")
    assert y.dtype == torch.float64
    assert torch.allclose(y, b[0] * x)


@pytest.mark.parametrize(("backend", "num_taps"), [("ssm", 1), ("diag_ssm", 1), ("diag_ssm", 3)])
@pytest.mark.parametrize("dtype", [torch.complex128, torch.float32, torch.complex64])
def test_lfilter_empty_a_complex_and_single_precision(
    backend: str, num_taps: int, dtype: torch.dtype
):
    # Compared with SciPy, so that a conjugated b is caught.
    rng = np.random.default_rng(0)

    def draw(*shape):
        v = rng.standard_normal(shape)
        return v + 1j * rng.standard_normal(shape) if dtype.is_complex else v

    b, x = draw(num_taps), draw(3, 16)
    a = torch.zeros(0, dtype=dtype)
    y = lfilter(torch.from_numpy(b).to(dtype), a, torch.from_numpy(x).to(dtype), backend=backend)
    assert y.dtype == dtype
    y_ref = signal.lfilter(b, [1.0], x, axis=-1)
    tol = 1e-12 if dtype == torch.complex128 else 1e-4
    assert np.allclose(y.numpy(), y_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize(("backend", "num_taps"), [("ssm", 1), ("diag_ssm", 1), ("diag_ssm", 3)])
def test_lfilter_empty_a_rejects_several_unknown_keywords(backend: str, num_taps: int):
    # The first unknown keyword is the one named.
    b = torch.ones(2, num_taps, dtype=torch.float64)
    a = torch.zeros(2, 0, dtype=torch.float64)
    x = torch.ones(2, 10, dtype=torch.float64)
    with pytest.raises(TypeError, match="unexpected keyword argument 'unrol_factor'"):
        lfilter(b, a, x, backend=backend, unrol_factor=3, foo=1)


@pytest.mark.parametrize("num_taps", [1, 3])
def test_lfilter_empty_a_diag_ssm_accepts_no_eigenbasis(num_taps: int):
    b = torch.ones(2, num_taps, dtype=torch.float64)
    a = torch.zeros(2, 0, dtype=torch.float64)
    x = torch.ones(2, 10, dtype=torch.float64)
    y = lfilter(b, a, x, backend="diag_ssm", V=None, Vinv=None)
    assert torch.allclose(y, lfilter(b, a, x, backend="diag_ssm"))


@pytest.mark.parametrize(("backend", "num_taps"), [("ssm", 1), ("diag_ssm", 1), ("diag_ssm", 3)])
def test_lfilter_empty_a_rejects_unknown_keyword(backend: str, num_taps: int):
    # A misspelt keyword must fail as it does on the general path.
    b = torch.ones(2, num_taps, dtype=torch.float64)
    a = torch.zeros(2, 0, dtype=torch.float64)
    x = torch.ones(2, 10, dtype=torch.float64)
    with pytest.raises(TypeError, match="unexpected keyword argument 'unrol_factor'"):
        lfilter(b, a, x, backend=backend, unrol_factor=3)


def test_lfilter_empty_a_keywords_of_each_backend():
    b = torch.ones(2, 1, dtype=torch.float64)
    a = torch.zeros(2, 0, dtype=torch.float64)
    x = torch.ones(2, 10, dtype=torch.float64)
    # diag_ssm takes L; ssm does not.
    assert lfilter(b, a, x, backend="diag_ssm", L=None).shape == x.shape
    with pytest.raises(TypeError, match="unexpected keyword argument 'L'"):
        lfilter(b, a, x, backend="ssm", L=None)
    # Both take unroll_factor.
    for backend in ("ssm", "diag_ssm"):
        assert lfilter(b, a, x, backend=backend, unroll_factor=3).shape == x.shape


@pytest.mark.parametrize(
    ("backend", "name", "num_taps"),
    # On "ssm", more than 1 tap takes the general path; on "diag_ssm", zi needs 1 tap.
    [
        ("ssm", "a", 1),
        ("ssm", "b", 1),
        ("ssm", "zi", 1),
        ("diag_ssm", "a", 1),
        ("diag_ssm", "b", 1),
        ("diag_ssm", "zi", 1),
        ("diag_ssm", "a", 3),
        ("diag_ssm", "b", 3),
    ],
)
def test_lfilter_empty_a_rejects_batch_mismatch(backend: str, name: str, num_taps: int):
    f64 = {"dtype": torch.float64}
    b = torch.ones(num_taps, **f64)
    a = torch.zeros(0, **f64)
    x = torch.ones(3, 10, **f64)
    kwargs = {"form": "tdf2"}
    if name == "a":
        a = torch.zeros(2, 0, **f64)
    elif name == "b":
        b = torch.ones(2, num_taps, **f64)
    else:
        kwargs["zi"] = torch.zeros(5, 0, **f64)
    with pytest.raises(ValueError, match=f"Batch size of {name}"):
        lfilter(b, a, x, backend=backend, **kwargs)


def _assert_every_input_gets_a_gradient(outputs, inputs):
    """Every input reaches the outputs, so none gets a gradient of None."""
    flat = [t for t in (outputs if isinstance(outputs, tuple) else (outputs,))]
    total = sum(t.sum() for t in flat)
    for grad in torch.autograd.grad(total, inputs, allow_unused=True):
        assert grad is not None


@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("with_zi", [False, True])
def test_fir_empty_signal(transpose: bool, with_zi: bool):
    # The convolutions reject empty signals; fir returns an empty output and
    # the unchanged state.
    b = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    x = torch.randn(2, 0, dtype=torch.float64, requires_grad=True)
    zi = torch.randn(2, 2, dtype=torch.float64, requires_grad=True) if with_zi else None
    result = fir(b, x, zi, transpose=transpose)
    y, zf = result if with_zi else (result, None)
    assert y.shape == (2, 0)
    if with_zi:
        torch.testing.assert_close(zf, zi)
    _assert_every_input_gets_a_gradient(result, [b, x] + ([zi] if with_zi else []))


_GRADIENT_CASES = [
    (backend, form, num_taps, num_poles)
    for backend, forms in (("ssm", ("df2", "tdf2", "df1", "tdf1")), ("diag_ssm", ("df2", "tdf2")))
    for form in forms
    for num_taps, num_poles in ((1, 0), (3, 0), (2, 1), (3, 1))
]


@pytest.mark.parametrize(("backend", "form", "num_taps", "num_poles"), _GRADIENT_CASES)
@pytest.mark.parametrize("samples", [0, 6])
def test_lfilter_every_input_gets_a_gradient(backend, form, num_taps, num_poles, samples):
    # Shortcuts must not drop inputs from the graph: an empty a, an empty
    # final state or an empty signal still gives zero gradients, not None.
    f64 = dict(dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, num_taps, **f64)
    a = (0.3 * torch.randn(2, num_poles, dtype=torch.float64)).requires_grad_()
    x = torch.randn(2, samples, **f64)
    inputs = [b, a, x]
    kwargs = {"form": form, "backend": backend}
    order = max(num_taps - 1, num_poles) if backend == "ssm" else num_poles
    takes_zi = form in ("df2", "tdf2") and not (backend == "diag_ssm" and num_taps > num_poles + 1)
    if takes_zi:
        kwargs["zi"] = torch.randn(2, order, **f64)
        inputs.append(kwargs["zi"])
    outputs = lfilter(b, a, x, **kwargs)
    _assert_every_input_gets_a_gradient(outputs, inputs)


@pytest.mark.parametrize("form", ["df1", "tdf1", "df2", "tdf2"])
@pytest.mark.parametrize("x_shape", [(10,), (3, 10)])
def test_lfilter_batched_b_that_fits_no_x_raises(form: str, x_shape: tuple[int, ...]):
    # "df2" once returned (1, 10, 4) for an unbatched x, and "df1" and "tdf1"
    # raised a RuntimeError.
    with pytest.raises(ValueError, match="Batch size of b must be 1 or match"):
        lfilter(torch.randn(4, 2), 0.3 * torch.ones(1), torch.randn(x_shape), form=form)


@pytest.mark.parametrize(("form", "backend"), [("df2", "ssm"), ("df2", "diag_ssm"), ("df1", "ssm")])
def test_lfilter_b_with_batch_of_one_is_shared(form: str, backend: str):
    # "df2" rejected it; "df1" already shared it.
    b, a, x = torch.randn(3), 0.3 * torch.ones(2), torch.randn(4, 12)
    y = lfilter(b, a, x, form=form, backend=backend)
    torch.testing.assert_close(lfilter(b[None], a, x, form=form, backend=backend), y)


@pytest.mark.parametrize("batched", [False, True])
def test_filtfilt_df2_empty_a_is_a_gain(batched: bool):
    # lfilter_zi indexed the empty state of an empty a.
    b = torch.randn((2, 1) if batched else (1,), dtype=torch.float64)
    x = torch.randn(2, 30, dtype=torch.float64)
    y = filtfilt(b, torch.zeros(0, dtype=torch.float64), x, form="df2")
    torch.testing.assert_close(y, b.reshape(-1, 1) ** 2 * x)


@pytest.mark.parametrize("backend", ["ssm", "diag_ssm"])
@pytest.mark.parametrize(
    ("b_dtype", "a"),
    [
        (torch.complex128, 0.3 * torch.ones(1, dtype=torch.complex128)),
        (torch.float32, 0.3 * torch.ones(1, dtype=torch.float32)),
        (torch.float32, torch.zeros(0, dtype=torch.float64)),
    ],
    ids=["complex", "float32", "fir-float32"],
)
def test_lfilter_mixed_dtypes_raise_the_same_error_in_every_backend(backend, b_dtype, a):
    # "diag_ssm" promoted or raised a RuntimeError; "ssm" promoted an FIR b.
    b = torch.ones(3, dtype=b_dtype)
    with pytest.raises(ValueError, match="share a dtype and a device"):
        lfilter(b, a, torch.randn(10, dtype=torch.float64), backend=backend)
