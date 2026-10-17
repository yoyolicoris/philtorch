import math

import pytest
import torch
from scipy.interpolate import CubicSpline
from scipy.signal import cspline1d, cspline1d_eval, kaiser_beta, kaiserord, resample_poly

from philtorch.lti.interp import (
    _QUALITY_PRESETS,
    _QUALITY_RECIPES,
    _quality_kaiser_params,
    cspline,
    cubic_spline,
    resample,
)


@pytest.mark.parametrize("n", [20, 100])
@pytest.mark.parametrize("m", [2, 3, 7])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.cuda,
        ),
    ],
)
@pytest.mark.parametrize("pfe", [True, False])
@pytest.mark.parametrize("scipy_padding", [True, False])
def test_cubic_spline(n, m, device, pfe, scipy_padding):
    x = torch.randn(n, device=device).double()
    if not scipy_padding:
        x = torch.cat([x, x.flip(0)[1:]])  # make it periodic

    y_philtorch = (
        cubic_spline(x.unsqueeze(0), m, scipy_padding=scipy_padding, parallel_form=pfe)
        .squeeze()
        .cpu()
    )
    t_new = torch.arange(0, len(x) - 1 + 1 / m / 2, 1 / m, dtype=torch.double)
    if scipy_padding:
        y_scipy = cspline1d_eval(cspline1d(x.cpu().numpy()), t_new.cpu().numpy())
    else:
        t = torch.arange(len(x)).double()
        cs = CubicSpline(t.cpu().numpy(), x.cpu().numpy(), bc_type="periodic")
        y_scipy = cs(t_new.cpu().numpy())

    assert torch.allclose(y_philtorch, torch.from_numpy(y_scipy)), (
        f"Max abs diff: {(y_philtorch - torch.from_numpy(y_scipy)).abs().max()}"
    )


@pytest.mark.parametrize("n", [20])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.cuda,
        ),
    ],
)
@pytest.mark.parametrize("pfe", [True, False])
def test_cspline(n, device, pfe):
    x = torch.randn(n, device=device).double()
    y_philtorch = cspline(x.unsqueeze(0), parallel_form=pfe, scipy_padding=True).squeeze().cpu()
    y_scipy = cspline1d(x.cpu().numpy())

    assert torch.allclose(y_philtorch, torch.from_numpy(y_scipy)), (
        f"Max abs diff: {(y_philtorch - torch.from_numpy(y_scipy)).abs().max()}"
    )


@pytest.mark.parametrize("shape", [(16,), (2, 2, 16)])
def test_cubic_spline_requires_batched_signals(shape):
    with pytest.raises(ValueError, match="must be 2D"):
        cubic_spline(torch.randn(shape), 4)


_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
_DTYPES = [torch.float64, torch.float32, torch.float16, torch.bfloat16]
# Tolerances against the float64 dense reference, per dtype: float64 and
# float32 accumulate in their own type; float16 and bfloat16 accumulate in
# float32 and round once, so they differ from the rounded reference by about
# one unit in the last place.
_TOLS = {
    torch.float64: dict(rtol=1e-12, atol=1e-12),
    torch.float32: dict(rtol=5e-6, atol=5e-6),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
    torch.bfloat16: dict(rtol=8e-3, atol=8e-3),
}
_MAX_TABLE = 2**20  # kMaxTable in sinc_resample.h
# A design with a short filter, for the gradient checks.
_GRAD_DESIGN = dict(quality=None, rolloff=0.85, beta=9.0, num_zeros=4)


def _rates(orig_freq, new_freq):
    """(p, q): new_freq and orig_freq divided by their gcd."""
    g = math.gcd(orig_freq, new_freq)
    return new_freq // g, orig_freq // g


def _default_out_len(T, orig_freq, new_freq):
    return -(-T * new_freq // orig_freq)


def _cutoff_and_support(orig_freq, new_freq, rolloff, num_zeros):
    """The cutoff c and the half-width W = num_zeros / c, in input samples."""
    p, q = _rates(orig_freq, new_freq)
    c = rolloff * min(1.0, p / q)
    return c, num_zeros / c


def _design(quality, rolloff=None, beta=None, num_zeros=None):
    """(rolloff, beta, num_zeros) of a call, as resample resolves them."""
    if quality is None:
        return rolloff, beta, num_zeros
    r, b, z = _QUALITY_PRESETS[quality]
    return (
        r if rolloff is None else rolloff,
        b if beta is None else beta,
        z if num_zeros is None else num_zeros,
    )


def _uses_table(orig_freq, new_freq, rolloff, num_zeros, out_len):
    """Which path the kernels take: sinc_resample.h's make_resample_plan."""
    p, _ = _rates(orig_freq, new_freq)
    _, support = _cutoff_and_support(orig_freq, new_freq, rolloff, num_zeros)
    taps = math.floor(2.0 * support) + 2
    return p <= out_len and p <= _MAX_TABLE // taps


def _kaiser_sinc(delta, c, support, beta):
    """c sinc(c delta) I0(beta sqrt(1 - (delta / W)^2)) / I0(beta), 0 beyond W.

    delta: offsets t_m - n in input samples, a float64 tensor; W = support.
    """
    u = delta / support
    arg = beta * torch.sqrt((1.0 - u * u).clamp_min(0.0))
    i0_beta = torch.special.i0(torch.tensor(beta, device=delta.device, dtype=torch.float64))
    window = torch.where(u.abs() < 1.0, torch.special.i0(arg) / i0_beta, 0.0)
    return c * torch.sinc(c * delta) * window


def _reference_resample(
    x: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    rolloff: float,
    beta: float,
    num_zeros: int,
    out_len: int,
) -> torch.Tensor:
    """Dense float64 evaluation of the formula in resample's docstring.

    The offsets t_m - n, with t_m = m q / p, are built from integers, as
    (m q // p - n) + (m q % p) / p, so that they carry no rounding of m q / p.
    """
    p, q = _rates(orig_freq, new_freq)
    c, support = _cutoff_and_support(orig_freq, new_freq, rolloff, num_zeros)
    m = torch.arange(out_len, device=x.device, dtype=torch.int64)
    n = torch.arange(x.size(1), device=x.device, dtype=torch.int64)
    whole = (m * q) // p
    frac = ((m * q) % p).to(torch.float64) / p
    delta = (whole[:, None] - n[None, :]).to(torch.float64) + frac[:, None]
    weights = _kaiser_sinc(delta, c, support, beta)
    return (x.to(torch.float64) @ weights.mT).to(x.dtype)


def test_quality_presets_follow_kaiser_design():
    # Kaiser's formulas, as scipy has them, from soxr's precision and passband.
    documented = {  # rolloff, beta and num_zeros, as the resample docstring gives them
        "LQ": (0.8381, 10.32, 18),
        "MQ": (0.9579, 10.32, 76),
        "HQ": (0.9569, 12.97, 92),
        "VHQ": (0.9557, 18.28, 126),
    }
    assert set(_QUALITY_RECIPES) == set(_QUALITY_PRESETS) == set(documented)
    for quality, (bits, passband_end) in _QUALITY_RECIPES.items():
        attenuation_db = (bits + 1) * 20.0 * math.log10(2.0)
        numtaps, beta = kaiserord(attenuation_db, 1.0 - passband_end)
        rolloff, kaiser_beta_, num_zeros = _quality_kaiser_params(quality)
        assert rolloff == 0.5 * (1.0 + passband_end)
        assert kaiser_beta_ == beta == kaiser_beta(attenuation_db)
        assert num_zeros == math.ceil(math.ceil((numtaps - 1) / 2) * rolloff)
        assert _QUALITY_PRESETS[quality] == (rolloff, kaiser_beta_, num_zeros)
        assert (round(rolloff, 4), round(kaiser_beta_, 2), num_zeros) == documented[quality]


# (orig_freq, new_freq, T, table): the dense reference against both paths.
# Table: p <= out_len. General: p > out_len, so each output makes its weights.
_REFERENCE_CASES = [
    (1, 4, 40, True),
    (3, 1, 90, True),
    (7, 10, 64, True),
    (147, 160, 200, True),
    (160, 147, 200, True),
    (44100, 48000, 60, False),  # reduces to (147, 160), out_len 66 < 160
    (999983, 1000003, 40, False),
    (1, 1, 50, True),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("orig_freq, new_freq, T, table", _REFERENCE_CASES)
def test_resample_matches_dense_reference(device, dtype, orig_freq, new_freq, T, table):
    rolloff, beta, num_zeros = _design("HQ")
    out_len = _default_out_len(T, orig_freq, new_freq)
    assert _uses_table(orig_freq, new_freq, rolloff, num_zeros, out_len) == table

    gen = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(3, T, device=device, dtype=dtype, generator=gen)
    actual = resample(x, orig_freq, new_freq, "HQ")
    expected = _reference_resample(x, orig_freq, new_freq, rolloff, beta, num_zeros, out_len)

    assert actual.dtype == dtype and actual.shape == (3, out_len)
    torch.testing.assert_close(actual, expected, **_TOLS[dtype])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "orig_freq, new_freq, T, kwargs",
    [
        (2, 3, 50, dict(quality="LQ", beta=5.0)),
        (5, 2, 70, dict(quality="MQ", num_zeros=7, rolloff=0.8)),
        (147, 160, 40, dict(quality=None, rolloff=1.0, beta=0.0, num_zeros=3)),
        (3, 4, 30, dict(quality="VHQ", num_zeros=5)),
    ],
)
def test_resample_design_overrides_match_reference(device, orig_freq, new_freq, T, kwargs):
    # Explicit arguments override the preset's values; beta = 0 is a boxcar window.
    rolloff, beta, num_zeros = _design(**kwargs)
    out_len = _default_out_len(T, orig_freq, new_freq)
    x = torch.randn(2, T, device=device, dtype=torch.float64)
    actual = resample(x, orig_freq, new_freq, **kwargs)
    expected = _reference_resample(x, orig_freq, new_freq, rolloff, beta, num_zeros, out_len)
    torch.testing.assert_close(actual, expected, **_TOLS[torch.float64])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(
    "orig_freq, new_freq, T",
    [(7, 10, 60), (147, 160, 250), (160, 147, 250), (2, 3, 80)],
)
def test_resample_table_and_general_paths_agree(device, dtype, orig_freq, new_freq, T):
    # The same outputs, evaluated through the table (out_len >= p) and through
    # the per-output weights (out_len < p): they differ only in rounding.
    rolloff, _, num_zeros = _design("HQ")
    p, _ = _rates(orig_freq, new_freq)
    short, long = p - 1, 3 * p
    assert not _uses_table(orig_freq, new_freq, rolloff, num_zeros, short)
    assert _uses_table(orig_freq, new_freq, rolloff, num_zeros, long)

    x = torch.randn(2, T, device=device, dtype=dtype)
    general = resample(x, orig_freq, new_freq, out_len=short)
    table = resample(x, orig_freq, new_freq, out_len=long)
    torch.testing.assert_close(general, table[:, :short], **_TOLS[dtype])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "orig_freq, new_freq, T, out_len",
    [(2, 3, 50, None), (5, 2, 60, None), (147, 160, 50, None), (160, 147, 50, None), (2, 3, 50, 7)],
)
def test_resample_adjoint_is_the_op_with_rates_swapped(device, orig_freq, new_freq, T, out_len):
    # <A x, y> = <x, A' y>, with A' the op with the two rates swapped, T
    # outputs and the same explicit scale: the identity the backward relies on.
    scale, rolloff, beta, num_zeros = 0.7, 0.9, 8.0, 6
    out_len = _default_out_len(T, orig_freq, new_freq) if out_len is None else out_len
    gen = torch.Generator(device=device).manual_seed(3)
    x = torch.randn(3, T, device=device, dtype=torch.float64, generator=gen)
    y = torch.randn(3, out_len, device=device, dtype=torch.float64, generator=gen)
    op = torch.ops.philtorch.resample
    ax = op(x, orig_freq, new_freq, scale, rolloff, beta, num_zeros, out_len)
    aty = op(y, new_freq, orig_freq, scale, rolloff, beta, num_zeros, T)
    torch.testing.assert_close((ax * y).sum(), (x * aty).sum(), **_TOLS[torch.float64])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("quality", ["LQ", "MQ", "HQ", "VHQ"])
@pytest.mark.parametrize("orig_freq, new_freq", [(2, 3), (1, 3)])
def test_resample_presets_reconstruct_passband_tone(device, quality, orig_freq, new_freq):
    # A tone at 30% of the Nyquist frequency, well inside every preset's
    # passband, is reconstructed away from the edges (where the filter's
    # support runs past the signal). The error stays below the preset's
    # design precision, 2 ** -(bits + 1). Upsampling, the images of the tone
    # are what is left in the error: they are in the stopband.
    bits, _ = _QUALITY_RECIPES[quality]
    T, margin = 2048, 300  # input samples, beyond the filter's half-width
    t_in = torch.arange(T, device=device, dtype=torch.float64)
    x = torch.sin(2.0 * torch.pi * 0.15 * t_in).unsqueeze(0)
    y = resample(x, orig_freq, new_freq, quality)
    m_lo, m_hi = -(-margin * new_freq // orig_freq), (T - margin) * new_freq // orig_freq
    t_out = torch.arange(m_lo, m_hi, device=device, dtype=torch.float64) * orig_freq / new_freq
    expected = torch.sin(2.0 * torch.pi * 0.15 * t_out).unsqueeze(0)
    rolloff, _, num_zeros = _design(quality)
    assert margin > _cutoff_and_support(orig_freq, new_freq, rolloff, num_zeros)[1]
    torch.testing.assert_close(y[:, m_lo:m_hi], expected, rtol=0, atol=2.0 ** -(bits + 1))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("quality", ["LQ", "MQ", "HQ", "VHQ"])
@pytest.mark.parametrize("orig_freq, new_freq", [(4, 1), (3, 2)])
@pytest.mark.parametrize("stop_edge_ratio", [1.0, 1.1, 1.5])
def test_resample_presets_attenuate_the_stopband(
    device, quality, orig_freq, new_freq, stop_edge_ratio
):
    # A tone above the new Nyquist frequency, from its stopband edge (1.0,
    # soxr's stopband_begin) outward, is removed before the decimation aliases
    # it: the interior RMS ratio is below the design attenuation
    # (bits + 1) * 20 log10(2) dB. The 1 dB of slack covers the RMS of a tone
    # with a fractional number of cycles in the measured stretch.
    bits, _ = _QUALITY_RECIPES[quality]
    attenuation_db = (bits + 1) * 20.0 * math.log10(2.0)
    rolloff, _, num_zeros = _design(quality)
    support = _cutoff_and_support(orig_freq, new_freq, rolloff, num_zeros)[1]
    T = int(4 * support) + 400
    freq = stop_edge_ratio * 0.5 * new_freq / orig_freq  # cycles per input sample
    t = torch.arange(T, device=device, dtype=torch.float64)
    x = torch.sin(2.0 * torch.pi * freq * t + 0.3).unsqueeze(0)
    y = resample(x, orig_freq, new_freq, quality)

    lo = int(support) + 5  # inputs whose full support is inside the signal
    hi = T - lo
    m_lo, m_hi = -(-lo * new_freq // orig_freq), hi * new_freq // orig_freq
    ratio = y[:, m_lo:m_hi].pow(2).mean().sqrt() / x[:, lo:hi].pow(2).mean().sqrt()
    assert 20.0 * math.log10(ratio.item()) < -(attenuation_db - 1.0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("quality", ["LQ", "MQ", "HQ", "VHQ"])
@pytest.mark.parametrize("orig_freq, new_freq", [(1, 3), (2, 3)])
def test_resample_presets_reject_the_images_when_upsampling(device, quality, orig_freq, new_freq):
    # Upsampling, a tone at 90% of the preset's passband edge has its image at
    # 1 - f cycles per input sample, just beyond the stopband edge (the input
    # Nyquist frequency): what the filter leaves of it is the error against
    # the ideal tone, which stays below the design precision 2 ** -(bits + 1).
    bits, passband_end = _QUALITY_RECIPES[quality]
    freq = 0.5 * 0.9 * passband_end  # cycles per input sample
    assert 1.0 - freq > 0.5
    rolloff, _, num_zeros = _design(quality)
    support = _cutoff_and_support(orig_freq, new_freq, rolloff, num_zeros)[1]
    T = int(4 * support) + 400
    t_in = torch.arange(T, device=device, dtype=torch.float64)
    x = torch.sin(2.0 * torch.pi * freq * t_in + 0.3).unsqueeze(0)
    y = resample(x, orig_freq, new_freq, quality)

    lo = int(support) + 5
    m_lo, m_hi = -(-lo * new_freq // orig_freq), (T - lo) * new_freq // orig_freq
    t_out = torch.arange(m_lo, m_hi, device=device, dtype=torch.float64) * orig_freq / new_freq
    expected = torch.sin(2.0 * torch.pi * freq * t_out + 0.3).unsqueeze(0)
    torch.testing.assert_close(y[:, m_lo:m_hi], expected, rtol=0, atol=2.0 ** -(bits + 1))


@pytest.mark.parametrize("orig_freq, new_freq, T", [(2, 3, 200), (5, 2, 300), (147, 160, 60)])
def test_resample_matches_scipy_resample_poly(orig_freq, new_freq, T):
    # resample_poly with the filter at the upsampled rate as `window`: h[j] =
    # c sinc(c j / p) K(j / p) / p, for j = -L .. L, since resample_poly
    # multiplies the window by p. It aligns output 0 with input 0 by padding
    # that odd-length, centred filter, and zero-pads the signal, as we do. Both
    # the table path (T = 200, 300) and the general one (T = 60) are covered.
    # resample_poly returns a copy for equal rates, so (1, 1) is left out.
    rolloff, beta, num_zeros = _design("LQ")
    p, q = _rates(orig_freq, new_freq)
    c, support = _cutoff_and_support(orig_freq, new_freq, rolloff, num_zeros)
    half = math.ceil(support * p) + 1
    delta = torch.arange(-half, half + 1, dtype=torch.float64) / p
    h = _kaiser_sinc(delta, c, support, beta) / p

    x = torch.randn(2, T, dtype=torch.float64)
    expected = resample_poly(x.numpy(), p, q, axis=1, window=h.numpy())
    actual = resample(x, orig_freq, new_freq, "LQ")
    torch.testing.assert_close(actual, torch.from_numpy(expected), **_TOLS[torch.float64])


# (orig_freq, new_freq, T, out_len): table forward and adjoint; the same with a
# short out_len, so that the forward is general; and a general pair both ways.
_GRAD_CASES = [(2, 3, 24, None), (5, 2, 30, None), (2, 3, 24, 2), (147, 160, 20, None)]


def _grad_fn(orig_freq, new_freq, out_len):
    def fn(x):
        return resample(x, orig_freq, new_freq, out_len=out_len, **_GRAD_DESIGN)

    return fn


def _resample_transpose(y, orig_freq, new_freq, T):
    """A' y for _grad_fn's operator: the op with the rates swapped, T outputs.

    Its scale is c = rolloff * min(1, p / q), as resample passes it.
    """
    rolloff, beta, num_zeros = _design(**_GRAD_DESIGN)
    p, q = _rates(orig_freq, new_freq)
    scale = rolloff * min(1.0, p / q)
    return torch.ops.philtorch.resample(y, new_freq, orig_freq, scale, rolloff, beta, num_zeros, T)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq, T, out_len", _GRAD_CASES)
def test_resample_gradcheck_and_gradgradcheck(device, orig_freq, new_freq, T, out_len):
    x = torch.randn(2, T, device=device, dtype=torch.float64, requires_grad=True)
    fn = _grad_fn(orig_freq, new_freq, out_len)
    assert torch.autograd.gradcheck(fn, (x,))
    assert torch.autograd.gradgradcheck(fn, (x,))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq, T, out_len", _GRAD_CASES)
def test_resample_forward_mode_and_forward_over_reverse(device, orig_freq, new_freq, T, out_len):
    fn = _grad_fn(orig_freq, new_freq, out_len)
    x = torch.randn(2, T, device=device, dtype=torch.float64)
    dx = torch.randn_like(x)

    # The operator is linear: its tangent is itself applied to the tangent.
    y, dy = torch.func.jvp(fn, (x,), (dx,))
    torch.testing.assert_close(y, fn(x))
    torch.testing.assert_close(dy, fn(dx))

    # grad of f(x) = sum(A x ** 2) is 2 A' A x, linear in x, so the tangent of
    # the gradient is the gradient at the tangent.
    def loss(t):
        return fn(t).pow(2).sum()

    grad_fn = torch.func.grad(loss)
    _, tangent = torch.func.jvp(grad_fn, (x,), (dx,))
    torch.testing.assert_close(tangent, grad_fn(dx))
    expected = 2.0 * _resample_transpose(fn(dx), orig_freq, new_freq, T)
    torch.testing.assert_close(tangent, expected)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq, T, out_len", _GRAD_CASES)
def test_resample_vmap_folds_into_the_batch(device, orig_freq, new_freq, T, out_len):
    fn = _grad_fn(orig_freq, new_freq, out_len)
    x = torch.randn(3, 2, T, device=device, dtype=torch.float64)
    vy = torch.func.vmap(fn)(x)
    torch.testing.assert_close(vy, fn(x.reshape(6, T)).reshape(3, 2, -1))
    # The vmapped dimension may sit anywhere.
    vy = torch.func.vmap(fn, in_dims=1)(x.transpose(0, 1))
    torch.testing.assert_close(vy, fn(x.reshape(6, T)).reshape(3, 2, -1))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq, T, out_len", _GRAD_CASES)
def test_resample_vmap_of_grad_gives_per_sample_gradients(device, orig_freq, new_freq, T, out_len):
    fn = _grad_fn(orig_freq, new_freq, out_len)
    x = torch.randn(3, T, device=device, dtype=torch.float64)

    def loss(row):
        return fn(row.unsqueeze(0)).pow(2).sum()

    per_sample = torch.func.vmap(torch.func.grad(loss))(x)
    for row, grad in zip(x, per_sample):
        row = row.clone().requires_grad_()
        (expected,) = torch.autograd.grad(loss(row), row)
        torch.testing.assert_close(grad, expected)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("out_len", [20, 2])  # table and general
def test_resample_opcheck(device, requires_grad, out_len):
    x = torch.randn(2, 12, device=device, dtype=torch.float64, requires_grad=requires_grad)
    torch.library.opcheck(torch.ops.philtorch.resample, (x, 2, 3, 0.9, 0.9, 10.0, 5, out_len))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq, T", [(2, 3, 30), (5, 2, 40), (147, 160, 20)])
def test_resample_compiled_gradient_matches_eager(monkeypatch, device, orig_freq, new_freq, T):
    # The on-disk compile caches replay old backwards: bypass them.
    monkeypatch.setattr(torch._inductor.config, "fx_graph_cache", False)
    monkeypatch.setattr(torch._functorch.config, "enable_autograd_cache", False)
    torch._dynamo.reset()

    def loss(t):
        y = resample(t, orig_freq, new_freq, "LQ")
        return (
            (y * torch.linspace(-1.0, 1.0, y.size(1), device=t.device, dtype=t.dtype)).pow(2).sum()
        )

    x = torch.randn(2, T, device=device, dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(loss, fullgraph=True)
    (grad_compiled,) = torch.autograd.grad(compiled(x), x)
    (grad_eager,) = torch.autograd.grad(loss(x), x)
    torch.testing.assert_close(grad_compiled, grad_eager)
    torch.testing.assert_close(compiled(x), loss(x))


@pytest.mark.parametrize("device", _DEVICES)
def test_resample_empty_inputs_and_outputs(device):
    assert resample(torch.empty(0, 16, device=device), 2, 3).shape == (0, 24)
    assert resample(torch.empty(2, 16, device=device), 2, 3, out_len=0).shape == (2, 0)
    zero_time = resample(torch.empty(2, 0, device=device), 2, 3, out_len=5)
    assert zero_time.shape == (2, 5) and (zero_time == 0).all()
    assert resample(torch.empty(2, 0, device=device), 2, 3).shape == (2, 0)

    # The gradient of an empty output is zeros of x's shape.
    x = torch.randn(2, 16, device=device, requires_grad=True)
    (grad,) = torch.autograd.grad(resample(x, 2, 3, out_len=0).sum(), x)
    assert grad.shape == x.shape and (grad == 0).all()


@pytest.mark.parametrize(
    "T, orig_freq, new_freq, out_len",
    [(441, 44100, 48000, 480), (7, 2, 3, 11), (7, 3, 2, 5), (7, 1, 1, 7), (6, 3, 2, 4)],
)
def test_resample_default_out_len_is_the_exact_ceiling(T, orig_freq, new_freq, out_len):
    # ceil(T * new_freq / orig_freq), in integers: 7 * 3 / 2 = 10.5 rounds up,
    # where Python's round would go to the even 10.
    assert resample(torch.zeros(1, T), orig_freq, new_freq).shape == (1, out_len)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq", [(2, 3), (147, 160)])
def test_resample_out_len_beyond_the_default(device, orig_freq, new_freq):
    T = 30
    x = torch.randn(2, T, device=device, dtype=torch.float64)
    default = resample(x, orig_freq, new_freq)
    assert default.size(1) == _default_out_len(T, orig_freq, new_freq)

    longer = resample(x, orig_freq, new_freq, out_len=default.size(1) + 700)
    torch.testing.assert_close(longer[:, : default.size(1)], default, **_TOLS[torch.float64])
    # Past the filter's reach of the last input, there is nothing to read.
    rolloff, _, num_zeros = _design("HQ")
    reach = _cutoff_and_support(orig_freq, new_freq, rolloff, num_zeros)[1]
    first_zero = math.ceil((T - 1 + reach) * new_freq / orig_freq) + 1
    assert first_zero < longer.size(1)
    assert (longer[:, first_zero:] == 0).all()


@pytest.mark.parametrize("device", _DEVICES)
def test_resample_non_contiguous_input(device):
    x = torch.randn(40, 3, device=device, dtype=torch.float64).mT
    assert not x.is_contiguous()
    torch.testing.assert_close(resample(x, 2, 3), resample(x.contiguous(), 2, 3))
    x = torch.randn(3, 80, device=device, dtype=torch.float64)[:, ::2]
    torch.testing.assert_close(resample(x, 147, 160), resample(x.contiguous(), 147, 160))


@pytest.mark.parametrize("device", _DEVICES)
def test_resample_rates_are_reduced_by_their_gcd(device):
    x = torch.randn(2, 40, device=device, dtype=torch.float64)
    torch.testing.assert_close(resample(x, 6, 4), resample(x, 3, 2), rtol=0, atol=0)
    torch.testing.assert_close(resample(x, 44100, 48000), resample(x, 147, 160), rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_resample_equal_rates_low_pass_instead_of_copying(device):
    # At orig_freq == new_freq the filter still cuts off at rolloff times the
    # Nyquist frequency (unlike scipy.signal.resample_poly, which copies).
    gen = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn(1, 400, device=device, dtype=torch.float64, generator=gen)
    assert (resample(noise, 1, 1) - noise)[:, 100:-100].abs().max() > 0.1

    t = torch.arange(400, device=device, dtype=torch.float64)
    tone = torch.sin(2.0 * torch.pi * 0.02 * t).unsqueeze(0)  # 4% of Nyquist
    torch.testing.assert_close(
        resample(tone, 5, 5)[:, 100:-100], tone[:, 100:-100], rtol=0, atol=1e-6
    )


@pytest.mark.cuda
@pytest.mark.parametrize("orig_freq, new_freq", [(2, 3), (147, 160)])
def test_resample_cuda_large_batch(orig_freq, new_freq):
    # More rows than a CUDA grid dimension holds (65535 blocks): the launch
    # is chunked. 70000 rows of 8 samples, against the CPU kernel.
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(70000, 8, dtype=torch.float64, generator=gen)
    expected = resample(x, orig_freq, new_freq)
    actual = resample(x.cuda(), orig_freq, new_freq)
    torch.testing.assert_close(actual.cpu(), expected, **_TOLS[torch.float64])


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("T", [48, 5000])
def test_resample_cuda_graph_capture(dtype, T):
    # No host-to-device copy in the launch path: a captured call replays on
    # new input. T = 48 takes the general path at 44100 -> 48000, 5000 the
    # table path.
    x = torch.randn(3, T, dtype=dtype, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        resample(x, 44100, 48000)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = resample(x, 44100, 48000)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.testing.assert_close(y, resample(x, 44100, 48000), rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(orig_freq=True, new_freq=2), "orig_freq must be a positive integer, got True"),
        (dict(orig_freq=2, new_freq=False), "new_freq must be a positive integer, got False"),
        (dict(orig_freq=2.0, new_freq=3), "orig_freq must be a positive integer, got 2.0"),
        (dict(orig_freq=2, new_freq=1.5), "new_freq must be a positive integer, got 1.5"),
        (dict(orig_freq="2", new_freq=3), "orig_freq must be a positive integer, got '2'"),
        (dict(orig_freq=0, new_freq=3), "orig_freq must be a positive integer, got 0"),
        (dict(orig_freq=2, new_freq=0), "new_freq must be a positive integer, got 0"),
        (dict(orig_freq=-2, new_freq=3), "orig_freq must be a positive integer, got -2"),
        (dict(orig_freq=2, new_freq=-3), "new_freq must be a positive integer, got -3"),
        (dict(quality="unknown"), "quality must be one of .* or None, got 'unknown'"),
        (dict(quality="hq"), "quality must be one of"),
        (dict(quality=3), "quality must be one of .* or None, got 3"),
        (dict(quality=None), "when quality is None, rolloff, beta and num_zeros must all be"),
        (dict(quality=None, rolloff=0.9, beta=10.0), "when quality is None"),
        (dict(quality=None, rolloff=0.9, num_zeros=4), "when quality is None"),
        (dict(quality=None, beta=10.0, num_zeros=4), "when quality is None"),
        (dict(rolloff=0.0), r"rolloff must be in \(0, 1\], got 0.0"),
        (dict(rolloff=1.5), r"rolloff must be in \(0, 1\], got 1.5"),
        (dict(rolloff=-0.5), r"rolloff must be in \(0, 1\], got -0.5"),
        (dict(rolloff=float("nan")), "rolloff must be finite, got nan"),
        (dict(rolloff=float("inf")), "rolloff must be finite, got inf"),
        (dict(rolloff=True), "rolloff must be a real number, got True"),
        (dict(rolloff="0.9"), "rolloff must be a real number, got '0.9'"),
        (dict(beta=-1.0), "beta must be non-negative, got -1.0"),
        (dict(beta=float("nan")), "beta must be finite, got nan"),
        (dict(beta=float("inf")), "beta must be finite, got inf"),
        (dict(beta=True), "beta must be a real number, got True"),
        (dict(num_zeros=0), "num_zeros must be a positive integer, got 0"),
        (dict(num_zeros=-3), "num_zeros must be a positive integer, got -3"),
        (dict(num_zeros=True), "num_zeros must be a positive integer, got True"),
        (dict(num_zeros=1.5), "num_zeros must be a positive integer, got 1.5"),
        (dict(out_len=-1), "out_len must be a non-negative integer, got -1"),
        (dict(out_len=True), "out_len must be a non-negative integer, got True"),
        (dict(out_len=2.0), "out_len must be a non-negative integer, got 2.0"),
    ],
)
def test_resample_invalid_arguments(kwargs, match):
    kwargs = {"orig_freq": 2, "new_freq": 3} | kwargs
    with pytest.raises(ValueError, match=match):
        resample(torch.randn(2, 16), **kwargs)


def test_resample_invalid_signals():
    with pytest.raises(ValueError, match=r"x must be 2D \(B, T\), got 1D"):
        resample(torch.randn(16), 2, 3)
    with pytest.raises(ValueError, match=r"x must be 2D \(B, T\), got 3D"):
        resample(torch.randn(2, 2, 16), 2, 3)
    with pytest.raises(ValueError, match="x must be a floating-point tensor, got torch.int64"):
        resample(torch.ones(2, 16, dtype=torch.int64), 2, 3)


# The checks below are the native kernels': ValueErrors from TORCH_CHECK_VALUE.


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("out_len", [30, 2], ids=["table", "general"])
def test_resample_beta_limit(device, dtype, out_len):
    # The Kaiser window's series needs more terms than the kernels hold from
    # beta of about 346 on: the limit is the same on every path and dtype.
    x = torch.randn(2, 16, device=device, dtype=dtype)
    design = dict(quality=None, rolloff=0.9, num_zeros=4, out_len=out_len)
    with pytest.raises(ValueError, match="beta .*too large"):
        resample(x, 2, 3, beta=1000.0, **design)
    assert resample(x, 2, 3, beta=300.0, **design).isfinite().all()


@pytest.mark.parametrize("device", _DEVICES)
def test_resample_rate_product_limit(device):
    # After the gcd, the product of the rates must stay below 2 ** 62.
    x = torch.randn(2, 16, device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="too large"):
        resample(x, 2**40, 2**40 + 1)


@pytest.mark.parametrize("device", _DEVICES)
def test_resample_taps_limit(device):
    # A downsampling ratio so low that the filter has more than 2 ** 30 taps.
    x = torch.randn(2, 16, device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="taps for these rates, rolloff and num_zeros, more than"):
        resample(x, 2**31, 1)


@pytest.mark.parametrize("device", _DEVICES)
def test_resample_op_rejects_bad_arguments(device):
    x = torch.randn(2, 16, device=device, dtype=torch.float64)
    op = torch.ops.philtorch.resample
    with pytest.raises(ValueError, match="orig_freq"):
        op(x, 0, 3, 1.0, 0.9, 5.0, 4, 8)
    with pytest.raises(ValueError, match="scale must be finite"):
        op(x, 2, 3, float("nan"), 0.9, 5.0, 4, 8)
