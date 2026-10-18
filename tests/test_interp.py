import math

import pytest
import torch
from scipy.interpolate import CubicSpline
from scipy.signal import cspline1d, cspline1d_eval, kaiser_beta, kaiserord, resample_poly

from philtorch.lti.interp import (
    _QUALITY_PRESETS,
    _QUALITY_RECIPES,
    _kaiser_design,
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
# Tolerances against the float64 reference; float16 and bfloat16 accumulate in
# float32 and round once.
_TOLS = {
    torch.float64: dict(rtol=1e-12, atol=1e-12),
    torch.float32: dict(rtol=5e-6, atol=5e-6),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
    torch.bfloat16: dict(rtol=8e-3, atol=8e-3),
}


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
        rolloff, kaiser_beta_, num_zeros = _kaiser_design(bits, passband_end)
        assert rolloff == 0.5 * (1.0 + passband_end)
        assert kaiser_beta_ == beta == kaiser_beta(attenuation_db)
        assert num_zeros == math.ceil(math.ceil((numtaps - 1) / 2) * rolloff)
        assert _QUALITY_PRESETS[quality] == (rolloff, kaiser_beta_, num_zeros)
        assert (round(rolloff, 4), round(kaiser_beta_, 2), num_zeros) == documented[quality]


# (orig_freq, new_freq, T): T = 64 at (7, 10) is shorter than the filter's
# width; (44100, 48000) reduces to (147, 160) and (1000, 1007) to itself, with
# more phases than outputs (p > out_len).
_REFERENCE_CASES = [
    (1, 4, 40),
    (3, 1, 90),
    (7, 10, 64),
    (147, 160, 200),
    (160, 147, 200),
    (44100, 48000, 60),
    (1000, 1007, 40),
    (1, 1, 50),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("orig_freq, new_freq, T", _REFERENCE_CASES)
def test_resample_matches_dense_reference(device, dtype, orig_freq, new_freq, T):
    rolloff, beta, num_zeros = _design("HQ")
    out_len = _default_out_len(T, orig_freq, new_freq)

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
@pytest.mark.parametrize("quality", ["LQ", "MQ", "HQ", "VHQ"])
@pytest.mark.parametrize("orig_freq, new_freq", [(2, 3), (1, 3)])
def test_resample_presets_reconstruct_passband_tone(device, quality, orig_freq, new_freq):
    # A tone at 30% of Nyquist, inside every passband, is reconstructed away
    # from the edges to the preset's precision, 2 ** -(bits + 1).
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
@pytest.mark.parametrize("stop_edge_ratio", [1.0, 1.5])
def test_resample_presets_attenuate_the_stopband(
    device, quality, orig_freq, new_freq, stop_edge_ratio
):
    # A tone in the stopband (from soxr's stopband_begin, 1.0, outward) is
    # attenuated by the design's (bits + 1) * 20 log10(2) dB, with 1 dB of
    # slack for the RMS of a fractional number of cycles.
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
    # Upsampling a tone at 90% of the passband edge, its image, just past the
    # stopband edge, is what remains of the error: below 2 ** -(bits + 1).
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


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq, T", [(2, 3, 200), (5, 2, 300), (147, 160, 60)])
def test_resample_matches_scipy_resample_poly(device, orig_freq, new_freq, T):
    # resample_poly, given our filter at the upsampled rate as `window` (it
    # multiplies the window by p, hence the / p), centres and zero-pads as we
    # do. It copies at equal rates, so (1, 1) is left out.
    rolloff, beta, num_zeros = _design("LQ")
    p, q = _rates(orig_freq, new_freq)
    c, support = _cutoff_and_support(orig_freq, new_freq, rolloff, num_zeros)
    half = math.ceil(support * p) + 1
    delta = torch.arange(-half, half + 1, dtype=torch.float64) / p
    h = _kaiser_sinc(delta, c, support, beta) / p

    x = torch.randn(2, T, dtype=torch.float64)
    expected = resample_poly(x.numpy(), p, q, axis=1, window=h.numpy())
    actual = resample(x.to(device), orig_freq, new_freq, "LQ").cpu()
    torch.testing.assert_close(actual, torch.from_numpy(expected), **_TOLS[torch.float64])


# (orig_freq, new_freq, T): a pair of rates each way, fewer outputs than phases
# (T = 1) and a rate with many phases.
_GRAD_CASES = [(2, 3, 24), (5, 2, 30), (2, 3, 1), (147, 160, 20)]


def _grad_fn(orig_freq, new_freq):
    """resample with a short filter, for the gradient checks."""

    def fn(x, rolloff=0.85, beta=9.0):
        return resample(x, orig_freq, new_freq, None, rolloff=rolloff, beta=beta, num_zeros=4)

    return fn


def _params(device, dtype=torch.float64, requires_grad=False):
    """The default rolloff and beta of _grad_fn, as 0-D tensors."""
    return [
        torch.tensor(v, device=device, dtype=dtype, requires_grad=requires_grad)
        for v in (0.85, 9.0)
    ]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("tensor_params", [False, True], ids=["x", "x-rolloff-beta"])
@pytest.mark.parametrize("orig_freq, new_freq, T", _GRAD_CASES)
def test_resample_gradcheck_and_gradgradcheck(device, tensor_params, orig_freq, new_freq, T):
    # Float parameters use the cached table, with only x differentiable; tensor
    # parameters that require grad build the table in the graph.
    x = torch.randn(2, T, device=device, dtype=torch.float64, requires_grad=True)
    inputs = (x, *_params(device, requires_grad=True)) if tensor_params else (x,)
    fn = _grad_fn(orig_freq, new_freq)
    assert torch.autograd.gradcheck(fn, inputs)
    assert torch.autograd.gradgradcheck(fn, inputs)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq, T", _GRAD_CASES)
def test_resample_forward_mode_and_forward_over_reverse(device, orig_freq, new_freq, T):
    fn = _grad_fn(orig_freq, new_freq)
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


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq, T", _GRAD_CASES)
def test_resample_vmap_folds_into_the_batch(device, orig_freq, new_freq, T):
    fn = _grad_fn(orig_freq, new_freq)
    x = torch.randn(3, 2, T, device=device, dtype=torch.float64)
    vy = torch.func.vmap(fn)(x)
    torch.testing.assert_close(vy, fn(x.reshape(6, T)).reshape(3, 2, -1))
    # The vmapped dimension may sit anywhere.
    vy = torch.func.vmap(fn, in_dims=1)(x.transpose(0, 1))
    torch.testing.assert_close(vy, fn(x.reshape(6, T)).reshape(3, 2, -1))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq, T", _GRAD_CASES)
def test_resample_vmap_of_grad_gives_per_sample_gradients(device, orig_freq, new_freq, T):
    fn = _grad_fn(orig_freq, new_freq)
    x = torch.randn(3, T, device=device, dtype=torch.float64)

    def loss(row):
        return fn(row.unsqueeze(0)).pow(2).sum()

    per_sample = torch.func.vmap(torch.func.grad(loss))(x)
    for row, grad in zip(x, per_sample):
        row = row.clone().requires_grad_()
        (expected,) = torch.autograd.grad(loss(row), row)
        torch.testing.assert_close(grad, expected)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("tensor_beta", [False, True])
@pytest.mark.parametrize("orig_freq, new_freq, T", [(2, 3, 30), (5, 2, 40), (147, 160, 20)])
def test_resample_compiled_gradient_matches_eager(device, tensor_beta, orig_freq, new_freq, T):
    # aot_eager, which needs no C++ compiler (Windows CI has none for Inductor)
    # and has no on-disk cache to replay an old backward. A tensor rolloff would
    # need a graph break, so only beta is a tensor.
    torch._dynamo.reset()
    x = torch.randn(2, T, device=device, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(9.0, device=device, dtype=torch.float64, requires_grad=True)
    beta, inputs = (beta, (x, beta)) if tensor_beta else (None, (x,))

    def loss(t, b):
        y = resample(t, orig_freq, new_freq, "LQ", beta=b)
        return (
            (y * torch.linspace(-1.0, 1.0, y.size(1), device=t.device, dtype=t.dtype)).pow(2).sum()
        )

    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True)
    actual = torch.autograd.grad(compiled(x, beta), inputs)
    expected = torch.autograd.grad(loss(x, beta), inputs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)
    torch.testing.assert_close(compiled(x, beta), loss(x, beta))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("orig_freq, new_freq, T", _GRAD_CASES)
def test_resample_tensor_parameters_match_floats(device, dtype, orig_freq, new_freq, T):
    x = torch.randn(2, T, device=device, dtype=dtype)
    rolloff, beta = _params(device)
    fn = _grad_fn(orig_freq, new_freq)
    expected = fn(x, 0.85, 9.0)
    # A negative tensor beta acts as |beta|.
    negative = torch.tensor(-9.0, device=device)
    for args in [(rolloff, beta), (rolloff, 9.0), (0.85, beta), (0.85, negative)]:
        actual = fn(x, *args)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, **_TOLS[dtype])


@pytest.mark.cuda
@pytest.mark.parametrize("orig_freq, new_freq, T", _GRAD_CASES)
def test_resample_cuda_parameter_gradients_match_cpu_in_float32(orig_freq, new_freq, T):
    # gradcheck covers float64 on CUDA; this compares CUDA with CPU in float32.
    fn = _grad_fn(orig_freq, new_freq)
    x = torch.randn(2, T)

    def grads(device):
        inputs = [x.to(device).requires_grad_(), *_params(device, torch.float32, True)]
        y = fn(*inputs)
        weights = torch.linspace(-1.0, 1.0, y.size(1), device=device)
        return [g.cpu() for g in torch.autograd.grad((y * weights).pow(2).sum(), inputs)]

    for actual, expected in zip(grads("cuda"), grads("cpu")):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("orig_freq, new_freq, T", _GRAD_CASES)
def test_resample_forward_mode_in_the_parameters(device, orig_freq, new_freq, T):
    fn = _grad_fn(orig_freq, new_freq)
    x = torch.randn(2, T, device=device, dtype=torch.float64)
    dx = torch.randn_like(x)
    rolloff, beta = _params(device)
    tangents = (dx, *(torch.tensor(t, device=device, dtype=torch.float64) for t in (0.01, 0.7)))

    # The jvp in all three arguments, and in beta alone (x and rolloff then have
    # no tangent), against central differences.
    eps = 1e-6
    calls = [
        (fn, (x, rolloff, beta), tangents),
        (lambda b: fn(x, rolloff, b), (beta,), tangents[2:]),
    ]
    for f, point, tangent in calls:
        _, dy = torch.func.jvp(f, point, tangent)
        plus = f(*(p + eps * t for p, t in zip(point, tangent)))
        minus = f(*(p - eps * t for p, t in zip(point, tangent)))
        torch.testing.assert_close(dy, (plus - minus) / (2 * eps), rtol=1e-6, atol=1e-6)


@pytest.mark.cuda
def test_resample_vmap_over_the_parameters_is_not_supported():
    x = torch.randn(3, 2, 12, device="cuda", dtype=torch.float64)
    betas = torch.tensor([6.0, 7.0, 8.0], device="cuda", dtype=torch.float64)
    rolloff = torch.tensor(0.85, device="cuda", dtype=torch.float64)
    fn = _grad_fn(2, 3)

    with pytest.raises(NotImplementedError, match="vmap over rolloff or beta"):
        torch.func.vmap(fn, in_dims=(0, None, 0))(x, rolloff, betas)
    # Over x alone, tensor parameters are fine.
    vy = torch.func.vmap(fn, in_dims=(0, None, None))(x, rolloff, betas[1])
    torch.testing.assert_close(vy, fn(x.reshape(6, 12), rolloff, betas[1]).reshape(3, 2, -1))


@pytest.mark.parametrize("device", _DEVICES)
def test_resample_trains_after_an_inference_mode_call(device):
    # The first call caches the filter, here in inference mode, and a later
    # call must still save it for backward. Rates no other test uses, so that
    # the cache is cold.
    x = torch.randn(2, 40, device=device, dtype=torch.float64, requires_grad=True)
    with torch.inference_mode():
        resample(x, 13, 17)
    (grad,) = torch.autograd.grad(resample(x, 13, 17).sum(), x)
    assert grad.shape == x.shape


@pytest.mark.parametrize("device", _DEVICES)
def test_resample_empty_inputs_and_outputs(device):
    assert resample(torch.empty(0, 16, device=device), 2, 3).shape == (0, 24)

    # An empty signal gives an empty output, which stays in the graph of x and
    # of tensor parameters, with zero gradients.
    x = torch.empty(2, 0, device=device, dtype=torch.float64, requires_grad=True)
    params = _params(device, requires_grad=True)
    y = _grad_fn(2, 3)(x, *params)
    assert y.shape == (2, 0)
    grad_x, *grad_params = torch.autograd.grad(y.sum(), (x, *params))
    assert grad_x.shape == x.shape and all(g == 0 for g in grad_params)


@pytest.mark.parametrize(
    "T, orig_freq, new_freq, out_len",
    [(441, 44100, 48000, 480), (7, 2, 3, 11), (7, 3, 2, 5), (7, 1, 1, 7), (6, 3, 2, 4)],
)
def test_resample_output_length_is_the_exact_ceiling(T, orig_freq, new_freq, out_len):
    # ceil(T * new_freq / orig_freq), in integers: 7 * 3 / 2 = 10.5 rounds up,
    # where Python's round would go to the even 10.
    assert resample(torch.zeros(1, T), orig_freq, new_freq).shape == (1, out_len)


@pytest.mark.parametrize("device", _DEVICES)
def test_resample_non_contiguous_input(device):
    x = torch.randn(40, 3, device=device, dtype=torch.float64).mT
    assert not x.is_contiguous()
    torch.testing.assert_close(resample(x, 2, 3), resample(x.contiguous(), 2, 3))
    x = torch.randn(3, 80, device=device, dtype=torch.float64)[:, ::2]
    torch.testing.assert_close(resample(x, 147, 160), resample(x.contiguous(), 147, 160))


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
    # 2**19 rows, more than the 65535 blocks of 8 rows a grid's y dimension
    # holds: the launch is split in two. The rows on either side of the split,
    # against the CPU convolution.
    x = torch.randn(2**19, 4, dtype=torch.float64, device="cuda")
    rows = slice(65535 * 8 - 8, None)
    expected = resample(x[rows].cpu(), orig_freq, new_freq)
    actual = resample(x, orig_freq, new_freq)[rows]
    torch.testing.assert_close(actual.cpu(), expected, **_TOLS[torch.float64])


@pytest.mark.cuda
def test_resample_cuda_longest_filter():
    # 6000 -> 1 in HQ: 1,153,762 taps, within the bound on the filter.
    x = torch.randn(1, 12000, dtype=torch.float64)
    expected = resample(x, 6000, 1)
    actual = resample(x.cuda(), 6000, 1)
    torch.testing.assert_close(actual.cpu(), expected, **_TOLS[torch.float64])


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("T", [48, 5000])
def test_resample_cuda_graph_capture(dtype, T):
    # No host-to-device copy in the launch path once the table is cached on
    # the device: the warm-up call caches it, and a captured call replays on
    # new input.
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


@pytest.mark.cuda
@pytest.mark.parametrize("p, q, out_len", [(3, 2, 20), (3, 2, 2), (2, 3, 7), (1, 1, 5)])
def test_resample_op_with_any_table(p, q, out_len):
    # A random table, unlike resample's, has nonzero end taps.
    table = torch.randn(p, 6, device="cuda", dtype=torch.float64, requires_grad=True)
    x = torch.randn(2, 12, device="cuda", dtype=torch.float64, requires_grad=True)
    op = torch.ops.philtorch.resample
    torch.library.opcheck(op, (x, table, q, out_len))
    assert torch.autograd.gradcheck(lambda x, t: op(x, t, q, out_len), (x, table))
    assert torch.autograd.gradgradcheck(lambda x, t: op(x, t, q, out_len), (x, table))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(orig_freq=2.0), "orig_freq must be a positive integer, got 2.0"),
        (dict(new_freq=0), "new_freq must be a positive integer, got 0"),
        (dict(quality="unknown"), "quality must be one of .* or None, got 'unknown'"),
        (dict(quality=None, rolloff=0.9), "when quality is None, rolloff, beta and num_zeros"),
        (dict(rolloff=1.5), r"rolloff must be in \(0, 1\], got 1.5"),
        (dict(rolloff="0.9"), "rolloff must be a real number, got '0.9'"),
        (dict(rolloff=torch.tensor(1.5)), r"rolloff must be in \(0, 1\], got 1.5"),
        (dict(rolloff=torch.ones(1)), "rolloff must be a real number or a 0-D floating-point"),
        (dict(beta=torch.ones(2)), "beta must be a real number or a 0-D floating-point"),
        (dict(beta=torch.tensor(1)), "beta must be a real number or a 0-D floating-point"),
        (dict(rolloff=float("nan")), "rolloff must be finite, got nan"),
        (dict(beta=-1.0), "beta must be non-negative, got -1.0"),
        (dict(num_zeros=0), "num_zeros must be a positive integer, got 0"),
        (dict(orig_freq=999983, new_freq=1000003), "limit_denominator"),
        # The table alone (5001 phases of 196 taps) would fit; the dense kernel,
        # 5001 x (5000 + 196), does not.
        (dict(orig_freq=5000, new_freq=5001), "limit_denominator"),
    ],
)
def test_resample_invalid_arguments(kwargs, match):
    kwargs = {"orig_freq": 2, "new_freq": 3} | kwargs
    with pytest.raises(ValueError, match=match):
        resample(torch.randn(2, 16), **kwargs)


def test_resample_invalid_signals():
    with pytest.raises(ValueError, match=r"x must be 2D \(B, T\), got 1D"):
        resample(torch.randn(16), 2, 3)
    with pytest.raises(ValueError, match="x must be a floating-point tensor, got torch.int64"):
        resample(torch.ones(2, 16, dtype=torch.int64), 2, 3)
