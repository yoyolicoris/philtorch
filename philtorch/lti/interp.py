"""Signal interpolation and band-limited resampling."""

import functools
import math
import numbers
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .._ops import (
    _conv_input,
    _resample_gradient,
    _resample_setup_context,
    _tap_columns,
    _use_on_current_stream,
)
from .delay import _as_int
from .recur import linear_recurrence

# (precision in bits, passband_end) of libsoxr 0.1.3's presets, as its
# soxr_quality_spec returns them (LQ's is 1385 / 2048): not ported, it is LGPL.
_QUALITY_RECIPES: dict[str, tuple[int, float]] = {
    "LQ": (16, 0.67626953125),
    "MQ": (16, 0.9158810491765671),
    "HQ": (20, 0.9137436532733066),
    "VHQ": (28, 0.9113283198951663),
}


def _kaiser_design(bits: int, passband_end: float) -> tuple[float, float, int]:
    """(rolloff, beta, num_zeros) for soxr's precision and passband end.

    Kaiser's formulas (J. F. Kaiser, 1974), as in
    :func:`scipy.signal.kaiser_beta` and :func:`scipy.signal.kaiserord`.
    """
    # As soxr does, the attenuation is that of one bit beyond the precision.
    att = (bits + 1) * 20.0 * math.log10(2.0)
    rolloff = 0.5 * (passband_end + 1.0)
    beta = 0.1102 * (att - 8.7)  # for att > 50 dB, as every preset has
    # Half of kaiserord's length, for a transition pi (1 - passband_end) wide.
    half_taps = math.ceil((att - 7.95) / (2.285 * 2.0 * math.pi * (1.0 - passband_end)))
    num_zeros = math.ceil(half_taps * rolloff)
    return rolloff, beta, num_zeros


_QUALITY_PRESETS = {name: _kaiser_design(*recipe) for name, recipe in _QUALITY_RECIPES.items()}

# The most coefficients of the CPU path's dense kernel, p (q + taps).
_MAX_KERNEL = 2**22

_HAS_RESAMPLE_OP = hasattr(torch.ops.philtorch, "resample")


def _positive_int(value: Any, name: str) -> int:
    message = f"{name} must be a positive integer, got {value!r}"
    result = _as_int(value, message)
    if result < 1:
        raise ValueError(message)
    return result


def _as_param(value: Any, name: str) -> float | Tensor:
    """A finite float or a 0-D floating-point tensor, as a design parameter."""
    if isinstance(value, Tensor):
        if value.dim() != 0 or not value.is_floating_point():
            raise ValueError(
                f"{name} must be a real number or a 0-D floating-point tensor, "
                f"got a {value.dim()}-D tensor of {value.dtype}"
            )
        return value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {float(value)}")
    return float(value)


def _first_order_filt(
    x: Tensor, a: Tensor, zi: Tensor, b: Tensor | None = None, **kwargs
) -> Tensor:
    xb = x if b is None else x * b
    return linear_recurrence(a, zi, xb)


def _cubic_coeff(x: Tensor, parallel_form: bool, scipy_padding: bool, **kwargs) -> Tensor:
    r = torch.tensor(3**0.5 - 2, device=x.device, dtype=x.dtype)
    # k_0 = min(14, x.shape[-1] - 1)

    if scipy_padding:
        # in scipy, a sequence [1, 2, 3] is mirrored to [3, 2, 1, 1, 2, 3, 3, 2, 1]
        powers = r ** torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
        causal_zi = x @ powers
    else:
        # while in torch reflect padding, [1, 2, 3] is padded to [2, 1, 2, 3, 2]
        powers = r ** torch.arange(x.shape[-1] - 1, device=x.device, dtype=x.dtype)
        causal_zi = x[..., 1:] @ powers

    if parallel_form:
        mirrored_x = torch.cat([x, x.flip(-1) if scipy_padding else x[..., :-1].flip(-1)], dim=-1)

        h = _first_order_filt(mirrored_x, r, causal_zi, **kwargs)
        causal_h, anticausal_h = h[..., : x.shape[-1]], h[..., -x.shape[-1] :].flip(-1)
        c = -6 * r / (1 - r * r) * (causal_h + anticausal_h - x)
    else:
        # causal inverse filtering
        h = _first_order_filt(x, r, causal_zi, **kwargs)
        if scipy_padding:
            zi = r / (r - 1) * h[..., -1]
        else:
            zi = -r / (1 - r * r) * (2 * h[..., -1] - x[..., -1])

        # anticausal inverse filtering
        c_flip = _first_order_filt(h[..., :-1].flip(-1), r, zi, -r, **kwargs).flip(-1)
        c = torch.cat([c_flip, zi.unsqueeze(-1)], dim=-1) * 6
    return c


def _cubic_spline_kernel(x: Tensor) -> Tensor:
    abs_x = x.abs()
    mask1 = abs_x <= 1
    mask2 = ~mask1 & (abs_x < 2)
    return torch.where(
        mask1,
        (4 - 6 * abs_x**2 + 3 * abs_x**3) / 6,
        torch.where(
            mask2,
            (2 - abs_x) ** 3 / 6,
            0.0,
        ),
    )


def cspline(
    x: Tensor,
    parallel_form: bool = True,
    scipy_padding: bool = False,
    lamb: float = 0.0,
    **kwargs,
) -> Tensor:
    """Compute the cubic B-spline coefficients of signals.

    The coefficients come from recursive filtering, as in M. Unser, "B-Spline
    Signal Processing: Part II---Efficient Design and Applications," IEEE
    Transactions on Signal Processing, 1993.

    Args:
        x (Tensor): signals, of shape :math:`(B, L)`.
        parallel_form (bool, optional): sum a causal and an anticausal filter
            if ``True``, or cascade them if ``False``; both give the same
            coefficients. Default: ``True``.
        scipy_padding (bool, optional): see :func:`cubic_spline`.
            Default: ``False``.
        lamb (float, optional): the smoothing coefficient; only ``0.0`` is
            implemented. Default: ``0.0``.
        **kwargs: unused.

    Returns:
        Tensor: the coefficients, of shape :math:`(B, L)`.

    Raises:
        NotImplementedError: if :attr:`lamb` is not zero.
    """
    if lamb != 0.0:
        raise NotImplementedError(
            "Regularization for cubic spline interpolation is not implemented."
        )
    return _cubic_coeff(x, parallel_form, scipy_padding, **kwargs)


def cubic_spline(x: Tensor, m: int, scipy_padding: bool = False, **kwargs) -> Tensor:
    """Upsample signals by an integer factor with cubic B-spline interpolation.

    This computes the cubic B-spline coefficients of :attr:`x` by recursive
    filtering, as in M. Unser, "B-Spline Signal Processing: Part II---Efficient
    Design and Applications," IEEE Transactions on Signal Processing, 1993,
    and evaluates the spline at :attr:`m` points per sample interval. The
    spline passes through the original samples.

    Args:
        x (Tensor): signals, of shape :math:`(B, L)`. Unlike SciPy, the batch
            dimension is required.
        m (int): the upsampling factor, at least 1.
        scipy_padding (bool, optional): the boundary condition. If ``True``,
            the signal is extended by mirror symmetry as in
            :func:`scipy.signal.cspline1d`, and the output equals
            :func:`scipy.signal.cspline1d_eval` on the upsampled grid. If
            ``False``, it is extended like the ``"reflect"`` mode of
            :func:`torch.nn.functional.pad`. Default: ``False``.
        **kwargs: options for the coefficients: ``parallel_form`` (bool)
            sums a causal and an anticausal filter if ``True``, the default,
            or cascades them if ``False``, with the same result; ``lamb``
            (float), the smoothing coefficient, must be ``0.0``.

    Returns:
        Tensor: the upsampled signals, of shape :math:`(B, (L - 1) m + 1)`.
        With ``m=1``, :attr:`x` itself.

    Raises:
        ValueError: if :attr:`m` is not an integer of at least 1, or
            :attr:`x` is not 2-D.
        NotImplementedError: if ``lamb`` is not zero.

    Example::

        >>> from philtorch.lti import cubic_spline
        >>> x = torch.randn(1, 16, dtype=torch.float64)
        >>> y = cubic_spline(x, 4)
        >>> y.shape
        torch.Size([1, 61])
        >>> torch.allclose(y[:, ::4], x)
        True
    """
    if not isinstance(m, int) or m < 1:
        raise ValueError("Interpolation factor m must be an integer >= 1.")
    if x.dim() != 2:
        raise ValueError(f"Input signal x must be 2D (batch, time), got {x.shape}")
    if m == 1:
        return x

    c = cspline(x, scipy_padding=scipy_padding, **kwargs)

    kernel_idx = torch.arange(-2, 2, 1 / m, device=x.device, dtype=x.dtype).reshape(4, m)
    kernel = _cubic_spline_kernel(kernel_idx).flip(0).T

    interped = F.conv1d(
        (
            torch.cat([c[:, :1], c, c[:, -2:].flip(-1)], dim=1).unsqueeze(1)
            if scipy_padding
            else F.pad(c.unsqueeze(1), (1, 2), mode="reflect")
        ),
        kernel.unsqueeze(1),
    ).mT.flatten(1, 2)
    return interped[..., : -(m - 1)]


def _cutoff(p: int, q: int, rolloff: float | Tensor) -> float | Tensor:
    """The cutoff c of the resampler for the rates q -> p."""
    return rolloff * p / q if p < q else rolloff


def _num_taps(p: int, q: int, rolloff: float, num_zeros: int) -> int:
    """The number of taps of one output of the resampler."""
    # The window is open, |t - n| < S = N_z / c: with K = ceil(S), the taps
    # n = floor(t) - K .. floor(t) + K + 1 hold it, 2 K + 2 in all.
    return 2 * math.ceil(num_zeros / _cutoff(p, q, rolloff)) + 2


def _constant_cache(fn):
    """Cache ``fn`` and make Dynamo take its result as a constant.

    Through a lambda: Dynamo would trace into an ``lru_cache`` wrapper. Built
    outside inference mode, so that later calls can save it for backward.
    """

    @functools.lru_cache(maxsize=16)
    def cached(*args):
        with torch.inference_mode(False):
            return fn(*args)

    return torch.compiler.assume_constant_result(lambda *args: cached(*args))


def _polyphase_table(
    p: int,
    q: int,
    rolloff: float | Tensor,
    beta: float | Tensor,
    num_zeros: int,
    taps: int,
    device: torch.device | None = None,
) -> Tensor:
    """The (p, taps) float64 table of the resampler for rates q -> p.

    Output ``f p + i`` is ``sum_k table[i, k] x[q f + (i q) // p - K + k]``,
    with ``taps = 2 K + 2``. Tensor ``rolloff`` and ``beta`` (0-D float64, on
    ``device``) make it differentiable in them.
    """
    rolloff, beta = (
        torch.as_tensor(
            v.to(device) if isinstance(v, Tensor) else v, dtype=torch.float64, device=device
        )
        for v in (rolloff, beta)
    )
    c = _cutoff(p, q, rolloff)
    frac = (((torch.arange(p, dtype=torch.int64, device=device) * q) % p).double() / p).unsqueeze(1)
    k = torch.arange(taps, dtype=torch.float64, device=device)
    # t - n = frac + K - k for the tap k of the phase starting at floor(t) - K.
    z = c * (frac + ((taps - 2) // 2 - k))
    inside = z.abs() < num_zeros
    # I0(beta sqrt(1 - u^2)) / I0(beta) from i0e, which can't overflow, even in
    # beta; the argument is 1 off the window, where sqrt's derivative is finite.
    arg = beta * torch.sqrt(torch.where(inside, 1 - (z / num_zeros) ** 2, 1.0))
    window = torch.special.i0e(arg) / torch.special.i0e(beta)
    window = window * torch.exp(arg.abs() - beta.abs())
    # sinc by hand: torch.sinc's second derivative is NaN at 0 (pytorch/pytorch#196703),
    # where z = 0 for every c (it is c times an integer).
    nonzero = z != 0
    z_pi = math.pi * torch.where(nonzero, z, 1.0)
    sinc = torch.where(nonzero, torch.sin(z_pi) / z_pi, 1.0)
    return torch.where(inside, c * sinc * window, 0.0)


@_constant_cache
def _table_on(
    p: int,
    q: int,
    design: tuple[float, float, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """The table of q -> p on ``device``."""
    return _polyphase_table(p, q, *design).to(device=device, dtype=dtype)


def _polyphase_conv(x: Tensor, table: Tensor, q: int, out_len: int) -> Tensor:
    """Resample with one strided convolution, for devices without the op."""
    p, taps = table.shape
    xp, width, _ = _conv_input(x, p, q, taps, out_len)
    columns = _tap_columns(p, q, taps, table.device)
    kernel = table.new_zeros(p, width).scatter(1, columns, table)
    y = F.conv1d(xp, kernel.unsqueeze(1), stride=q)
    return y.mT.reshape(x.size(0), -1)[:, :out_len]


class _Resample(torch.autograd.Function):
    """The CUDA op in eager mode, with ``jvp`` and ``vmap``.

    Compiled code calls the op, whose autograd shares ``_resample_gradient``.
    """

    @staticmethod
    def forward(x: Tensor, table: Tensor, q: int, out_len: int) -> Tensor:
        return torch.ops.philtorch.resample(x, table, q, out_len)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _resample_setup_context(ctx, inputs, output)
        ctx.save_for_forward(*inputs[:2])

    @staticmethod
    def backward(ctx, grad):
        return _resample_gradient(_Resample.apply, ctx, grad)

    @staticmethod
    def jvp(ctx, dx, dtable, *_):
        # Bilinear in (x, table): one call per tangent.
        x, table = ctx.saved_tensors
        rest = (ctx.q, ctx.out_len)
        if dtable is None:
            return _Resample.apply(dx, table, *rest)
        tangent = _Resample.apply(x, dtable, *rest)
        return tangent if dx is None else tangent + _Resample.apply(dx, table, *rest)

    @staticmethod
    def vmap(info, in_dims, x, *params):
        if in_dims[1] is not None:
            raise NotImplementedError("resample: vmap over rolloff or beta is not supported")
        x = x.movedim(in_dims[0], 0)
        V, B, T = x.shape
        y = _Resample.apply(x.reshape(V * B, T), *params)
        return y.reshape(V, B, params[-1]), 0


def resample(
    x: Tensor,
    orig_freq: int,
    new_freq: int,
    quality: str | None = "HQ",
    *,
    rolloff: float | Tensor | None = None,
    beta: float | Tensor | None = None,
    num_zeros: int | None = None,
) -> Tensor:
    r"""Resample signals between two sampling rates with a Kaiser-sinc filter.

    For an input signal :math:`x[b, n]` of length :math:`T` at the rate
    :attr:`orig_freq` and the resampling ratio
    :math:`r = \text{new\_freq} / \text{orig\_freq}`, this evaluates the
    band-limited Kaiser-sinc reconstruction at output sample indices
    :math:`m \in \{0, \dots, L_{\text{out}} - 1\}`, corresponding to input
    times :math:`t_m = m \cdot \text{orig\_freq} / \text{new\_freq}`, exactly:

    .. math::
        y[b, m] = \sum_{n = \lceil t_m - W \rceil}^{\lfloor t_m + W \rfloor}
        x[b, n]\, c\, \operatorname{sinc}\!\bigl(c (t_m - n)\bigr)\,
        \frac{I_0\!\left(\beta \sqrt{1 - ((t_m - n) / W)^2}\right)}{I_0(\beta)},

    where :math:`\operatorname{sinc}(z) = \sin(\pi z) / (\pi z)`,
    :math:`c = \rho \min(1, r)` is the anti-aliasing cutoff in normalized input
    Nyquist units, :math:`W = N_z / c` is the window half-width in input
    samples, and samples outside the signal (:math:`n < 0` or :math:`n \ge T`)
    are zero. The filter also acts at equal rates, where it low-passes at
    :math:`\rho` times the Nyquist frequency, unlike
    :func:`scipy.signal.resample_poly`, which then copies the input.

    The rates are reduced by their greatest common divisor to :math:`q \to p`,
    and the weights are evaluated exactly, once, in ``float64``, as a
    polyphase filter of about :math:`p (q + 2 N_z / c)` coefficients, at most
    :math:`2^{22}`. For a ratio beyond that, approximate it:
    ``f = Fraction(new_freq, orig_freq).limit_denominator(1000)``, from
    :mod:`fractions`, then ``resample(x, f.denominator, f.numerator)``.

    The presets of :attr:`quality` realise the passband, stopband and
    precision targets of ``libsoxr`` with one Kaiser-windowed sinc stage, not
    soxr's multi-stage design, so the outputs are not bit-identical to soxr's.
    ``float32`` accumulation gives errors of around -110 dB to -140 dB. That
    can miss the 20-bit target of ``"HQ"``, about -126 dB, and misses the
    28-bit target of ``"VHQ"``, about -175 dB: use ``float64`` for them.

    Derivatives of any order exist in :attr:`x`, and in :attr:`rolloff` and
    :attr:`beta` when they are 0-D tensors. :attr:`rolloff`'s value sets the
    filter's taps, which its gradient holds fixed; a tensor :attr:`rolloff`
    costs a sync and a graph break under :func:`torch.compile`. A negative
    tensor :attr:`beta` acts as its absolute value. On CUDA,
    :func:`torch.func.jvp` and :func:`torch.func.vmap` (over :attr:`x`) work
    in eager mode; :func:`torch.compile` supports reverse mode.

    Args:
        x (Tensor): signals, of shape :math:`(B, T)`, in a floating-point
            dtype (``float16``, ``bfloat16``, ``float32`` or ``float64``).
        orig_freq (int): the sampling rate of :attr:`x`, a positive integer.
        new_freq (int): the sampling rate of the output, a positive integer.
        quality (str or None, optional): filter design preset, from
            ``libsoxr``'s targets, with the parameters :math:`\rho`,
            :math:`\beta` and :math:`N_z`:

            - ``"LQ"``: 16-bit precision (:math:`\rho = 0.8381`,
              :math:`\beta \approx 10.32`, :math:`N_z = 18`);
            - ``"MQ"``: 16-bit precision (:math:`\rho = 0.9579`,
              :math:`\beta \approx 10.32`, :math:`N_z = 76`);
            - ``"HQ"``: 20-bit precision (:math:`\rho = 0.9569`,
              :math:`\beta \approx 12.97`, :math:`N_z = 92`);
            - ``"VHQ"``: 28-bit precision (:math:`\rho = 0.9557`,
              :math:`\beta \approx 18.28`, :math:`N_z = 126`);
            - ``None``: require explicit :attr:`rolloff`, :attr:`beta` and
              :attr:`num_zeros`.

            Explicit :attr:`rolloff`, :attr:`beta` or :attr:`num_zeros`
            override the preset's values. Default: ``"HQ"``.
        rolloff (float or Tensor, optional): the filter cutoff
            :math:`\rho \in (0, 1]` as a fraction of the lower Nyquist
            frequency. Default: ``None``.
        beta (float or Tensor, optional): the Kaiser window shape parameter
            :math:`\beta \ge 0`. Default: ``None``.
        num_zeros (int, optional): the number of zero-crossings
            :math:`N_z \ge 1` per side of the sinc mainlobe. Default: ``None``.

    Returns:
        Tensor: the resampled signals, of shape :math:`(B, L_{\text{out}})`,
        where :math:`L_{\text{out}} = \lceil T \cdot \text{new\_freq} /
        \text{orig\_freq} \rceil`, computed in integer arithmetic, as
        :func:`scipy.signal.resample_poly` does.

    Raises:
        ValueError: if :attr:`x` is not a 2-D floating-point tensor, if
            :attr:`orig_freq` or :attr:`new_freq` is not a positive integer,
            if :attr:`quality` is unknown or ``None`` without all of
            :attr:`rolloff`, :attr:`beta` and :attr:`num_zeros`, if
            :attr:`rolloff` or :attr:`beta` is a tensor that is not 0-D and
            floating-point, if :attr:`rolloff`, :attr:`beta` or
            :attr:`num_zeros` is out of range, or if the polyphase filter would
            hold more than :math:`2^{22}` coefficients.

    Example::

        >>> from philtorch.lti import resample
        >>> t = torch.arange(441, dtype=torch.float64)
        >>> x = torch.sin(2 * torch.pi * 10 * t / 441).unsqueeze(0)
        >>> y = resample(x, 44100, 48000)
        >>> y.shape
        torch.Size([1, 480])
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D (B, T), got {x.dim()}D")
    if not x.is_floating_point():
        raise ValueError(f"x must be a floating-point tensor, got {x.dtype}")

    orig = _positive_int(orig_freq, "orig_freq")
    new = _positive_int(new_freq, "new_freq")
    gcd = math.gcd(orig, new)
    orig, new = orig // gcd, new // gcd

    if quality is not None and not (isinstance(quality, str) and quality in _QUALITY_PRESETS):
        valid = ", ".join(repr(k) for k in _QUALITY_PRESETS)
        raise ValueError(f"quality must be one of {{{valid}}} or None, got {quality!r}")
    preset = _QUALITY_PRESETS.get(quality, (None, None, None))
    if quality is None and any(v is None for v in (rolloff, beta, num_zeros)):
        raise ValueError("when quality is None, rolloff, beta and num_zeros must all be given")
    rolloff = preset[0] if rolloff is None else _as_param(rolloff, "rolloff")
    # The filter's length follows rolloff's value: a sync for a CUDA tensor.
    rolloff_value = float(rolloff.detach() if isinstance(rolloff, Tensor) else rolloff)
    if not 0.0 < rolloff_value <= 1.0:
        raise ValueError(f"rolloff must be in (0, 1], got {rolloff_value}")
    beta = preset[1] if beta is None else _as_param(beta, "beta")
    if not isinstance(beta, Tensor) and beta < 0.0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    num_zeros = preset[2] if num_zeros is None else _positive_int(num_zeros, "num_zeros")

    taps = _num_taps(new, orig, rolloff_value, num_zeros)
    if new * (orig + taps) > _MAX_KERNEL:
        raise ValueError(
            f"the filter for the reduced rates {orig} -> {new} needs {new} phases "
            f"spanning {orig + taps} inputs, more than 2**22 coefficients; approximate "
            "the ratio, e.g. f = Fraction(new_freq, orig_freq).limit_denominator(1000), "
            "then resample(x, f.denominator, f.numerator)"
        )

    out_len = -(-x.size(1) * new // orig)
    if x.numel() == 0 or out_len == 0:
        # Nothing to compute; the output stays in the graph of x and of tensor
        # parameters, with zero gradients.
        y = F.pad(x[:, :0], (0, out_len))
        for v in (rolloff, beta):
            if isinstance(v, Tensor):
                y = y + 0 * v.to(y.device, y.dtype)
        return y
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    if isinstance(rolloff, Tensor) or isinstance(beta, Tensor):
        # The table the gradient reaches the tensors through: built uncached in
        # float64, on x's device, or on the CPU for MPS, which has no float64.
        device = torch.device("cpu") if x.device.type == "mps" else x.device
        table = _polyphase_table(new, orig, rolloff, beta, num_zeros, taps, device)
        table = table.to(device=x.device, dtype=dtype)
    else:
        table = _table_on(new, orig, (rolloff, beta, num_zeros, taps), x.device, dtype)
        if not torch.compiler.is_compiling():
            table = _use_on_current_stream(table)
    if x.is_cuda and _HAS_RESAMPLE_OP:
        # Dynamo can't trace an autograd.Function with a jvp: compiled code
        # calls the registered op, whose autograd is the same transpose.
        apply = torch.ops.philtorch.resample if torch.compiler.is_compiling() else _Resample.apply
        return apply(x, table, orig, out_len)
    return _polyphase_conv(x.to(dtype), table, orig, out_len).to(x.dtype)
