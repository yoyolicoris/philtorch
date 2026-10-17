"""Signal interpolation and band-limited resampling."""

import math
import numbers
import operator
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .._ops import _resample_backward_with, _resample_setup_context
from .recur import linear_recurrence

# libsoxr's quality_spec (soxr.c): (precision in bits, passband_end), with
# stopband_begin = 1. The last three passband ends are
# 1 - .05 / lsx_to_3dB(precision * 20 log10(2)).
_QUALITY_RECIPES: dict[str, tuple[int, float]] = {
    "LQ": (16, 1385 / 2048),
    "MQ": (16, 0.9158810491765671),
    "HQ": (20, 0.9137436532733066),
    "VHQ": (28, 0.9113283198951663),
}


def _quality_kaiser_params(quality: str) -> tuple[float, float, int]:
    """Compute (rolloff, beta, num_zeros) from libsoxr's bits and passband.

    The window follows Kaiser's design formulas (J. F. Kaiser, "Nonrecursive
    digital filter design using the I0-sinh window function", 1974), as in
    :func:`scipy.signal.kaiser_beta` (0.1102 (A - 8.7), for an attenuation A of
    more than 50 dB) and :func:`scipy.signal.kaiserord` (a length of
    (A - 7.95) / (2.285 w) + 1, for a transition width w in rad/sample).
    """
    bits, passband_end = _QUALITY_RECIPES[quality]
    # soxr's cr.c: att = (bits1 + 1) * linear_to_dB(2.)
    att = (bits + 1) * 20.0 * math.log10(2.0)
    # Every preset's att is above 50 dB (102 to 175 dB), where the beta formula
    # holds; a test pins the presets.
    rolloff = 0.5 * (passband_end + 1.0)
    beta = 0.1102 * (att - 8.7)
    # Half of kaiserord's length for a transition pi (1 - passband_end)
    # rad/sample wide.
    half_taps = math.ceil((att - 7.95) / (2.285 * 2.0 * math.pi * (1.0 - passband_end)))
    num_zeros = math.ceil(half_taps * rolloff)
    return rolloff, beta, num_zeros


_QUALITY_PRESETS = {quality: _quality_kaiser_params(quality) for quality in _QUALITY_RECIPES}


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _as_int(value: Any, name: str, minimum: int) -> int:
    """An integer of at least ``minimum``, 0 or 1; bools are not integers."""
    kind = "a positive" if minimum else "a non-negative"
    if isinstance(value, bool):
        raise ValueError(f"{name} must be {kind} integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be {kind} integer, got {value!r}") from None
    if result < minimum:
        raise ValueError(f"{name} must be {kind} integer, got {value!r}")
    return result


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


class _Resample(torch.autograd.Function):
    """Kaiser-sinc resampling, differentiable in ``x`` to any order in eager mode.

    Resampling is linear in ``x``: its tangent (``jvp``) is the same resampling
    of ``dx``, and its gradient (``backward``) is the transpose, resampling
    with the two rates swapped. The op's own autograd under
    :func:`torch.compile` shares that backward, from :mod:`philtorch._ops`.
    Both call ``apply`` again, so reverse and forward mode compose to any
    order. Under :func:`torch.func.vmap`, the vmapped dimension joins the
    batch. Dynamo can't trace an autograd function with a ``jvp``, so compiled
    code skips this class.
    """

    @staticmethod
    def forward(
        x: Tensor,
        orig_freq: int,
        new_freq: int,
        scale: float,
        rolloff: float,
        beta: float,
        num_zeros: int,
        out_len: int,
    ) -> Tensor:
        return torch.ops.philtorch.resample(
            x, orig_freq, new_freq, scale, rolloff, beta, num_zeros, out_len
        )

    setup_context = staticmethod(_resample_setup_context)
    backward = staticmethod(_resample_backward_with(lambda *args: _Resample.apply(*args)))

    @staticmethod
    def jvp(ctx, dx, *_):
        return _Resample.apply(dx, *ctx.params)

    @staticmethod
    def vmap(info, in_dims, x, *params):
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
    rolloff: float | None = None,
    beta: float | None = None,
    num_zeros: int | None = None,
    out_len: int | None = None,
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

    The two rates are reduced by their greatest common divisor. When the
    reduced :attr:`new_freq`, the number of phases, is at most the output
    length and its product with the number of taps, about :math:`2 N_z / c`,
    is at most :math:`2^{20}`, the weights come from a polyphase table
    computed once in ``float64``. Otherwise they are computed per output. Both
    are exact evaluations of the formula, with no interpolated table, in a
    native kernel for the CPU and CUDA; there is none for MPS. To approximate
    an arbitrary ratio, take ``f = Fraction(ratio).limit_denominator(1000)``,
    from :mod:`fractions`, and pass ``orig_freq=f.denominator`` and
    ``new_freq=f.numerator``; a large denominator means many phases, and the
    slower per-output path.

    The presets of :attr:`quality` realise the passband, stopband and
    precision targets of ``libsoxr`` with one Kaiser-windowed sinc stage, not
    soxr's multi-stage design, so the outputs are not bit-identical to soxr's.
    ``float32`` accumulation gives errors of around -120 dB to -140 dB. That
    can miss the 20-bit target of ``"HQ"``, about -126 dB, on the per-output
    path, and misses the 28-bit target of ``"VHQ"``, about -175 dB, on both:
    use ``float64`` for them.

    Gradients and forward-mode derivatives exist in :attr:`x` to any order:
    the transpose of the resampling is the resampling with the two rates
    swapped. Forward mode (:func:`torch.func.jvp`) and :func:`torch.func.vmap`
    work in eager mode only; :func:`torch.compile` supports reverse mode.

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
        rolloff (float, optional): the filter cutoff :math:`\rho \in (0, 1]` as
            a fraction of the lower Nyquist frequency. Default: ``None``.
        beta (float, optional): the Kaiser window shape parameter
            :math:`\beta \ge 0`. Default: ``None``.
        num_zeros (int, optional): the number of zero-crossings
            :math:`N_z \ge 1` per side of the sinc mainlobe. Default: ``None``.
        out_len (int, optional): the output sequence length
            :math:`L_{\text{out}} \ge 0`. Default: ``None``, which uses
            ``ceil(T * new_freq / orig_freq)``, computed in integer
            arithmetic as in :func:`scipy.signal.resample_poly`.

    Returns:
        Tensor: the resampled signals, of shape :math:`(B, L_{\text{out}})`.

    Raises:
        ValueError: if :attr:`x` is not a 2-D floating-point tensor, if
            :attr:`orig_freq` or :attr:`new_freq` is not a positive integer,
            if :attr:`quality` is unknown or ``None`` without all of
            :attr:`rolloff`, :attr:`beta` and :attr:`num_zeros`, or if
            :attr:`rolloff`, :attr:`beta`, :attr:`num_zeros` or
            :attr:`out_len` is out of range. Also, from the native kernel,
            if :attr:`beta` exceeds about 346 (for every path and dtype), if
            the product of the reduced rates exceeds :math:`2^{62}`, if
            :attr:`out_len` reaches input positions beyond :math:`2^{62}`, or
            if the filter has more than :math:`2^{30}` taps.

    Example::

        >>> from philtorch.lti import resample
        >>> t = torch.arange(441, dtype=torch.float64)
        >>> x = torch.sin(2 * torch.pi * 10 * t / 441).unsqueeze(0)
        >>> y = resample(x, 44100, 48000, quality="HQ")
        >>> y.shape
        torch.Size([1, 480])
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D (B, T), got {x.dim()}D")
    if not x.is_floating_point():
        raise ValueError(f"x must be a floating-point tensor, got {x.dtype}")

    orig = _as_int(orig_freq, "orig_freq", 1)
    new = _as_int(new_freq, "new_freq", 1)
    gcd = math.gcd(orig, new)
    orig, new = orig // gcd, new // gcd

    if quality is None:
        if rolloff is None or beta is None or num_zeros is None:
            raise ValueError("when quality is None, rolloff, beta and num_zeros must all be given")
        design = [rolloff, beta, num_zeros]
    elif isinstance(quality, str) and quality in _QUALITY_PRESETS:
        design = list(_QUALITY_PRESETS[quality])
    else:
        valid = ", ".join(repr(k) for k in _QUALITY_PRESETS)
        raise ValueError(f"quality must be one of {{{valid}}} or None, got {quality!r}")

    # The presets are valid: only the given values are checked.
    if rolloff is not None:
        design[0] = _as_float(rolloff, "rolloff")
        if not 0.0 < design[0] <= 1.0:
            raise ValueError(f"rolloff must be in (0, 1], got {design[0]}")
    if beta is not None:
        design[1] = _as_float(beta, "beta")
        if design[1] < 0.0:
            raise ValueError(f"beta must be non-negative, got {design[1]}")
    if num_zeros is not None:
        design[2] = _as_int(num_zeros, "num_zeros", 1)
    rolloff_val, beta_val, zeros_val = design

    if out_len is None:
        out_len_val = -(-x.shape[1] * new // orig)
    else:
        out_len_val = _as_int(out_len, "out_len", 0)

    scale = rolloff_val * new / orig if new < orig else rolloff_val
    # Dynamo can't trace an autograd.Function with a jvp: compiled code calls
    # the registered op, whose autograd is the same transpose.
    apply = torch.ops.philtorch.resample if torch.compiler.is_compiling() else _Resample.apply
    return apply(x, orig, new, scale, rolloff_val, beta_val, zeros_val, out_len_val)
