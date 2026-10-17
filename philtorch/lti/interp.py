"""Signal interpolation and band-limited resampling."""

import functools
import math
import numbers
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .._ops import _resample_gradient, _resample_setup_context
from .delay import _as_int
from .recur import linear_recurrence

# Passband ends of libsoxr's presets, from soxr.c and filter.c: LQ is
# 1385 / 2048, and the others are 1 - .05 / lsx_to_3dB(rej), with
# rej = bits * 20 log10(2) and lsx_to_3dB(a) = 1 - lsx_inv_f_resp(-3, a).


def _soxr_inv_f_resp(drop: float, a: float) -> float:
    """Port of lsx_inv_f_resp (soxr's filter.c)."""
    x = ((2.0517e-07 * a - 1.1303e-04) * a + 0.023154) * a + 0.55924  # sinePhi
    power = math.log(0.5) / math.log(math.sin(x * 0.5))  # sinePow
    drop = 10 ** (drop / 20)  # dB_to_linear
    s = 1 - drop if drop > 0.5 else drop
    x = math.asin(s ** (1 / power)) / x
    return x if drop > 0.5 else 1 - x


def _soxr_passband_end(bits: int) -> float:
    rejection = bits * 20.0 * math.log10(2.0)
    return 1 - 0.05 / (1 - _soxr_inv_f_resp(-3, rejection))


# (precision in bits, passband_end), with stopband_begin = 1.
_QUALITY_RECIPES: dict[str, tuple[int, float]] = {
    "LQ": (16, 1385 / 2048),
    "MQ": (16, _soxr_passband_end(16)),
    "HQ": (20, _soxr_passband_end(20)),
    "VHQ": (28, _soxr_passband_end(28)),
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

# The most coefficients of the dense polyphase kernel of the conv1d path, p rows
# of about q + taps (32 MB in float64). It bounds the table, p * taps, as well,
# and the transpose's kernel is the same size.
_MAX_KERNEL = 2**22

_HAS_RESAMPLE_OP = hasattr(torch.ops.philtorch, "resample")


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


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


def _filter_taps(p: int, q: int, rolloff: float, num_zeros: int) -> tuple[float, int]:
    """The cutoff c and the number of taps of one output of the resampler."""
    c = rolloff * p / q if p < q else rolloff
    # The window is open, |c (frac - k)| < N_z: at most floor(2 S) + 1 taps for
    # S = N_z / c; one more is a guard against rounding at the edge.
    return c, math.floor(2 * num_zeros / c) + 2


def _constant_cache(fn):
    """Cache ``fn``, of Python arguments, and make Dynamo run it, not trace it.

    Dynamo would trace through an ``lru_cache`` and rebuild its tensors in the
    compiled graph; as a constant, the cached tensors are lifted instead.
    """
    cached = functools.lru_cache(maxsize=16)(fn)
    return torch.compiler.assume_constant_result(lambda *args: cached(*args))


def _polyphase_table(
    p: int, q: int, rolloff: float, beta: float, num_zeros: int, scale: float
) -> tuple[Tensor, Tensor]:
    """The weights of the resampler in the polyphase form, for rates q -> p.

    Output ``f p + i`` is ``sum_k table[i, k] x[q f + first[i] + k]``, with x
    zero outside the signal. The table, of shape (p, taps) in float64, holds
    ``scale * g(c (t - n))`` for the window g of :func:`resample`, at the input
    time ``t = (i q) / p`` of phase i and the n of its taps. The integer part
    of t and the remainder (i q mod p) / p are computed exactly.
    """
    c, taps = _filter_taps(p, q, rolloff, num_zeros)
    i = torch.arange(p, dtype=torch.int64)
    whole = (i * q) // p
    frac = ((i * q) % p).double() / p
    first = whole + torch.ceil(frac - num_zeros / c).long()
    k = torch.arange(taps, dtype=torch.float64)
    z = c * (frac - (first - whole).double()).unsqueeze(1) - c * k
    # I0(beta sqrt(1 - u^2)) / I0(beta) from the scaled i0e, which cannot
    # overflow for a large beta.
    arg = beta * torch.sqrt((1 - (z / num_zeros) ** 2).clamp(min=0))
    window = torch.special.i0e(arg) / torch.special.i0e(torch.tensor(beta, dtype=torch.float64))
    window = window * torch.exp(arg - beta)
    table = torch.where(z.abs() < num_zeros, scale * torch.sinc(z) * window, 0.0)
    return table, first


@_constant_cache
def _tables_on(
    p: int,
    q: int,
    design: tuple[float, float, int, float],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The table of q -> p and its transpose's (p -> q), on ``device``.

    The transpose of the resampling is the resampling with the rates swapped
    and the same scale, see :func:`philtorch._ops._resample_gradient`.
    """
    table, first = _polyphase_table(p, q, *design)
    table_t, first_t = _polyphase_table(q, p, *design)
    return (
        table.to(device=device, dtype=dtype),
        first.to(device),
        table_t.to(device=device, dtype=dtype),
        first_t.to(device),
    )


@_constant_cache
def _conv_kernel(
    p: int,
    q: int,
    design: tuple[float, float, int, float],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, int]:
    """The (p, 1, L) kernel of :func:`_polyphase_conv`, and ``min(first)``.

    The first tap of a phase grows with the phase, so ``first[0]`` is the
    smallest and ``first[-1]`` the largest.
    """
    table, first = _polyphase_table(p, q, *design)
    shift, last = int(first[0]), int(first[-1])
    offset = (first - shift).unsqueeze(1) + torch.arange(table.size(1))
    kernel = table.new_zeros(p, last - shift + table.size(1)).scatter_(1, offset, table)
    return kernel.unsqueeze(1).to(device=device, dtype=dtype), shift


def _polyphase_conv(x: Tensor, kernel: Tensor, shift: int, q: int, out_len: int) -> Tensor:
    """Resample with one strided convolution, for devices without the op.

    Frame f of the convolution, of stride q, reads ``x[q f + shift + j]`` for
    the kernel's taps j, and gives p outputs, one per row of the kernel.
    """
    batch_size = x.size(0)
    p, _, width = kernel.shape
    frames = -(-out_len // p)
    needed = q * (frames - 1) + width
    if shift < 0:
        x = F.pad(x, (-shift, 0))
    else:
        x = x[:, shift:]
    x = x[:, :needed]
    x = F.pad(x, (0, needed - x.size(1)))
    y = F.conv1d(x.unsqueeze(1), kernel, stride=q)
    return y.mT.reshape(batch_size, -1)[:, :out_len]


class _Resample(torch.autograd.Function):
    """The native resampling op, differentiable in ``x`` to any order in eager mode.

    Resampling is linear in ``x``: its tangent (``jvp``) is the same resampling
    of ``dx``, and its gradient (``backward``) is the transpose, the op with
    the tables swapped. The op's own autograd under :func:`torch.compile`
    shares that backward, from :mod:`philtorch._ops`. Both call the op again,
    so reverse and forward mode compose to any order. Under
    :func:`torch.func.vmap`, the vmapped dimension joins the batch. Dynamo
    can't trace an autograd function with a ``jvp``, so compiled code skips
    this class.
    """

    @staticmethod
    def forward(
        x: Tensor,
        table: Tensor,
        first: Tensor,
        table_t: Tensor,
        first_t: Tensor,
        q: int,
        out_len: int,
    ) -> Tensor:
        return torch.ops.philtorch.resample(x, table, first, table_t, first_t, q, out_len)

    setup_context = staticmethod(_resample_setup_context)

    @staticmethod
    def backward(ctx, grad):
        return _resample_gradient(_Resample.apply, ctx, grad)

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

    The two rates are reduced by their greatest common divisor. The filter
    weights come from a polyphase table, computed once in ``float64``, with
    one phase per reduced :attr:`new_freq` and about :math:`2 N_z / c` taps
    per phase: an exact evaluation of the formula, with no interpolated table.
    It runs as a strided convolution, and on CUDA, when the native extension
    has its kernel, as that kernel. The convolution's kernel, about
    :math:`p (q + 2 N_z / c)` coefficients for the reduced rates
    :math:`q \to p`, is limited to :math:`2^{22}`, so to approximate an
    arbitrary ratio, take
    ``f = Fraction(new_freq, orig_freq).limit_denominator(1000)``, from
    :mod:`fractions`, and pass ``orig_freq=f.denominator`` and
    ``new_freq=f.numerator``.

    The presets of :attr:`quality` realise the passband, stopband and
    precision targets of ``libsoxr`` with one Kaiser-windowed sinc stage, not
    soxr's multi-stage design, so the outputs are not bit-identical to soxr's.
    ``float32`` accumulation gives errors of around -110 dB to -140 dB. That
    can miss the 20-bit target of ``"HQ"``, about -126 dB, and misses the
    28-bit target of ``"VHQ"``, about -175 dB: use ``float64`` for them.

    Gradients and forward-mode derivatives exist in :attr:`x` to any order:
    the transpose of the resampling is the resampling with the two rates
    swapped. On CUDA with the native kernel, forward mode
    (:func:`torch.func.jvp`) and :func:`torch.func.vmap` work in eager mode
    only; :func:`torch.compile` supports reverse mode.

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
            :attr:`rolloff`, :attr:`beta` or :attr:`num_zeros` is out of
            range, or if the polyphase kernel for
            the reduced rates and the filter would hold more than
            :math:`2^{22}` coefficients.

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

    message = "{} must be a positive integer, got {!r}"
    orig = _as_int(orig_freq, message.format("orig_freq", orig_freq))
    new = _as_int(new_freq, message.format("new_freq", new_freq))
    if orig < 1:
        raise ValueError(message.format("orig_freq", orig_freq))
    if new < 1:
        raise ValueError(message.format("new_freq", new_freq))
    gcd = math.gcd(orig, new)
    orig, new = orig // gcd, new // gcd

    if quality is None:
        if rolloff is None or beta is None or num_zeros is None:
            raise ValueError("when quality is None, rolloff, beta and num_zeros must all be given")
        preset = (None, None, None)
    elif isinstance(quality, str) and quality in _QUALITY_PRESETS:
        preset = _QUALITY_PRESETS[quality]
    else:
        valid = ", ".join(repr(k) for k in _QUALITY_PRESETS)
        raise ValueError(f"quality must be one of {{{valid}}} or None, got {quality!r}")

    rolloff = preset[0] if rolloff is None else _as_float(rolloff, "rolloff")
    if not 0.0 < rolloff <= 1.0:
        raise ValueError(f"rolloff must be in (0, 1], got {rolloff}")
    beta = preset[1] if beta is None else _as_float(beta, "beta")
    if beta < 0.0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if num_zeros is None:
        num_zeros = preset[2]
    else:
        num_zeros = _as_int(num_zeros, message.format("num_zeros", num_zeros))
        if num_zeros < 1:
            raise ValueError(message.format("num_zeros", num_zeros))

    out_len = -(-x.size(1) * new // orig)

    taps = _filter_taps(new, orig, rolloff, num_zeros)[1]
    if new * (orig + taps) > _MAX_KERNEL:
        raise ValueError(
            f"the filter for the reduced rates {orig} -> {new} needs {new} phases "
            f"spanning {orig + taps} inputs, more than 2**22 coefficients; approximate "
            "the ratio, e.g. "
            "f = Fraction(new_freq, orig_freq).limit_denominator(1000), then "
            "resample(x, f.denominator, f.numerator)"
        )

    scale = rolloff * new / orig if new < orig else rolloff
    design = (rolloff, beta, num_zeros, scale)
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    if x.is_cuda and _HAS_RESAMPLE_OP:
        tables = _tables_on(new, orig, design, x.device, dtype)
        # Dynamo can't trace an autograd.Function with a jvp: compiled code
        # calls the registered op, whose autograd is the same transpose.
        apply = torch.ops.philtorch.resample if torch.compiler.is_compiling() else _Resample.apply
        return apply(x, *tables, orig, out_len)
    if x.size(0) == 0 or out_len == 0:
        # Nothing to compute; the pad keeps the output in x's graph, with zero gradient.
        return F.pad(x[:, :0], (0, out_len))
    kernel, shift = _conv_kernel(new, orig, design, x.device, dtype)
    return _polyphase_conv(x.to(dtype), kernel, shift, orig, out_len).to(x.dtype)
