"""IIR and FIR filtering with time-invariant coefficients."""

import operator
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from ..mat import companion
from ..poly import polydiv
from ..utils import chain_functions
from .recur import linear_recurrence
from .ssm import diag_state_space, state_space, state_space_recursion


def comb_filter(a: Tensor, delay: int, x: Tensor, zi: Tensor | None = None, **kwargs) -> Tensor:
    r"""Filter signals with an all-pole comb filter.

    This computes

    .. math::
        y[n] = x[n] - a\, y[n - D],

    the filter :math:`1 / (1 + a z^{-D})` with delay :math:`D`, by running
    :math:`D` interleaved first-order recurrences.

    Args:
        a (Tensor): the feedback coefficient, of shape :math:`()` to share it
            or :math:`(B)` for one per signal.
        delay (int): the delay :math:`D`, at least 1.
        x (Tensor): input signals, of shape :math:`(B, N)`.
        zi (Tensor, optional): the :math:`D` outputs before the signal, newest
            first: :math:`y[-1], \dots, y[-D]`, of shape :math:`(D)` or
            :math:`(B, D)`. Default: ``None``, all zero.
        **kwargs: passed to :func:`linear_recurrence`, e.g. ``unroll_factor``.

    Returns:
        Tensor or tuple of Tensor: the filtered signals, of shape
        :math:`(B, N)`, and with :attr:`zi`, the final state: the last
        :math:`D` outputs newest first, of shape :math:`(B, D)`.

    Raises:
        ValueError: if :attr:`delay` is less than 1, :attr:`a` has the
            wrong shape, or :attr:`x` is not 2-D.

    Example::

        >>> from philtorch.lti import comb_filter
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        >>> comb_filter(torch.tensor(-0.5), 2, x)
        tensor([[1.0000, 0.0000, 0.5000, 0.0000, 0.2500, 0.0000]])
    """
    if a.dim() > 1:
        raise ValueError("Denominator coefficients a must be at most 1D.")
    if x.dim() != 2:
        raise ValueError("Input signal x must be 2D.")
    if delay < 1:
        raise ValueError("Delay must be at least 1.")
    if a.dim() == 1:
        if a.size(0) != x.size(0):
            raise ValueError("The first dimension of a must match the batch size of x.")

    remainder = x.size(1) % delay
    if remainder != 0:
        x = F.pad(x, (0, delay - remainder))

    folded_x = x.unflatten(1, (-1, delay)).mT
    if a.dim() == 1:
        a = a.repeat_interleave(delay)
    if zi is not None:
        return_zf = True
        # The outputs before the signal, newest first, for the final state.
        history = zi.expand(x.size(0), -1)
        if zi.dim() == 1:
            zi = zi.flip(0).repeat(x.size(0))
        else:
            zi = zi.flip(1).flatten()
    else:
        return_zf = False
        zi = torch.zeros_like(a)

    y = (
        linear_recurrence(-a, zi, folded_x.flatten(0, 1), **kwargs)
        .unflatten(0, (-1, delay))
        .mT.flatten(1, 2)
    )
    if remainder != 0:
        y = y[:, : -(delay - remainder)]
    if return_zf:
        # The last D outputs, which reach back into zi when N < D. Only the
        # last D of y can be part of it, so copy just those.
        return y, torch.cat([history.flip(1), y[:, -delay:]], dim=1)[:, -delay:].flip(1)
    return y


def lfiltic(b: Tensor, a: Tensor, y: Tensor, x: Tensor | None = None) -> Tensor:
    r"""Construct the initial state of :func:`lfilter` from a signal's past.

    This returns the state, for ``form="tdf2"``, that continues a signal whose
    most recent outputs and inputs were :attr:`y` and :attr:`x`, as
    :func:`scipy.signal.lfiltic` does.

    Args:
        b (Tensor): numerator coefficients :math:`b_0, \dots, b_{M_b}`, of
            shape :math:`(*, M_b + 1)`, where :math:`*` is zero or more batch
            dimensions.
        a (Tensor): denominator coefficients :math:`a_1, \dots, a_{M_a}`,
            without the leading 1, of shape :math:`(*, M_a)`.
        y (Tensor): past outputs, newest first:
            :math:`y[-1], \dots, y[-M_a]`, of shape :math:`(*, M_a)`. As in
            SciPy, fewer values are padded with zeros, and only the first
            :math:`M_a` are used.
        x (Tensor, optional): past inputs, newest first:
            :math:`x[-1], \dots, x[-M_b]`, of shape :math:`(*, M_b)`, padded
            or truncated like :attr:`y`. Default: ``None``, all zero.

    Returns:
        Tensor: the initial state, of shape :math:`(*, \max(M_a, M_b))`.

    Raises:
        ValueError: if :attr:`b` or :attr:`a` is 0-D.

    Note:
        Unlike :func:`scipy.signal.lfiltic`, :attr:`a` omits the leading
        :math:`a_0 = 1`.

    Example::

        >>> from philtorch.lti import lfilter, lfiltic
        >>> b, a = torch.tensor([0.5, 0.5]), torch.tensor([-0.5])
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y = lfilter(b, a, x)
        >>> # Continue from the second sample. With longer histories, pass
        >>> # the past values newest first, e.g. y[:k].flip(-1).
        >>> zi = lfiltic(b, a, y[1:2], x[1:2])
        >>> y_rest, _ = lfilter(b, a, x[2:], zi=zi)
        >>> torch.allclose(y_rest, y[2:])
        True
    """
    if b.dim() < 1:
        raise ValueError("Numerator coefficients b must be at least 1D.")
    if a.dim() < 1:
        raise ValueError("Denominator coefficients a must be at least 1D.")

    n = a.size(-1)
    m = b.size(-1) - 1
    k = max(n, m)

    # As in SciPy, a shorter history is padded with zeros and a longer one is
    # truncated to the values the filter uses.
    y = F.pad(y, (0, n - y.size(-1)))
    x = b.new_zeros(m) if x is None else F.pad(x, (0, m - x.size(-1)))

    zi = y.new_zeros(y.shape[:-1] + (k,))
    if m > 0:
        b_mat = F.pad(b[..., 1:], (0, m - 1), value=0.0).unfold(-1, m, 1)
        zi = zi + F.pad((b_mat @ x.unsqueeze(-1)).squeeze(-1), (0, k - m))
    if n > 0:
        a_mat = F.pad(a, (0, n - 1), value=0.0).unfold(-1, n, 1)
        zi = zi - F.pad((a_mat @ y.unsqueeze(-1)).squeeze(-1), (0, k - n))
    return zi


def lfilter_zi(a: Tensor, b: Tensor | None = None, transpose: bool = True) -> Tensor:
    r"""Return the initial state of :func:`lfilter` for a steady step response.

    With this state, filtering a constant input of 1 starts in steady state.
    Scale it by the first input sample for other signals, as
    :func:`filtfilt` does.

    With ``transpose=True``, this is the state for ``form="tdf2"``, the same
    as :func:`scipy.signal.lfilter_zi`. With ``transpose=False``, it is the
    state for ``form="df2"``, which depends on :attr:`a` alone: the solution
    of :math:`(I - A) \mathbf{z} = \mathbf{e}_1` for the
    :func:`~philtorch.mat.companion` matrix :math:`A`. It only fits
    ``form="df2"`` when :math:`M_b \le M_a`.

    Args:
        a (Tensor): denominator coefficients :math:`a_1, \dots, a_{M_a}`,
            without the leading 1, of shape :math:`(*, M_a)`, where :math:`*`
            is zero or more batch dimensions.
        b (Tensor, optional): numerator coefficients, of shape
            :math:`(*, M_b + 1)`. Required with ``transpose=True`` and ignored
            otherwise. Default: ``None``.
        transpose (bool, optional): return the state for ``form="tdf2"`` if
            ``True``, or for ``form="df2"`` if ``False``. Default: ``True``.

    Returns:
        Tensor: the initial state, of shape :math:`(*, M)`, where
        :math:`M = \max(M_a, M_b)` with ``transpose=True`` and :math:`M_a`
        otherwise.

    Raises:
        ValueError: if :attr:`a` is 0-D, or, with ``transpose=True``, if
            :attr:`b` is ``None`` or 0-D.

    Note:
        The arguments are ``(a, b)``, the reverse of SciPy's ``(b, a)``, and
        :attr:`a` omits the leading :math:`a_0 = 1`.

    Example::

        >>> from philtorch.lti import lfilter, lfilter_zi
        >>> b, a = torch.tensor([0.5]), torch.tensor([-0.5])
        >>> y, _ = lfilter(b, a, torch.ones(4), zi=lfilter_zi(a, b))
        >>> y
        tensor([1., 1., 1., 1.])
    """
    if a.dim() < 1:
        raise ValueError("Denominator coefficients a must be at least 1D.")

    if not transpose:
        n = a.size(-1) + 1
        A = companion(a)
        B = a.new_zeros(n - 1)
        B[:1] = 1.0  # a slice, as an empty a has no first state
    elif b is not None:
        if b.dim() < 1:
            raise ValueError("Numerator coefficients b must be at least 1D.")
        n = max(b.size(-1), a.size(-1) + 1)
        if b.size(-1) < n:
            b = F.pad(b, (0, n - b.size(-1)), value=0.0)
        if a.size(-1) < n - 1:
            a = F.pad(a, (0, n - 1 - a.size(-1)), value=0.0)
        A = companion(a).mT
        B = b[..., 1:] - b[..., :1] * a
    else:
        raise ValueError("Numerator coefficients b must be provided for transpose=True.")

    IminusA = torch.eye(n - 1, device=a.device, dtype=a.dtype) - A

    if IminusA.ndim == 2 and B.ndim > 1:
        IminusA = IminusA.expand(*B.shape[:-1], -1, -1)
    zi = torch.linalg.solve(IminusA, B)
    return zi


def fir(
    b: Tensor,
    x: Tensor,
    zi: Tensor | None = None,
    transpose: bool = True,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Filter signals with batched FIR filters.

    This computes

    .. math::
        y[n] = \sum_{k=0}^{M} b_k\, x[n - k]

    with a grouped convolution, one filter per signal.

    Args:
        b (Tensor): filter coefficients :math:`b_0, \dots, b_M`, of shape
            :math:`(B, M + 1)`.
        x (Tensor): input signals, of shape :math:`(B, N)`.
        zi (Tensor, optional): initial state, of shape :math:`(B, M)`. With
            ``transpose=True``, it is the transposed direct-form state, the
            same as SciPy's ``zi`` and :func:`lfiltic`; with
            ``transpose=False``, the past inputs newest first,
            :math:`x[-1], \dots, x[-M]`. Default: ``None``, all zero.
        transpose (bool, optional): use the transposed direct form
            (:func:`torch.nn.functional.conv_transpose1d`) if ``True``, or the
            direct form (:func:`torch.nn.functional.conv1d`) if ``False``.
            Default: ``True``.

    Returns:
        Tensor or tuple of Tensor: the filtered signals, of shape
        :math:`(B, N)`, and with :attr:`zi`, the final state in the same
        convention, of shape :math:`(B, M)`.

    Raises:
        ValueError: if :attr:`b`, :attr:`x` or :attr:`zi` is not 2-D or
            their shapes do not match.

    Note:
        Unlike :func:`lfilter`, :attr:`b` and :attr:`x` must both be 2-D,
        with one filter per signal.

    Example::

        >>> from philtorch.lti import fir
        >>> fir(torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0, 2.0, 3.0]]))
        tensor([[1., 3., 5.]])
    """
    if b.dim() != 2:
        raise ValueError("Numerator coefficients b must be 2D.")
    if x.dim() != 2:
        raise ValueError("Input signal x must be 2D.")
    B, N = x.shape
    if b.shape[0] != B:
        raise ValueError("The first dimension of b must match the batch size of x.")
    M = b.size(1) - 1

    if zi is not None:
        if zi.dim() != 2:
            raise ValueError("Initial conditions zi must be 2D.")
        if zi.size(0) != B:
            raise ValueError("The first dimension of zi must match the batch size.")
        if zi.size(1) != M:
            raise ValueError("The second dimension of zi must match the filter order.")

    if N == 0:
        # The convolutions reject empty signals. With no steps the output is
        # empty and the state unchanged; the empty slices keep b, x and zi in
        # the graph, so each gets a zero gradient.
        gain = b[:, :0].sum(-1, keepdim=True)
        if zi is None:
            return x + gain
        return x + gain + zi[:, :0], zi + gain

    if transpose:
        y = F.conv_transpose1d(
            x.unsqueeze(0),
            b.unsqueeze(1),
            stride=1,
            groups=B,
        ).squeeze(0)
        # y holds N + M samples; slice by N, as -M is empty when M = 0.
        if zi is not None:
            # zi adds to the first M samples, which reach into the final state
            # when N < M.
            y = y + F.pad(zi, (0, N))
            return y[:, :N], y[:, N:]
        return y[:, :N]

    if zi is None:
        zf = None
        padded_x = F.pad(x, (M, 0))
    else:
        padded_x = torch.cat([zi.flip(1), x], dim=1)
        zf = padded_x[:, N:].flip(1)

    y = F.conv1d(
        padded_x.unsqueeze(0),
        b.flip(1).unsqueeze(1),
        groups=B,
    ).squeeze(0)
    if zf is not None:
        return y, zf
    return y


def lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Tensor | None = None,
    form: str = "tdf2",
    backend: str = "ssm",
    **kwargs,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Filter signals with batched IIR filters.

    Each filter computes the difference equation

    .. math::
        y[n] = \sum_{k=0}^{M_b} b_k\, x[n - k]
             - \sum_{k=1}^{M_a} a_k\, y[n - k],

    whose transfer function is

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + \cdots + b_{M_b} z^{-M_b}}
                    {1 + a_1 z^{-1} + \cdots + a_{M_a} z^{-M_a}}.

    Filtering runs along the last dimension. :attr:`b`, :attr:`a` and
    :attr:`x` are batched along the first dimension, and coefficients without
    one are shared by every signal. If all inputs are unbatched, so are the
    outputs.

    Args:
        b (Tensor): numerator coefficients :math:`b_0, \dots, b_{M_b}`, of
            shape :math:`(B, M_b + 1)` or :math:`(M_b + 1)`. A batch of one,
            :math:`(1, M_b + 1)`, is shared by every signal of a batched
            :attr:`x`.
        a (Tensor): denominator coefficients :math:`a_1, \dots, a_{M_a}`,
            without the leading 1, of shape :math:`(B, M_a)` or :math:`(M_a)`.
            :math:`M_a` may be 0, for an FIR filter.
        x (Tensor): input signals, of shape :math:`(B, N)` or :math:`(N)`.
        zi (Tensor, optional): initial state, of shape :math:`(B, M)` or
            :math:`(M)` and the dtype of :attr:`x`, where
            :math:`M = \max(M_a, M_b)`, or :math:`M_a` with
            ``backend="diag_ssm"``. For ``form="tdf2"`` it is SciPy's ``zi``
            (see :func:`lfiltic` and :func:`lfilter_zi`); for ``form="df2"``,
            the direct form II state. ``"df1"`` and ``"tdf1"`` do not take
            it, nor does ``"diag_ssm"`` when :math:`M_b > M_a`.
            Default: ``None``, all zero.
        form (str, optional): the filter structure: ``"df2"`` or ``"tdf2"``,
            direct form II or its transpose, or ``"df1"`` or ``"tdf1"``,
            direct form I or its transpose. Default: ``"tdf2"``.
        backend (str, optional): ``"ssm"`` runs the filter as a state-space
            model of the :func:`~philtorch.mat.companion` matrix (see
            :func:`state_space`). ``"diag_ssm"`` diagonalizes it and runs one
            first-order recurrence per pole (see :func:`diag_state_space`);
            it supports ``"df2"`` and ``"tdf2"`` only. Default: ``"ssm"``.
        **kwargs: passed to the backend: ``unroll_factor`` (see
            :func:`state_space`) and, for ``"diag_ssm"``, ``L``, ``V`` and
            ``Vinv`` (see :func:`diag_state_space`) and ``delayed_form``. When
            :math:`M_b > M_a`, ``"diag_ssm"`` splits off an FIR part;
            ``delayed_form=True`` delays the recursive part by
            :math:`M_b - M_a + 1` samples instead. Both give the same output.

    Returns:
        Tensor or tuple of Tensor: the filtered signals, of the shape of
        :attr:`x`, and with :attr:`zi`, the final state, of shape
        :math:`(B, M)`, or :math:`(M)` when :attr:`b`, :attr:`a`, :attr:`x`
        and :attr:`zi` are all unbatched; a shared :attr:`zi` with batched
        signals still gives one final state per signal. A real filter gives
        real outputs with either backend.

    Raises:
        ValueError: if :attr:`form` or :attr:`backend` is unknown,
            :attr:`b` or :attr:`a` is not 1-D or 2-D, :attr:`x` has more
            than 2 dimensions, :attr:`zi` is not 1-D or 2-D or is given to a
            form or backend that does not take it, ``unroll_factor`` is below
            1, a batched :attr:`b` has a batch size other than 1 or that of
            :attr:`x`, or a non-empty :attr:`a` or :attr:`zi` has a dtype or
            device other than that of :attr:`x`. For an empty :attr:`a`, also
            if :attr:`b` has a dtype or device other than that of :attr:`x`.
            For a pure gain (one coefficient in :attr:`b` and an empty
            :attr:`a`), and for any empty :attr:`a` with
            ``backend="diag_ssm"``, also if a batched :attr:`a` or :attr:`zi`
            does not match the batch size of :attr:`x`; for a pure gain, also
            if :attr:`zi` has states.
        TypeError: if a keyword argument is not one the backend takes, or
            ``unroll_factor`` is not an integer.

    Note:
        Unlike :func:`scipy.signal.lfilter`, :attr:`a` omits the leading
        :math:`a_0`, which is taken to be 1, so normalize the coefficients
        first. There is no ``axis`` argument.

    Example::

        >>> from philtorch.lti import lfilter
        >>> # y[n] = 0.5 x[n] + 0.5 y[n - 1]
        >>> b, a = torch.tensor([0.5]), torch.tensor([-0.5])
        >>> lfilter(b, a, torch.tensor([1.0, 0.0, 0.0, 0.0]))
        tensor([0.5000, 0.2500, 0.1250, 0.0625])
    """

    squeeze_first = (
        (x.dim() == 1) & (b.dim() == 1) & (a.dim() == 1) & ((zi is None) or (zi.dim() == 1))
    )
    if x.dim() == 2 and b.dim() == 2 and b.size(0) == 1:
        b = b[0]  # a batch of one is shared by every signal
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() != 2:
        raise ValueError("Input signal x must be 1D or 2D.")

    if b.dim() not in (1, 2):
        raise ValueError("Numerator coefficients b must be 1D or 2D.")
    if b.dim() == 2 and b.size(0) != x.size(0):
        raise ValueError(
            "Batch size of b must be 1 or match batch size of x, "
            f"got b: {b.size(0)}, x: {x.size(0)}"
        )
    if a.dim() not in (1, 2):
        raise ValueError("Denominator coefficients a must be 1D or 2D.")
    # a, or b when a is empty (an empty a holds no values), and zi must share
    # x's dtype and device.
    coeffs, name = (b, "b") if a.size(-1) == 0 else (a, "a")
    if any(t is not None and (t.dtype != x.dtype or t.device != x.device) for t in (coeffs, zi)):
        raise ValueError(
            f"{name}, zi and x must share a dtype and a device, got "
            f"{[(t.dtype, str(t.device)) for t in (coeffs, zi, x) if t is not None]}"
        )

    match backend:
        case "ssm":
            y = _ssm_lfilter(b, a, x, zi, form=form, **kwargs)
        case "diag_ssm":
            y = _diag_ssm_lfilter(b, a, x, zi, form=form, **kwargs)
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    if isinstance(y, tuple):
        y, zf = y
        if squeeze_first:
            y = y.squeeze(0)
            zf = zf.squeeze(0)
        return y, zf
    return y.squeeze(0) if squeeze_first else y


# The keyword arguments the general path forwards: ``state_space`` and
# ``state_space_recursion`` take only ``unroll_factor`` of those that
# ``lfilter`` leaves free, and ``diag_state_space`` also ``L``, ``V`` and
# ``Vinv`` (``delayed_form`` is a parameter of ``_diag_ssm_lfilter``).
_SSM_KWARGS = ("unroll_factor",)
_DIAG_SSM_KWARGS = ("L", "V", "Vinv", "unroll_factor")


def _check_kwargs(kwargs: dict, accepted: tuple[str, ...]) -> None:
    """Raise the ``TypeError`` of the general path for an unknown keyword."""
    for name in kwargs:
        if name not in accepted:
            raise TypeError(f"lfilter() got an unexpected keyword argument {name!r}")


def _check_unroll_factor(kwargs: dict) -> None:
    """Raise the errors of the general path for a bad ``unroll_factor``."""
    unroll_factor = kwargs.get("unroll_factor")
    if unroll_factor is None:
        return
    if unroll_factor < 1:
        raise ValueError(f"unroll_factor must be None or at least 1, got {unroll_factor}")
    # Block sizes are integers: the general path fails on a float in a pad.
    try:
        operator.index(unroll_factor)
    except TypeError:
        raise TypeError(
            f"unroll_factor must be None or an integer, got {unroll_factor!r}"
        ) from None


def _check_batch_sizes(a: Tensor, x: Tensor, zi: Tensor | None) -> None:
    """Check that a batched a and zi match x's batch size; lfilter checks b."""
    for name, t in (("a", a), ("zi", zi)):
        if t is not None and t.dim() == 2 and t.size(0) != x.size(0):
            raise ValueError(
                f"Batch size of {name} must match batch size of x, "
                f"got {name}: {t.size(0)}, x: {x.size(0)}"
            )


def _ssm_lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Tensor | None = None,
    form: str = "df2",
    **kwargs,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run :func:`lfilter` with the ``"ssm"`` backend."""
    if b.size(-1) < a.size(-1) + 1:
        b = F.pad(b, (0, a.size(-1) + 1 - b.size(-1)))
    elif b.size(-1) > a.size(-1) + 1:
        a = F.pad(a, (0, b.size(-1) - a.size(-1) - 1))

    if zi is not None and form in ("df1", "tdf1"):
        raise ValueError(f"form={form!r} does not take zi; use 'df2' or 'tdf2'.")

    if a.size(-1) == 0 and form in ("df2", "tdf2", "df1", "tdf1"):
        # A pure gain has no state: the output is b0 * x, and the final state
        # is empty. The state-space functions of the general path would have
        # checked the keyword arguments, unroll_factor and the batch sizes.
        _check_kwargs(kwargs, _SSM_KWARGS)
        _check_unroll_factor(kwargs)
        _check_batch_sizes(a, x, zi)
        if zi is not None and zi.dim() not in (1, 2):
            raise ValueError(f"Initial conditions zi must be 1D or 2D, got {zi.shape}")
        if zi is not None and zi.size(-1) != 0:
            raise ValueError(f"zi must have 0 states for a pure gain, got {zi.size(-1)}.")
        # The empty a and the empty final state stay in the graph, so a and zi
        # get zero gradients, as on the general path.
        y = b[..., :1] * x + a.sum(-1, keepdim=True).to(x)
        return y if zi is None else (y, y[:, :0] + zi.expand(y.size(0), 0))

    A = companion(a)

    match form:
        case "df2":
            b0 = b[..., :1]  # First coefficient of the FIR filter
            C = b[..., 1:] - b0 * a
            D = b[..., 0]
            filt = partial(
                state_space,
                A,
                B=None,
                C=C,
                D=D,
                zi=zi,
                out_idx=None,
                **kwargs,
            )
        case "tdf2":
            b0 = b[..., :1]  # First coefficient of the FIR filter
            B = b[..., 1:] - b0 * a
            D = b[..., 0]
            filt = partial(
                state_space,
                A.mT,
                B=B,
                C=None,
                D=D,
                zi=zi,
                out_idx=0,
                **kwargs,
            )
        case "df1":
            zi = x.new_zeros((x.size(0), A.size(-1)))
            filt = chain_functions(
                partial(fir, b.broadcast_to((x.size(0), -1))),
                partial(state_space_recursion, A, zi, out_idx=0, **kwargs),
            )
        case "tdf1":
            zi = x.new_zeros((x.size(0), A.size(-1)))
            filt = chain_functions(
                partial(state_space_recursion, A.mT, zi, out_idx=0, **kwargs),
                partial(fir, b.broadcast_to((x.size(0), -1)), transpose=True),
            )
        case _:
            raise ValueError(f"Unknown filter form: {form}")

    return filt(x)


def _diag_ssm_lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Tensor | None = None,
    form: str = "df2",
    delayed_form: bool = False,
    **kwargs,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run :func:`lfilter` with the ``"diag_ssm"`` backend."""
    # The eigendecomposition is complex for complex poles, but a real filter
    # still has a real output and state.
    real = not any(t.is_complex() for t in (b, a, x) + (() if zi is None else (zi,)))

    if form not in ("df2", "tdf2"):
        raise ValueError(f"Unknown filter form: {form}")

    too_long = b.size(-1) > a.size(-1) + 1
    if too_long and zi is not None:
        raise ValueError(
            "backend='diag_ssm' does not take zi when b has more than "
            "len(a) + 1 coefficients; use backend='ssm'."
        )

    if a.size(-1) == 0:
        # With no poles, the filter is the FIR filter b. The general path
        # would have checked the keyword arguments, unroll_factor and the
        # batch sizes.
        _check_kwargs(kwargs, _DIAG_SSM_KWARGS)
        _check_unroll_factor(kwargs)
        _check_batch_sizes(a, x, zi)
        if b.size(-1) == 0:
            # fir needs a tap: no coefficient means a zero filter, as with
            # backend="ssm", which pads b to len(a) + 1.
            b = F.pad(b, (0, 1))
        if zi is not None and zi.dim() == 1:
            zi = zi.unsqueeze(0).expand(x.size(0), -1)
        # The empty a stays in the graph, so it gets a zero gradient.
        no_poles = a.sum(-1, keepdim=True).to(x)
        result = fir(b.broadcast_to((x.size(0), -1)), x, zi)
        if zi is None:
            return result + no_poles
        return result[0] + no_poles, result[1]

    if too_long:
        # fir takes one filter per signal.
        b = b.broadcast_to((x.size(0), b.size(-1)))
        if delayed_form:
            q, r = polydiv(b, F.pad(a, (1, 0), value=1.0))
            direct_filt = partial(fir, q)
            delay = b.size(-1) - a.size(-1)
            b = r
        else:
            rev_q, rev_r = polydiv(b.flip(-1), F.pad(a.flip(-1), (0, 1), value=1.0))
            direct_filt = partial(fir, rev_q.flip(-1))
            delay = 0
            b = rev_r.flip(-1)
    else:
        direct_filt = None
        delay = 0

    if b.size(-1) < a.size(-1) + 1:
        b = F.pad(b, (0, a.size(-1) + 1 - b.size(-1)))

    A = companion(a)

    match form:
        case "df2":
            b0 = b[..., :1]  # First coefficient of the FIR filter
            C = b[..., 1:] - b0 * a
            D = b[..., 0]
            filt = partial(
                diag_state_space,
                A=A,
                B=None,
                C=C,
                D=D,
                zi=zi,
                **kwargs,
            )
        case "tdf2":
            b0 = b[..., :1]  # First coefficient of the FIR filter
            B = b[..., 1:] - b0 * a
            D = b[..., 0]
            filt = partial(
                diag_state_space,
                A=A.mT,
                B=B,
                C=None,
                D=D,
                zi=zi,
                out_idx=0,
                **kwargs,
            )

    results = filt(x)
    if isinstance(results, tuple):
        y, zf = results
        return (y.real, zf.real) if real else (y, zf)
    y = results.real if real else results

    if delay > 0:
        # Delay by padding and truncating, which also works when delay > N.
        y = F.pad(y, (delay, 0))[:, : y.size(1)]

    if direct_filt is not None:
        y = y + direct_filt(x)

    return y


def filtfilt(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    padmode: str | None = "replicate",
    padlen: int | None = None,
    method: str = "pad",
    irlen: int | None = None,
    form: str = "tdf2",
    **kwargs,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Apply a filter forward and backward for zero phase.

    This filters :attr:`x` with :func:`lfilter`, then filters the
    time-reversed result again, so the output has zero phase and the squared
    magnitude response of the filter. Each pass starts from the steady state
    given by :func:`lfilter_zi`, scaled by its first sample, and the signal is
    extended at both ends to reduce transients.

    Args:
        b (Tensor): numerator coefficients, of shape :math:`(B, M_b + 1)` or
            :math:`(M_b + 1)`.
        a (Tensor): denominator coefficients without the leading 1, of shape
            :math:`(B, M_a)` or :math:`(M_a)`.
        x (Tensor): input signals, of shape :math:`(B, N)` or :math:`(N)`.
        padmode (str or None, optional): how to extend the signal, as a mode of
            :func:`torch.nn.functional.pad`: ``"reflect"`` is SciPy's
            ``padtype="even"``, ``"replicate"`` is SciPy's ``"constant"``, and
            ``"constant"`` pads zeros. ``None`` disables the extension.
            Default: ``"replicate"``.
        padlen (int or None, optional): the number of samples to add at each
            end, less than :math:`N`. Default: ``None``, which uses
            :math:`3 \max(M_a + 1, M_b + 1)`, as SciPy does.
        method (str, optional): only ``"pad"`` is implemented; ``"gust"``
            raises :class:`NotImplementedError`. Default: ``"pad"``.
        irlen (int or None, optional): unused, for compatibility with SciPy.
            Default: ``None``.
        form (str, optional): ``"df2"`` or ``"tdf2"``; see :func:`lfilter`.
            Default: ``"tdf2"``.
        **kwargs: passed to :func:`lfilter`, e.g. ``backend`` or
            ``unroll_factor``.

    Returns:
        Tensor: the filtered signals, of the shape of :attr:`x`.

    Raises:
        ValueError: if :attr:`method` is not ``"pad"`` or ``"gust"``,
            :attr:`x` is not longer than the padding, or :attr:`form` is not
            ``"df2"`` or ``"tdf2"``.
        NotImplementedError: if :attr:`method` is ``"gust"``.

    Note:
        SciPy's default ``padtype="odd"`` has no equivalent here.

    Example::

        >>> from philtorch.lti import filtfilt
        >>> b, a = torch.tensor([0.25, 0.5, 0.25]), torch.tensor([-0.2])
        >>> x = torch.zeros(21)
        >>> x[10] = 1.0
        >>> y = filtfilt(b, a, x)
        >>> # Zero phase: a symmetric input gives a symmetric output.
        >>> torch.allclose(y, y.flip(0))
        True
    """
    if method not in ("pad", "gust"):
        raise ValueError("Method must be either 'pad' or 'gust'.")

    if method == "gust":
        raise NotImplementedError("Gustafsson's method is not implemented yet.")
    if form not in ("df2", "tdf2"):
        raise ValueError(f"filtfilt needs form 'df2' or 'tdf2', got {form!r}.")

    if padmode is None:
        padlen = 0

    ntaps = max(b.size(-1), a.size(-1) + 1)
    if padlen is None:
        edge = 3 * ntaps
    else:
        edge = padlen

    if x.size(-1) <= edge:
        raise ValueError(
            f"Input signal length {x.size(-1)} must be greater than pad length {edge}."
        )

    if edge > 0 and padmode is not None:
        ext = F.pad(
            x.view(-1, x.size(-1)),
            (edge, edge),
            mode=padmode,
        )
        if x.dim() == 1:
            ext = ext.squeeze(0)
    else:
        ext = x

    zi = lfilter_zi(a, b, transpose=(form == "tdf2"))
    x0 = ext[..., :1]

    y, _ = lfilter(b, a, ext, zi=zi * x0, form=form, **kwargs)
    y0 = y[..., -1:]

    y, _ = lfilter(b, a, y.flip(-1), zi=zi * y0, form=form, **kwargs)

    if edge > 0:
        y = y[..., edge:-edge]

    return y.flip(-1)
