"""IIR and FIR filtering with time-invariant coefficients."""

import operator
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from .. import _ssm
from ..mat import companion, vandermonde
from ..poly import polydiv
from ..utils import _check_shared, _squeeze_first_result, chain_functions
from .recur import linear_recurrence
from .ssm import diag_state_space, state_space, state_space_recursion


def comb_filter(a: Tensor, delay: int, x: Tensor, zi: Tensor | None = None, **kwargs) -> Tensor:
    r"""Filter signals with an all-pole comb filter.

    This computes

    .. math::
        y[n] = x[n] - a\, y[n - D],

    the filter :math:`1 / (1 + a z^{-D})` with delay :math:`D`.

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
            wrong shape, :attr:`x` is not 2-D, or :attr:`a` or :attr:`zi`
            doesn't share a dtype and a device with :attr:`x`.

    Example::

        >>> from philtorch.lti import comb_filter
        >>> # y[n] = x[n] + 0.5 y[n - 2]
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
    _check_shared(a=a, zi=zi, x=x)

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
        ValueError: if :attr:`b` or :attr:`a` is 0-D, or :attr:`b`,
            :attr:`a` and :attr:`y` (and :attr:`x`) don't share a dtype and a
            device.

    Example::

        >>> from philtorch.lti import lfilter, lfiltic
        >>> # y[n] = 0.5 x[n] + 0.5 x[n - 1] + 0.5 y[n - 1]
        >>> b, a = torch.tensor([0.5, 0.5]), torch.tensor([-0.5])
        >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y = lfilter(b, a, x)
        >>> y[2:]
        tensor([3.3750, 5.1875])
        >>> # Continue from the second sample. With longer histories, pass
        >>> # the past values newest first, e.g. y[:k].flip(-1).
        >>> zi = lfiltic(b, a, y[1:2], x[1:2])
        >>> y_rest, _ = lfilter(b, a, x[2:], zi=zi)
        >>> y_rest
        tensor([3.3750, 5.1875])
    """
    if b.dim() < 1:
        raise ValueError("Numerator coefficients b must be at least 1D.")
    if a.dim() < 1:
        raise ValueError("Denominator coefficients a must be at least 1D.")
    _check_shared(b=b, a=a, y=y, x=x)

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
            :attr:`b` is ``None`` or 0-D, or doesn't share a dtype and a
            device with :attr:`a`.

    Note:
        Unlike :func:`scipy.signal.lfilter_zi`, the arguments are
        ``(a, b)``.

    Example::

        >>> from philtorch.lti import lfilter, lfilter_zi
        >>> # 0.5 / (1 - 0.5 z^-1)
        >>> b, a = torch.tensor([0.5]), torch.tensor([-0.5])
        >>> y, _ = lfilter(b, a, torch.ones(4), zi=lfilter_zi(a, b))
        >>> y
        tensor([1., 1., 1., 1.])
    """
    if a.dim() < 1:
        raise ValueError("Denominator coefficients a must be at least 1D.")
    _check_shared(b=b if transpose else None, a=a)

    if not transpose:
        n = a.size(-1) + 1
        A = companion(a)
        B = a.new_zeros(n - 1)
        B[:1] = 1.0  # a slice, as an empty a has no first state
    elif b is not None:
        if b.dim() < 1:
            raise ValueError("Numerator coefficients b must be at least 1D.")
        b, a, B, _ = _ssm.direct_form(b, a)
        n = a.size(-1) + 1
        A = companion(a).mT
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
        y[n] = \sum_{i=0}^{M} b_i\, x[n - i].

    Args:
        b (Tensor): filter coefficients :math:`b_0, \dots, b_M`, of shape
            :math:`(B, M + 1)`.
        x (Tensor): input signals, of shape :math:`(B, N)`.
        zi (Tensor, optional): initial state, of shape :math:`(B, M)`. With
            ``transpose=True``, it is the transposed direct-form state, the
            same as SciPy's ``zi`` and :func:`lfiltic`; with
            ``transpose=False``, the past inputs newest first,
            :math:`x[-1], \dots, x[-M]`. Default: ``None``, all zero.
        transpose (bool, optional): use the transposed direct form if
            ``True``, or the direct form if ``False``. Default: ``True``.

    Returns:
        Tensor or tuple of Tensor: the filtered signals, of shape
        :math:`(B, N)`, and with :attr:`zi`, the final state in the same
        convention, of shape :math:`(B, M)`, empty when :math:`M = 0`.

    Raises:
        ValueError: if :attr:`b`, :attr:`x` or :attr:`zi` is not 2-D,
            their shapes do not match, or :attr:`b` or :attr:`zi` doesn't
            share a dtype and a device with :attr:`x`.

    Example::

        >>> from philtorch.lti import fir
        >>> # y[n] = x[n] + x[n - 1]
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
    _check_shared(b=b, zi=zi, x=x)

    if N == 0:
        # The convolutions reject empty signals.
        y = x.new_empty(x.shape)
        return y if zi is None else (y, zi.clone())

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
        y[n] = \sum_{i=0}^{M_b} b_i\, x[n - i]
             - \sum_{j=1}^{M_a} a_j\, y[n - j],

    whose transfer function is

    .. math::
        H(z) = \frac{B(z)}{A(z)}
             = \frac{b_0 + b_1 z^{-1} + \cdots + b_{M_b} z^{-M_b}}
                    {1 + a_1 z^{-1} + \cdots + a_{M_a} z^{-M_a}}.

    Filtering runs along the last dimension. :attr:`b`, :attr:`a` and
    :attr:`x` are batched along the first dimension, and coefficients without
    one are shared by every signal. If all inputs are unbatched, so are the
    outputs.

    Args:
        b (Tensor): numerator coefficients :math:`b_0, \dots, b_{M_b}`, of
            shape :math:`(B, M_b + 1)` or :math:`(M_b + 1)`.
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
            :func:`state_space`) and, for ``"diag_ssm"``, the poles ``L``,
            the eigenvalues of the state matrix, with its eigenvectors ``V``
            or ``Vinv`` (see :func:`diag_state_space`), computed from ``L``
            when neither is given, and ``delayed_form``. When
            :math:`M_b > M_a`, ``"diag_ssm"`` splits off an FIR part;
            ``delayed_form=True`` delays the recursive part by
            :math:`M_b - M_a + 1` samples instead. Both give the same output.

    Returns:
        Tensor or tuple of Tensor: the filtered signals, of the shape of
        :attr:`x`, and with :attr:`zi`, the final state, of shape
        :math:`(B, M)`, or :math:`(M)` when :attr:`b`, :attr:`a`, :attr:`x`
        and :attr:`zi` are all unbatched; a shared :attr:`zi` with batched
        signals still gives one final state per signal, empty for a pure
        gain. A real filter gives real outputs with either backend.

    Raises:
        ValueError: if :attr:`form` or :attr:`backend` is unknown,
            :attr:`b` or :attr:`a` is not 1-D or 2-D, :attr:`x` has more
            than 2 dimensions, :attr:`zi` is not 1-D or 2-D or is given to a
            form or backend that does not take it, ``unroll_factor`` is below
            1, a batched :attr:`b`, :attr:`a` or :attr:`zi` has a batch size
            other than that of :attr:`x`, :attr:`b`, a non-empty :attr:`a` or
            :attr:`zi` has a dtype or device other than that of :attr:`x`,
            ``backend`` is ``"diag_ssm"`` and :attr:`x` is neither
            floating-point nor complex, or :attr:`zi` has states for a pure
            gain (one coefficient in :attr:`b` and an empty :attr:`a`).
        TypeError: if a keyword argument is not one the backend takes, or
            ``unroll_factor`` is not an integer.

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
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() != 2:
        raise ValueError("Input signal x must be 1D or 2D.")

    if b.dim() not in (1, 2):
        raise ValueError("Numerator coefficients b must be 1D or 2D.")
    if a.dim() not in (1, 2):
        raise ValueError("Denominator coefficients a must be 1D or 2D.")
    for name, t in (("b", b), ("a", a), ("zi", zi)):
        if t is not None and t.dim() == 2 and t.size(0) != x.size(0):
            raise ValueError(
                f"Batch size of {name} must match batch size of x, "
                f"got {name}: {t.size(0)}, x: {x.size(0)}"
            )
    # An empty a holds no values, so it may have any dtype, as a bare
    # ``torch.tensor([])`` does.
    _check_shared(b=b, a=a if a.size(-1) else None, zi=zi, x=x)
    if a.size(-1) == 0:
        a = x.new_zeros(a.shape)

    match backend:
        case "ssm":
            y = _ssm_lfilter(b, a, x, zi, form=form, **kwargs)
        case "diag_ssm":
            y = _diag_ssm_lfilter(b, a, x, zi, form=form, **kwargs)
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    return _squeeze_first_result(y, squeeze_first)


def _check_sos(sos: Tensor) -> None:
    if sos.dim() != 2:
        raise ValueError(
            "Sections sos must be 2D: (K, 5) for one cascade, or (S, 5) with a "
            f"batch_index for one per row, got shape {tuple(sos.shape)}."
        )
    if sos.size(-1) != 5:
        raise ValueError(f"The last dimension of sos must be 5, got {sos.size(-1)}.")
    if sos.size(-2) == 0:
        raise ValueError("sos must have at least one section.")


def _sections_by_row(
    sos: Tensor, batch_index: Tensor, batch_size: int | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Group flat sections by their row, keeping their listed order in each.

    Args:
        sos: the sections, (S, 5), checked by :func:`_check_sos`.
        batch_index: the row of each section, (S,).
        batch_size: the number of rows B; ``None`` takes max(batch_index) + 1.

    Returns:
        The sections padded with identity sections [1, 0, 0, 0, 0] to
        (B, K, 5), K being the most sections of a row; each row's number of
        sections, (B,); and each section's place in the padded rows flattened
        to (B * K,), (S,).

    Raises:
        ValueError: if ``batch_index`` is not of shape (S,), not int32 or int64,
            not on sos's device, or out of range.
    """
    S = sos.size(0)
    if batch_index.shape != (S,):
        raise ValueError(f"batch_index must be of shape ({S},), got {tuple(batch_index.shape)}.")
    if batch_index.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"batch_index must be int32 or int64, got {batch_index.dtype}.")
    if batch_index.device != sos.device:
        raise ValueError(
            f"batch_index must be on sos's device, {sos.device}, got {batch_index.device}."
        )
    if (batch_index < 0).any():
        raise ValueError("batch_index must not be negative.")
    if batch_size is None:
        batch_size = int(batch_index.max()) + 1
    elif (batch_index >= batch_size).any():
        raise ValueError(f"batch_index must be from 0 to B - 1 = {batch_size - 1}.")
    batch_index = batch_index.long()
    counts = torch.bincount(batch_index, minlength=batch_size)
    K = int(counts.max())
    order = torch.argsort(batch_index, stable=True)
    rows = batch_index[order]
    position = torch.arange(S, device=sos.device) - (counts.cumsum(0) - counts)[rows]
    place = torch.empty_like(order).index_copy_(0, order, rows * K + position)
    identity = sos.new_tensor([1.0, 0.0, 0.0, 0.0, 0.0]).expand(batch_size * K, 5)
    padded = identity.index_copy(0, place, sos).view(batch_size, K, 5)
    return padded, counts, place


def sosfilt(
    sos: Tensor, x: Tensor, zi: Tensor | None = None, batch_index: Tensor | None = None
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Filter signals with cascades of second-order sections.

    A signal goes through the :math:`K` sections of its cascade one after
    another, so that the cascade's transfer function is the product of the
    sections':

    .. math::
        H(z) = \prod_{l=1}^{K}
            \frac{b_{l0} + b_{l1} z^{-1} + b_{l2} z^{-2}}
                 {1 + a_{l1} z^{-1} + a_{l2} z^{-2}}.

    Without :attr:`batch_index`, every signal goes through all the sections.
    With it, a signal's cascade is the sections listed for it, in the order
    they are listed, and :math:`K` is their number; a signal without a
    section comes out unchanged.

    Args:
        sos (Tensor): the sections, of shape :math:`(K, 5)` for one cascade,
            :math:`K \ge 1`, or :math:`(S, 5)` with :attr:`batch_index`,
            :math:`S \ge 1`. Each row is one section,
            :math:`b_{l0}, b_{l1}, b_{l2}, a_{l1}, a_{l2}`, :math:`l` being its
            place in its cascade.
        x (Tensor): input signals, of shape :math:`(B, N)` or :math:`(N)`.
        zi (Tensor, optional): the initial state of each section, of the
            dtype of :attr:`x`: of shape :math:`(B, K, 2)` for a 2-D
            :attr:`x` and :math:`(K, 2)` for a 1-D one, or :math:`(S, 2)`
            with :attr:`batch_index`. Each section's state is that of SciPy's
            ``zi``, of the transposed direct form II. Without
            :attr:`batch_index` and with a 2-D :attr:`x`, expand
            :func:`sosfilt_zi`'s :math:`(K, 2)` states to :math:`(B, K, 2)`.
            Default: ``None``, all zero.
        batch_index (Tensor, optional): the signal each section filters, of
            shape :math:`(S)`, int32 or int64 on :attr:`x`'s device, from 0 to
            :math:`B - 1`, where :math:`B = 1` for a 1-D :attr:`x`. Sections of
            the same signal need not be consecutive rows; they run in the
            order they are listed. Default: ``None``, one cascade of all the
            sections for every signal.

    Returns:
        Tensor or tuple of Tensor: the filtered signals, of the shape of
        :attr:`x`, and with :attr:`zi`, the final states, of the shape of
        :attr:`zi`. Pass the final states as the next chunk's :attr:`zi` to
        filter a signal in chunks.

    Raises:
        ValueError: if :attr:`sos` is not 2-D, its last dimension is not 5, or
            it has no section, :attr:`x` is not 1-D or 2-D, :attr:`zi` is not
            of the shape given for it above, :attr:`batch_index` is not of
            shape :math:`(S)`, not int32 or int64, not on :attr:`x`'s device,
            or not from 0 to :math:`B - 1`, or :attr:`sos` or :attr:`zi` has a
            dtype or device other than that of :attr:`x`.

    Example::

        >>> from philtorch.lti import sosfilt
        >>> # two sections, y[n] = 0.5 x[n] + 0.5 y[n - 1] and a gain of 2
        >>> sos = torch.tensor([[0.5, 0.0, 0.0, -0.5, 0.0],
        ...                     [2.0, 0.0, 0.0, 0.0, 0.0]])
        >>> sosfilt(sos, torch.tensor([1.0, 0.0, 0.0, 0.0]))
        tensor([1.0000, 0.5000, 0.2500, 0.1250])
        >>> # the first section for signal 0, both for signal 1, none for
        >>> # signal 2, which comes out unchanged
        >>> batch_index = torch.tensor([0, 1, 1])
        >>> sos3 = torch.cat([sos[:1], sos])
        >>> x = torch.tensor([[1.0, 0.0, 0.0]] * 3)
        >>> sosfilt(sos3, x, batch_index=batch_index)
        tensor([[0.5000, 0.2500, 0.1250],
                [1.0000, 0.5000, 0.2500],
                [1.0000, 0.0000, 0.0000]])
        >>> # The step response of 12 sections of unit gain at DC, poles of
        >>> # radius 0.95, ends at 1: the cascade gets there, while the
        >>> # recursion of the expanded transfer function diverges.
        >>> from philtorch.lti import lfilter, sos2tf
        >>> a1 = -1.9 * torch.linspace(0.02, 0.1, 12).cos()
        >>> a2 = torch.full_like(a1, 0.9025)
        >>> b = (1 + a1 + a2) / 4
        >>> sos = torch.stack([b, 2 * b, b, a1, a2], -1)
        >>> x = torch.ones(1000)
        >>> sosfilt(sos, x)[-1]
        tensor(1.0000)
        >>> lfilter(*sos2tf(sos), x)[-1]
        tensor(nan)
    """
    _check_sos(sos)
    if x.dim() not in (1, 2):
        raise ValueError("Input signal x must be 1D or 2D.")
    batched = x.dim() == 2
    x = x if batched else x.unsqueeze(0)
    B = x.size(0)
    shared = batch_index is None
    K = sos.size(0)
    zshape = (B, K, 2) if shared and batched else (K, 2)
    if zi is not None and zi.shape != zshape:
        raise ValueError(f"zi must have shape {zshape}, got {tuple(zi.shape)}.")
    _check_shared(sos=sos, zi=zi, x=x)

    if shared:
        rows, live = sos, [B] * K
    else:
        # The rows sorted by their number of sections, most first, so that
        # the rows with a section at each position are a prefix; live counts
        # them at each position.
        padded, counts, place = _sections_by_row(sos, batch_index, B)
        order = counts.argsort(descending=True, stable=True)
        rows, x = padded[order], x[order]
        positions = torch.arange(padded.size(1), device=counts.device)
        live = (counts > positions.unsqueeze(-1)).sum(-1).tolist()
    _, a, C, D = _ssm.direct_form(rows[..., :3], rows[..., 3:])
    A = companion(a)
    if not shared:
        # by position first, each position contiguous
        A, C, D = (t.movedim(1, 0).contiguous() for t in (A, C, D))
    if zi is not None:
        if shared:
            z = (zi if batched else zi.unsqueeze(0)).movedim(1, 0)
        else:
            z = zi.new_zeros(B * padded.size(1), 2).index_copy(0, place, zi)
            z = z.view(B, -1, 2)[order].movedim(1, 0).contiguous()

    y, done, zfs = x, [], []
    for pos, n in enumerate(live):
        if n < y.size(0):  # the rows without a section here are finished
            done.append(y[n:])
            y = y[:n]
        at = (lambda t: t[pos]) if shared else (lambda t: t[pos, :n])
        out = state_space(
            at(A).mT, y, B=at(C), D=at(D), zi=at(z) if zi is not None else None, out_idx=0
        )
        y, zf = out if zi is not None else (out, None)
        zfs.append(zf)
    y = torch.cat([y, *done[::-1]]) if done else y
    if shared:
        y = y if batched else y.squeeze(0)
        if zi is None:
            return y
        zf = torch.stack(zfs, 1)
        return y, zf if batched else zf.squeeze(0)
    inverse = order.argsort()
    y = y[inverse] if batched else y[inverse].squeeze(0)
    if zi is None:
        return y
    # the final states of each position, over the rows that ran it
    zf = torch.stack([F.pad(f, (0, 0, 0, B - f.size(0))) for f in zfs], 1)
    return y, zf[inverse].reshape(-1, 2)[place]


def sosfilt_zi(sos: Tensor, batch_index: Tensor | None = None) -> Tensor:
    r"""Return the initial states of :func:`sosfilt` for a steady step response.

    With these states, as with :func:`scipy.signal.sosfilt_zi`, filtering a
    constant input of 1 starts every section in steady state. Section
    :math:`l` gets the state :func:`lfilter_zi` gives for its own
    coefficients, scaled by :math:`G_1 G_2 \cdots G_{l-1}`, the gain at DC of
    the sections before it in its cascade, where section :math:`l`'s is

    .. math::
        G_l = \frac{b_{l0} + b_{l1} + b_{l2}}{1 + a_{l1} + a_{l2}}.

    With :attr:`batch_index`, a section's cascade is the sections listed for
    the same signal, in the order they are listed, :math:`l` counting places
    in it.

    Args:
        sos (Tensor): the sections, of shape :math:`(K, 5)` for one cascade,
            :math:`K \ge 1`, or :math:`(S, 5)` with :attr:`batch_index`,
            :math:`S \ge 1`, as in :func:`sosfilt`.
        batch_index (Tensor, optional): the signal each section filters, of
            shape :math:`(S)`, int32 or int64 on :attr:`sos`'s device and not
            negative, as in :func:`sosfilt`, which checks it against its
            signals. Default: ``None``, one cascade of all the sections.

    Returns:
        Tensor: the initial states, of :attr:`sos`'s dtype and of shape
        :math:`(K, 2)`, or :math:`(S, 2)` with :attr:`batch_index`, as
        :func:`sosfilt` takes them for a 1-D signal or with
        :attr:`batch_index`. For :math:`B` signals in a 2-D tensor and without
        :attr:`batch_index`, expand them to :math:`(B, K, 2)`.

    Raises:
        ValueError: if :attr:`sos` is not 2-D, its last dimension is not 5, or
            it has no section, or :attr:`batch_index` is not of shape
            :math:`(S)`, not int32 or int64, not on :attr:`sos`'s device, or
            negative.

    Example::

        >>> from philtorch.lti import sosfilt, sosfilt_zi
        >>> # 0.5 / (1 - 0.5 z^-1), then a gain of 2
        >>> sos = torch.tensor([[0.5, 0.0, 0.0, -0.5, 0.0],
        ...                     [2.0, 0.0, 0.0, 0.0, 0.0]])
        >>> y, _ = sosfilt(sos, torch.ones(4), zi=sosfilt_zi(sos))
        >>> y
        tensor([2., 2., 2., 2.])
        >>> # two sections of gain 2 at DC: in one cascade, the second
        >>> # section's state is doubled; as the only section of a second
        >>> # signal, it is not
        >>> sos = torch.tensor([[1.0, 0.0, 0.0, -0.5, 0.0]] * 2)
        >>> sosfilt_zi(sos)
        tensor([[1., 0.],
                [2., 0.]])
        >>> sosfilt_zi(sos, torch.tensor([0, 1]))
        tensor([[1., 0.],
                [1., 0.]])
    """
    _check_sos(sos)
    if batch_index is None:
        padded = sos
    else:
        padded, _, place = _sections_by_row(sos, batch_index)
    b, a = padded[..., :3], padded[..., 3:]
    gain = b.sum(-1) / (1 + a.sum(-1))
    # the gains of the sections before each one, the first having none
    before = torch.cumprod(F.pad(gain, (1, 0), value=1.0)[..., :-1], dim=-1)
    zi = lfilter_zi(a, b) * before.unsqueeze(-1)
    return zi if batch_index is None else zi.reshape(-1, 2)[place]


def pfefilt(
    r: Tensor,
    p: Tensor,
    g: Tensor,
    x: Tensor,
    zi: Tensor | None = None,
    batch_index: Tensor | None = None,
    multiplicity: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Filter signals with filters in parallel form.

    The filter is a partial fraction expansion, a sum of sections:

    .. math::
        H(z) = g + \sum_{i=1}^{U} \sum_{j=1}^{m_i}
            \frac{r_{ij} z^{-j}}{(1 - p_i z^{-1})^{j}},

    with :math:`U` poles :math:`p_i`, each of multiplicity :math:`m_i`, and
    :math:`M = m_1 + \dots + m_U` terms. :func:`sos2pfe` expands second-order
    sections into this form, with the :attr:`batch_index` and
    :attr:`multiplicity` to pass on. Term :math:`(i, j)` is
    :math:`r_{ij} s_{ij}`, where :math:`s_{ij}` is the output of the
    first-order section

    .. math::
        s_{ij}[n + 1] = p_i s_{ij}[n] + u_{ij}[n], \quad
        s_{ij}[0] = \text{zi}_{ij},

    whose input :math:`u_{ij}` is :math:`x` for :math:`j = 1` and
    :math:`s_{i,j-1}` for :math:`j > 1`: the powers of a repeated pole are a
    chain of sections, each delaying its input by one sample.

    A real :attr:`g` makes a real filter, which takes a real :attr:`x`. Each
    of its non-real poles stands for itself and its conjugate, whose terms
    have the conjugates of :math:`r_{ij}`, :math:`p_i` and
    :math:`\text{zi}_{ij}`, so that each pair of terms sums to a real one:
    list one pole of each conjugate pair, as :func:`sos2pfe` does, not both,
    as :func:`residuez` does. Its real poles need real residues and
    initial states. For a complex :attr:`x`, filter its real and imaginary
    parts. A complex :attr:`g` makes a complex filter, which lists every pole.

    Args:
        r (Tensor): the residues :math:`r_{ij}`, real or complex, of shape
            :math:`(M)`: each pole's :math:`m_i` residues in a row, by
            ascending :math:`j`, the poles in the order of :attr:`p`.
        p (Tensor): the poles, real or complex, of shape :math:`(U)`.
        g (Tensor): the direct gain, of shape :math:`()` or :math:`(B)`.
        x (Tensor): input signals, of shape :math:`(B, N)` or :math:`(N)`.
        zi (Tensor, optional): the initial states :math:`s_{ij}[0]`, real or
            complex with the precision of :attr:`x`, listed as :attr:`r` is,
            of shape :math:`(B, M)` for a 2-D :attr:`x` without
            :attr:`batch_index`, and :math:`(M)` otherwise. Default:
            ``None``, all zero.
        batch_index (Tensor, optional): the row of :attr:`x` each pole
            filters and adds to, of shape :math:`(U)`, from 0 to
            :math:`B - 1`, where :math:`B = 1` for a 1-D :attr:`x`. Poles may
            come in any order. Default: ``None``, one filter of all the poles
            for every row.
        multiplicity (Tensor, optional): the multiplicities :math:`m_i`, of
            shape :math:`(U)`, int32 or int64 and at least 1. Default:
            ``None``, every pole of multiplicity 1, so that :math:`M = U`.

    Returns:
        Tensor or tuple of Tensor: the filtered signals, of the shape of
        :attr:`x`, real when :attr:`g` is, and with :attr:`zi`, the final
        states :math:`s_{ij}[N]`, complex, of the shape of :attr:`zi`. Pass
        them as the next chunk's :attr:`zi` to filter a signal in chunks.

    Raises:
        ValueError: if :attr:`x` is not 1-D or 2-D, :attr:`g` is not 0-D or
            1-D, :attr:`p` is not 1-D, :attr:`batch_index` or
            :attr:`multiplicity` is not of shape :math:`(U)`, the length of
            :attr:`r` is not :math:`M`, :attr:`zi` or a batched :attr:`g` has
            the wrong shape, :attr:`batch_index` or :attr:`multiplicity` isn't
            int32 or int64 on :attr:`x`'s device, :attr:`batch_index` is out of
            range, :attr:`multiplicity` is below 1, the inputs differ in
            precision or device, are neither floating-point nor complex, or
            are bfloat16, which has no complex dtype, :attr:`g` is real and
            :attr:`x` complex, or a real pole has a complex residue or initial
            state.

    Example::

        >>> from philtorch.lti import pfefilt, sosfilt
        >>> # 1 / (1 - 0.5 z^-1) as a section and as a term
        >>> sos = torch.tensor([[1.0, 0.0, 0.0, -0.5, 0.0]])
        >>> x = torch.tensor([1.0, 0.0, 0.0, 0.0])
        >>> r = p = torch.tensor([0.5])
        >>> pfefilt(r, p, torch.tensor(1.0), x)
        tensor([1.0000, 0.5000, 0.2500, 0.1250])
        >>> sosfilt(sos, x)
        tensor([1.0000, 0.5000, 0.2500, 0.1250])
        >>> # a double pole: 1 / (1 - 0.5 z^-1)^2
        >>> #   = 1 + z^-1 / (1 - 0.5 z^-1) + 0.25 z^-2 / (1 - 0.5 z^-1)^2
        >>> r = torch.tensor([1.0, 0.25])
        >>> pfefilt(r, p, torch.tensor(1.0), x, multiplicity=torch.tensor([2]))
        tensor([1.0000, 1.0000, 0.7500, 0.5000])
    """
    if x.dim() not in (1, 2):
        raise ValueError("Input signal x must be 1D or 2D.")
    if g.dim() not in (0, 1):
        raise ValueError("Gain g must be 0D or 1D.")
    if p.dim() != 1:
        raise ValueError("Poles p must be 1D.")
    U = p.size(0)
    for name, t in (("batch_index", batch_index), ("multiplicity", multiplicity)):
        if t is not None and t.shape != (U,):
            raise ValueError(f"{name} must be 1D of length {U}, got shape {tuple(t.shape)}.")
        if t is not None and t.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be int32 or int64, got {t.dtype}.")
        if t is not None and t.device != x.device:
            raise ValueError(f"{name} must be on x's device, {x.device}, got {t.device}.")
    if multiplicity is not None and (multiplicity < 1).any():
        raise ValueError("multiplicity must be at least 1.")
    M = U if multiplicity is None else int(multiplicity.sum())
    if r.shape != (M,):
        raise ValueError(f"r must be 1D of length {M}, one per term, got shape {tuple(r.shape)}.")
    if g.dim() == 1 and (x.dim() == 1 or g.size(0) != x.size(0)):
        raise ValueError(
            f"A 1D g needs a 2D x of its batch size, got g: {tuple(g.shape)}, x: {tuple(x.shape)}"
        )
    batched = x.dim() == 2
    x = x if batched else x.unsqueeze(0)
    B, N = x.shape
    shared = batch_index is None
    zshape = (B, M) if shared and batched else (M,)
    if zi is not None and zi.shape != zshape:
        raise ValueError(f"zi must have shape {zshape}, got {tuple(zi.shape)}.")
    if not shared and ((batch_index < 0) | (batch_index >= B)).any():
        raise ValueError(f"batch_index must be from 0 to B - 1 = {B - 1}.")
    _check_shared(precision=True, r=r, p=p, g=g, zi=zi, x=x)
    if not (x.is_floating_point() or x.is_complex()) or x.dtype == torch.bfloat16:
        raise ValueError(
            f"pfefilt takes floating-point or complex inputs other than bfloat16, which "
            f"has no complex dtype, got {x.dtype}."
        )
    real = not g.is_complex()
    if real and x.is_complex():
        raise ValueError(
            "A real g makes a real filter, which takes a real x; filter the real and "
            "imaginary parts of a complex x."
        )
    return_zf = zi is not None
    cdtype = x.dtype.to_complex()
    r, p = r.to(cdtype), p.to(cdtype)
    if shared:
        # every row runs every term
        batch_index = torch.arange(B, device=x.device).repeat_interleave(U)
        r, p = r.repeat(B), p.repeat(B)
        if multiplicity is not None:
            multiplicity = multiplicity.repeat(B)
    zi = r.new_zeros(r.shape) if zi is None else zi.to(cdtype).flatten()
    if multiplicity is None:
        # every pole is a chain of one term
        first, goes_on = torch.ones(r.numel(), dtype=torch.bool, device=x.device), None
    else:
        # the first term of each pole, and whether the next entry continues the chain
        pole = torch.arange(p.numel(), device=x.device)
        pole = pole.repeat_interleave(multiplicity, output_size=r.numel())
        p, batch_index = p[pole], batch_index[pole]
        first = torch.ones_like(pole, dtype=torch.bool)
        first[1:] = pole[1:] != pole[:-1]
        goes_on = torch.zeros_like(first)
        goes_on[:-1] = pole[1:] == pole[:-1]
    if real:
        real_pole = p.imag == 0
        if (real_pole & ((r.imag != 0) | (zi.imag != 0))).any():
            raise ValueError(
                "A real g makes a real filter, whose real poles need real residues "
                "and initial states."
            )

    # Each round runs one power of the chains of one kind, from power 1, adds
    # their terms to the output, and passes on the chains that go on. A real
    # filter runs its real poles in the real dtype and each listed non-real
    # pole for its conjugate pair; a complex one runs every term in complex.
    y = g.unsqueeze(-1) * x
    zf = zi.clone() if return_zf else None
    for in_real in (True, False) if real else (False,):
        (i,) = (first & (real_pole == in_real) if real else first).nonzero(as_tuple=True)
        b = batch_index[i]
        u = (x if in_real else x.to(cdtype))[b]
        while N and i.numel():
            a, init, c = p[i], zi[i], r[i]
            if in_real:
                a, init, c = a.real, init.real, c.real
            elif real:
                c = 2 * c  # the pole and its conjugate: 2 Re(c s)
            h = linear_recurrence(a, init, u)
            # the output and the next power take the states before each input
            before = h[:, :-1]
            term, term0 = c.unsqueeze(-1) * before, c * init
            if real:
                term, term0 = term.real, term0.real
            y[:, 0].index_add_(0, b, term0)
            y[:, 1:].index_add_(0, b, term)
            if return_zf:
                zf[i] = h[:, -1].to(cdtype)
            if goes_on is None:
                break
            (k,) = goes_on[i].nonzero(as_tuple=True)
            u = torch.cat((init[k].unsqueeze(-1), before[k]), -1)
            i, b = i[k] + 1, b[k]
    y = y if batched else y.squeeze(0)
    return (y, zf.reshape(zshape)) if return_zf else y


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


def _ssm_lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Tensor | None = None,
    form: str = "df2",
    **kwargs,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run :func:`lfilter` with the ``"ssm"`` backend."""
    b, a = _ssm.pad_coefficients(b, a)

    if zi is not None and form in ("df1", "tdf1"):
        raise ValueError(f"form={form!r} does not take zi; use 'df2' or 'tdf2'.")

    if a.size(-1) == 0 and form in ("df2", "tdf2", "df1", "tdf1"):
        # A pure gain has no state: the output is b0 * x, and the final state
        # is empty. The state-space functions of the general path would have
        # checked the keyword arguments and unroll_factor.
        _check_kwargs(kwargs, _SSM_KWARGS)
        _check_unroll_factor(kwargs)
        if zi is not None and zi.dim() not in (1, 2):
            raise ValueError(f"Initial conditions zi must be 1D or 2D, got {zi.shape}")
        if zi is not None and zi.size(-1) != 0:
            raise ValueError(f"zi must have 0 states for a pure gain, got {zi.size(-1)}.")
        y = b[..., :1] * x
        return y if zi is None else (y, zi.expand(y.size(0), 0))

    A = companion(a)

    match form:
        case "df2" | "tdf2":
            filt = _ssm.form2_state_space(state_space, form, A, b, a, zi, **kwargs)
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
    if not (x.is_floating_point() or x.is_complex()):
        raise ValueError(
            f"backend='diag_ssm' takes floating-point or complex inputs, got {x.dtype}; "
            "use backend='ssm'."
        )
    # The eigendecomposition is complex for complex poles, but a real filter
    # still has a real output and state.
    real = not x.is_complex()

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
        # would have checked the keyword arguments and unroll_factor.
        _check_kwargs(kwargs, _DIAG_SSM_KWARGS)
        _check_unroll_factor(kwargs)
        if b.size(-1) == 0:
            # fir needs a tap: no coefficient means a zero filter, as with
            # backend="ssm", which pads b to len(a) + 1.
            b = F.pad(b, (0, 1))
        if zi is not None and zi.dim() == 1:
            zi = zi.unsqueeze(0).expand(x.size(0), -1)
        return fir(b.broadcast_to((x.size(0), -1)), x, zi)

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

    b, a, terms, D = _ssm.direct_form(b, a)
    # The companion matrix's eigendecomposition, or the poles L with its
    # eigenvectors for them, the Vandermonde matrix, unless V or Vinv is given.
    L = kwargs.get("L")
    if L is None:
        A = companion(a)
        kwargs["A"] = A if form == "df2" else A.mT
    elif kwargs.get("V") is None and kwargs.get("Vinv") is None:
        V = vandermonde(L)
        kwargs |= {"V": V} if form == "df2" else {"Vinv": V.mT}

    match form:
        case "df2":
            filt = partial(diag_state_space, B=None, C=terms, D=D, zi=zi, **kwargs)
        case "tdf2":
            filt = partial(diag_state_space, B=terms, C=None, D=D, zi=zi, out_idx=0, **kwargs)

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
            :attr:`x` is not longer than the padding, :attr:`form` is not
            ``"df2"`` or ``"tdf2"``, or :attr:`b` or :attr:`a` doesn't share
            a dtype and a device with :attr:`x`.
        NotImplementedError: if :attr:`method` is ``"gust"``.

    Note:
        SciPy's default ``padtype="odd"`` has no equivalent here.

    Example::

        >>> from philtorch.lti import filtfilt
        >>> # (0.25 + 0.5 z^-1 + 0.25 z^-2) / (1 - 0.2 z^-1)
        >>> b, a = torch.tensor([0.25, 0.5, 0.25]), torch.tensor([-0.2])
        >>> x = torch.zeros(21)
        >>> x[10] = 1.0
        >>> # Zero phase: a symmetric input gives a symmetric output.
        >>> filtfilt(b, a, x)[8:13]
        tensor([0.1350, 0.3625, 0.5000, 0.3625, 0.1350])
    """
    if method not in ("pad", "gust"):
        raise ValueError("Method must be either 'pad' or 'gust'.")

    if method == "gust":
        raise NotImplementedError("Gustafsson's method is not implemented yet.")
    if form not in ("df2", "tdf2"):
        raise ValueError(f"filtfilt needs form 'df2' or 'tdf2', got {form!r}.")
    _check_shared(b=b, a=a, x=x)

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

    if edge > 0:
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
