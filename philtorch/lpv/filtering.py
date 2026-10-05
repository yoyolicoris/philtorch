"""IIR and FIR filtering with time-varying coefficients."""

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from .._torchlpc import lpc
from ..mat import companion
from ..utils import chain_functions
from .ssm import state_space, state_space_recursion
from .utils import diag_shift


def fir(
    b: Tensor, x: Tensor, zi: Tensor | None = None, transpose: bool = True
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Filter signals with time-varying FIR filters.

    The direct form, ``transpose=False``, weights each input with the
    coefficients of the output's step:

    .. math::
        y[n] = \sum_{k=0}^{M} b_k[n]\, x[n - k].

    The transposed direct form weights each input with the coefficients of its
    own step:

    .. math::
        y[n] = \sum_{k=0}^{M} b_k[n - k]\, x[n - k].

    The two agree for constant coefficients.

    Args:
        b (Tensor): coefficients :math:`b_k[n]`, of shape :math:`(B, N, M + 1)`.
        x (Tensor): input signals, of shape :math:`(B, N)`.
        zi (Tensor, optional): initial state, of shape :math:`(B, M)`. With
            ``transpose=True``, the transposed direct-form state; with
            ``transpose=False``, the past inputs newest first,
            :math:`x[-1], \dots, x[-M]`. Default: ``None``, all zero.
        transpose (bool, optional): use the transposed direct form if
            ``True``, or the direct form if ``False``. Default: ``True``.

    Returns:
        Tensor or tuple of Tensor: the filtered signals, of shape
        :math:`(B, N)`, and with :attr:`zi`, the final state in the same
        convention, of shape :math:`(B, M)`.

    Raises:
        AssertionError: if the shapes do not match.

    Example::

        >>> from philtorch.lpv import fir
        >>> b = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
        >>> x = torch.ones(1, 3)
        >>> fir(b, x, transpose=False)  # y[n] = b_0[n] x[n] + b_1[n] x[n - 1]
        tensor([[1., 4., 6.]])
        >>> fir(b, x)  # y[n] = b_0[n] x[n] + b_1[n - 1] x[n - 1]
        tensor([[1., 3., 5.]])
    """
    assert b.dim() == 3, "Numerator coefficients b must be 3D."
    assert x.dim() == 2, "Input signal x must be 2D."

    B, T = x.shape
    assert b.shape[:2] == x.shape, "The first two dimensions of b must match the shape of x."

    if zi is None:
        return_zf = False
        zi = x.new_zeros((B, b.size(2) - 1))
    else:
        assert zi.dim() == 2, "Initial conditions zi must be 2D."
        assert zi.size(0) == B, "The first dimension of zi must match the batch size."
        assert zi.size(1) == b.size(2) - 1, (
            "The second dimension of zi must match the filter order."
        )

        return_zf = True

    if b.size(2) == 1:
        # A one-tap filter is a gain and keeps no state, so its final state
        # is empty, (B, 0), as in lti.fir. It is a tensor rather than None so
        # callers can chain it and compare its size with other states.
        y = b[..., 0] * x
        return (y, zi) if return_zf else y

    if transpose:
        shifted_b = diag_shift(b, discard_end=not return_zf)
        # vecdot conjugates its first argument, so conjugate b to cancel it.
        y = torch.linalg.vecdot(
            shifted_b.flip(2).conj(),
            F.pad(
                x,
                (shifted_b.size(2) - 1, 0 if not return_zf else shifted_b.size(2) - 1),
            ).unfold(1, shifted_b.size(2), 1),
        )
        if return_zf:
            # y holds T + M samples. zi adds to the first M, which reach into
            # the final state when T < M.
            y = y + F.pad(zi, (0, T))
            return tuple(torch.split_with_sizes(y, [T, b.size(2) - 1], 1))
        return y

    unfolded_x = torch.cat([zi.flip(1), x], dim=1).unfold(1, b.size(2), 1)
    y = torch.linalg.vecdot(unfolded_x.conj(), b.flip(2))

    if return_zf:
        return y, unfolded_x[:, -1, 1:].flip(1)
    return y


def allpole(
    a: Tensor, x: Tensor, zi: Tensor | None = None, transpose: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Filter signals with time-varying all-pole filters.

    The direct form, the default, uses the coefficients of the output's step:

    .. math::
        y[n] = x[n] - \sum_{k=1}^{M} a_k[n]\, y[n - k].

    The transposed direct form, ``transpose=True``, uses the coefficients of
    the step each past output was produced at:

    .. math::
        y[n] = x[n] - \sum_{k=1}^{M} a_k[n - k]\, y[n - k].

    The two agree for constant coefficients. Both run the vendored torchlpc
    kernel.

    Args:
        a (Tensor): coefficients :math:`a_k[n]`, of shape :math:`(B, N, M)`.
        x (Tensor): input signals, of shape :math:`(B, N)`.
        zi (Tensor, optional): initial state, of shape :math:`(B, M)`. With
            ``transpose=False``, the past outputs newest first,
            :math:`y[-1], \dots, y[-M]`; with ``transpose=True``, the
            transposed direct-form state. Default: ``None``, all zero.
        transpose (bool, optional): use the transposed direct form if
            ``True``, or the direct form if ``False``. Default: ``False``.

    Returns:
        Tensor or tuple of Tensor: the filtered signals, of shape
        :math:`(B, N)`, and with :attr:`zi`, the final state in the same
        convention, of shape :math:`(B, M)`.

    Raises:
        AssertionError: if the shapes do not match.

    Example::

        >>> from philtorch.lpv import allpole
        >>> a = torch.full((1, 4, 1), -0.5)  # y[n] = x[n] + 0.5 y[n - 1]
        >>> allpole(a, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        tensor([[1.0000, 0.5000, 0.2500, 0.1250]])
    """
    assert a.dim() == 3, "Denominator coefficients a must be 3D."
    assert x.dim() == 2, "Input signal x must be 2D."
    B, T = x.shape
    assert a.shape[:2] == x.shape, "The first two dimensions of a must match the shape of x."

    if zi is None:
        return_zf = False
        zi = x.new_zeros((B, a.size(2)))
    else:
        assert zi.dim() == 2, "Initial conditions zi must be 2D."
        assert zi.size(0) == B, "The first dimension of zi must match the batch size."
        assert zi.size(1) == a.size(2), (
            f"The second dimension of zi must match the filter order, "
            f"but got {zi.size(1)} instead of {a.size(2)}"
        )

        return_zf = True

    if transpose:
        a = diag_shift(a, offset=1, discard_end=not return_zf)
        if return_zf:
            # Run M steps past the signal to read out the final state. zi adds
            # to the first M samples, which reach past the signal when T < M.
            x = torch.cat([x, torch.zeros_like(zi)], dim=1) + F.pad(zi, (0, T))
        y = lpc(x, a, a.new_zeros(a.size(0), a.size(2)))
        if return_zf:
            return torch.split_with_sizes(y, [T, a.size(2)], 1)
        return y

    y = lpc(x, a, zi)
    if return_zf:
        return y, y[:, -a.size(2) :].flip(1)
    return y  # type: ignore[return-value]


def lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Tensor | None = None,
    form: str | None = None,
    backend: str = "ssm",
    **kwargs: dict | None,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Filter signals with time-varying IIR filters.

    The coefficients carry a time dimension aligned with :attr:`x`, so the
    filter can change at every sample. With constant coefficients, every form
    computes what :func:`philtorch.lti.lfilter` does. With time-varying ones,
    the forms are different filters:

    - ``"df1"``, direct form I:
      :math:`v[n] = \sum_{k=0}^{M_b} b_k[n]\, x[n - k]` and
      :math:`y[n] = v[n] - \sum_{k=1}^{M_a} a_k[n]\, y[n - k]`.
    - ``"df2"``, direct form II:
      :math:`w[n] = x[n] - \sum_{k=1}^{M_a} a_k[n]\, w[n - k]` and
      :math:`y[n] = \sum_{k=0}^{M_b} b_k[n]\, w[n - k]`.
    - ``"tdf2"``, transposed direct form II, with
      :math:`M = \max(M_a, M_b)` states and :math:`s_{M+1} = 0`:

      .. math::
          y[n] &= b_0[n]\, x[n] + s_1[n], \\
          s_k[n + 1] &= s_{k+1}[n] + b_k[n]\, x[n] - a_k[n]\, y[n].

    - ``"tdf1"``, transposed direct form I: :func:`allpole` then
      :func:`fir`, both with ``transpose=True``.

    The transposed forms update each state with the coefficients at that
    step.

    Args:
        b (Tensor): numerator coefficients :math:`b_k[n]`, of shape
            :math:`(B, N, M_b + 1)`, or :math:`(N, M_b + 1)` to share them.
        a (Tensor): denominator coefficients :math:`a_k[n]`, without the
            leading 1, of shape :math:`(B, N, M_a)` or :math:`(N, M_a)`.
        x (Tensor): input signals, of shape :math:`(B, N)` or :math:`(N)`.
        zi (Tensor, optional): initial state, of shape :math:`(B, M)` or
            :math:`(M)`, where :math:`M = \max(M_a, M_b)`. For ``"tdf2"``, the
            states :math:`s_k[0]`; for ``"df2"``, the past values of :math:`w`
            newest first, :math:`w[-1], \dots, w[-M]`. ``"df1"`` and
            ``"tdf1"`` do not take it. Default: ``None``, all zero.
        form (str or None, optional): ``"df2"``, ``"tdf2"``, ``"df1"`` or
            ``"tdf1"``. Default: ``None``, which uses ``"tdf2"`` with the
            ``"ssm"`` backend and ``"df2"`` with ``"torchlpc"``.
        backend (str, optional): ``"ssm"`` runs the filter as a state-space
            model of the :func:`~philtorch.mat.companion` matrices (see
            :func:`state_space`); ``"torchlpc"`` runs :func:`allpole` and
            :func:`fir`, and has no ``"tdf2"``. Default: ``"ssm"``.
        **kwargs: ``unroll_factor`` for the ``"ssm"`` backend (see
            :func:`state_space`); ignored by ``"torchlpc"``.

    Returns:
        Tensor or tuple of Tensor: the filtered signals, of the shape of
        :attr:`x`, and with :attr:`zi`, the final state, of the shape of
        :attr:`zi`.

    Raises:
        ValueError: if :attr:`x` has more than 2 dimensions, :attr:`form` or
            :attr:`backend` is unknown, or :attr:`zi` is given with ``"df1"``
            or ``"tdf1"``.
        NotImplementedError: for ``form="tdf2"`` with ``backend="torchlpc"``.
        AssertionError: if the shapes do not match.

    Note:
        Unlike :func:`philtorch.lti.lfilter`, :attr:`b` and :attr:`a` always
        have a time dimension. As there, :attr:`a` omits the leading
        :math:`a_0 = 1`.

    Example::

        >>> from philtorch.lpv import lfilter
        >>> # Constant coefficients: y[n] = 0.5 x[n] + 0.5 y[n - 1]
        >>> b = torch.tensor([0.5]).expand(4, 1)
        >>> a = torch.tensor([-0.5]).expand(4, 1)
        >>> lfilter(b, a, torch.tensor([1.0, 0.0, 0.0, 0.0]))
        tensor([0.5000, 0.2500, 0.1250, 0.0625])
    """

    squeeze_first = (
        (x.dim() == 1) & (b.dim() == 2) & (a.dim() == 2) & ((zi is None) or (zi.dim() == 1))
    )

    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        raise ValueError("Input signal x must be 1D or 2D.")

    assert b.dim() in (2, 3), "Numerator coefficients b must be 2D or 3D."
    assert a.dim() in (2, 3), "Denominator coefficients a must be 2D or 3D."

    if form is None:
        form = "df2" if backend == "torchlpc" else "tdf2"
    if zi is not None and form in ("df1", "tdf1"):
        raise ValueError(f"form={form!r} does not take zi; use 'df2' or 'tdf2'.")

    match backend:
        case "ssm":
            y = _ssm_lfilter(b, a, x, zi=zi, form=form, **kwargs)
        case "torchlpc":
            y = _torchlpc_lfilter(b, a, x, zi=zi, form=form)
        case _:
            raise ValueError(
                f"Unknown backend: {backend}. Supported backends are 'ssm', 'torchlpc'."
            )

    if isinstance(y, tuple):
        y, zf = y
        if squeeze_first:
            y = y.squeeze(0)
            zf = zf.squeeze(0)
        return y, zf

    if squeeze_first:
        y = y.squeeze(0)
    return y


def _torchlpc_lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Tensor | None = None,
    form: str = "df2",
) -> Tensor | tuple[Tensor, Tensor]:
    """Run :func:`lfilter` with the ``"torchlpc"`` backend."""
    _, T = x.shape

    if b.dim() == 2:
        b = b.unsqueeze(0)
    elif b.dim() == 3:
        pass
    else:
        raise ValueError("Numerator coefficients b must be 2D or 3D.")
    if a.dim() == 2:
        a = a.unsqueeze(0)
    elif a.dim() == 3:
        pass
    else:
        raise ValueError("Denominator coefficients a must be 2D or 3D.")

    assert b.shape[1] == a.shape[1] == T, (
        "The number of time steps in b and a must match the input signal x."
    )

    order = max(a.shape[2], b.shape[2] - 1)

    B = max(b.size(0), a.size(0), x.size(0))

    return_zf = (zi is not None) and (form in ("df2", "tdf2"))
    if zi is None:
        zi = x.new_zeros((B, order))
    elif zi.dim() == 1:
        assert zi.shape[0] == order, "Initial conditions zi must match filter order."
        zi = zi.unsqueeze(0).expand(B, -1)
    elif zi.dim() == 2:
        assert zi.shape[1] == order, "Initial conditions zi must match filter order."
        B = max(B, zi.size(0))
        zi = zi.expand(B, -1)
    else:
        raise ValueError("Initial conditions zi must be 1D or 2D.")

    broadcasted_b = b.expand(B, -1, -1)
    broadcasted_a = a.expand(B, -1, -1)
    broadcasted_x = x.expand(B, -1)

    match form:
        case "df2":
            filt = chain_functions(
                partial(allpole, broadcasted_a, zi=zi[:, : a.shape[2]]),
                lambda x, a_zf: (
                    fir(
                        broadcasted_b,
                        x,
                        zi=zi[:, : b.shape[2] - 1],
                        transpose=False,
                    )
                    + (a_zf,)
                ),
                lambda x, b_zf, a_zf: (
                    (
                        x,
                        b_zf if b_zf.size(1) > a_zf.size(1) else a_zf,
                    )
                    if return_zf
                    else x
                ),
            )
        case "tdf2":
            raise NotImplementedError("Transposed Direct Form II (tdf2) is not implemented yet.")
        case "df1":
            # lfilter rejects zi for direct form I.
            filt = chain_functions(
                partial(fir, broadcasted_b, transpose=False),
                partial(allpole, broadcasted_a),
            )
        case "tdf1":
            # lfilter rejects zi for transposed direct form I.
            filt = chain_functions(
                partial(
                    allpole,
                    broadcasted_a,
                    transpose=True,
                ),
                partial(fir, broadcasted_b, transpose=True),
            )
        case _:
            raise ValueError(
                f"Unknown filter form: {form}. Supported forms are 'df2', 'tdf2', 'df1', 'tdf1'."
            )

    return filt(broadcasted_x)


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
                partial(
                    fir,
                    b.broadcast_to((x.size(0), -1, -1)),
                    transpose=False,
                ),
                partial(
                    state_space_recursion,
                    A,
                    zi,
                    out_idx=0,
                    **kwargs,
                ),
            )
        case "tdf1":
            zi = x.new_zeros((x.size(0), A.size(-1)))
            # state_space_recursion(A, ...) returns h[1..N] of
            # h[n + 1] = A[n] h[n] + x[n], so its output at step n uses A[n].
            # The textbook update, as in tdf2, uses A[n] to go from step n to
            # n + 1, so the output at step n must use A[n - 1]: pass A one
            # step late. A[0] is never used, as zi is zero.
            A_late = torch.cat([A[..., :1, :, :], A[..., :-1, :, :]], dim=-3)
            filt = chain_functions(
                partial(
                    state_space_recursion,
                    A_late.mT,
                    zi,
                    out_idx=0,
                    **kwargs,
                ),
                partial(fir, b.broadcast_to((x.size(0), -1, -1)), transpose=True),
            )
        case _:
            raise ValueError(f"Unknown filter form: {form}")

    return filt(x)
