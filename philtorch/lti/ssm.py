"""State-space models with time-invariant coefficients."""

from functools import partial

import torch
from torch import Tensor

from .. import _ssm
from ..mat import find_eigenvectors
from ..utils import _check_shared
from .recur import linear_recurrence


def state_space_recursion(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    *,
    unroll_factor: int | None = None,
    out_idx: int | None = None,
) -> Tensor:
    r"""Compute the states of a linear time-invariant recurrence.

    This computes

    .. math::
        \mathbf{h}[n + 1] = A \mathbf{h}[n] + \mathbf{x}[n],
        \quad n = 0, \dots, N - 1,

    starting from :math:`\mathbf{h}[0] = \mathbf{z}_i`, and returns
    :math:`\mathbf{h}[1], \dots, \mathbf{h}[N]`. A 2-D input feeds the
    first state only: :math:`\mathbf{x}[n] = x[n] \mathbf{e}_1`.

    Args:
        A (Tensor): state matrices, of shape :math:`(M, M)` or
            :math:`(B, M, M)`.
        zi (Tensor): the initial state :math:`\mathbf{h}[0]`, of shape
            :math:`(B, M)`.
        x (Tensor): inputs, of shape :math:`(B, N, M)` or :math:`(B, N)`.
        unroll_factor (int, optional): ``None`` runs a native kernel, which
            doesn't take every input (see Raises). An integer runs PyTorch ops
            on any device, in the dtypes their matrix products take there:
            integers on CPU, but not on CUDA. ``1`` runs the sequential loop,
            as does :math:`N` or more, and a value from 2 to :math:`N - 1` is
            the block length of a block-unrolled recursion. Default: ``None``.
        out_idx (int, optional): return only this state. Default: ``None``.

    Returns:
        Tensor: the states :math:`\mathbf{h}[1], \dots, \mathbf{h}[N]`, of
        shape :math:`(B, N, M)`, or :math:`(B, N)` with :attr:`out_idx`.

    Raises:
        ValueError: if :attr:`A`, :attr:`zi` and :attr:`x` don't share a dtype
            and a device, if the shapes do not match, or :attr:`unroll_factor`
            is 0 or negative.
        NotImplementedError: with ``unroll_factor=None``, if no native kernel
            takes the input, such as :math:`M > 128` (64 complex) on CUDA and
            MPS.

    Example::

        >>> from philtorch.lti import state_space_recursion
        >>> # Fibonacci numbers: the first state of A = [[1, 1], [1, 0]] for
        >>> # an impulse x[n] and h[0] = 0.
        >>> A = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        >>> state_space_recursion(A, torch.zeros(1, 2), x, out_idx=0)
        tensor([[1., 1., 2., 3., 5.]])
    """
    if x.dim() not in (2, 3):
        raise ValueError(f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}")
    _check_shared(A=A, zi=zi, x=x)
    _check_A(A, x)
    if x.dim() == 3 and A.size(-1) != x.size(-1):
        raise ValueError(
            f"Last dimension of A must match last dimension of x, "
            f"got A: {A.size(-1)}, x: {x.size(-1)}"
        )

    batch_size = x.size(0)
    M = A.size(-1)

    if zi.dim() != 2:
        raise ValueError(f"Initial conditions zi must be 2D, got {zi.shape}")
    if zi.size(0) != batch_size:
        raise ValueError(
            f"Batch size of zi must match batch size of x, got zi: {zi.size(0)}, x: {batch_size}"
        )
    if zi.size(1) != M:
        raise ValueError(
            f"Last dimension of zi must match last dimension of A, got zi: {zi.size(1)}, A: {M}"
        )

    return _ssm.recursion(_ssm.from_lti(A), zi, x, unroll_factor=unroll_factor, out_idx=out_idx)


def state_space(
    A: Tensor,
    x: Tensor,
    B: Tensor | None = None,
    C: Tensor | None = None,
    D: Tensor | None = None,
    zi: Tensor | None = None,
    unroll_factor: int | None = None,
    out_idx: int | None = None,
):
    r"""Compute the outputs of a linear time-invariant state-space model.

    This computes

    .. math::
        \mathbf{h}[n + 1] &= A \mathbf{h}[n] + B \mathbf{x}[n], \\
        \mathbf{y}[n] &= C \mathbf{h}[n] + D \mathbf{x}[n],

    starting from :math:`\mathbf{h}[0] = \mathbf{z}_i`. The recursion runs as
    in :func:`state_space_recursion`.

    Args:
        A (Tensor): state matrices, of shape :math:`(M, M)` or
            :math:`(B, M, M)`.
        x (Tensor): inputs, of shape :math:`(B, N, F)`, or :math:`(B, N)` for
            a scalar input.
        B (Tensor, optional): the input matrix, of shape :math:`(M)` or
            :math:`(B, M)` for a 2-D :attr:`x`, and :math:`(M, F)` or
            :math:`(B, M, F)` for a 3-D one. Default: ``None``: a 2-D
            :attr:`x` feeds the first state, and a 3-D one needs
            :math:`F = M`.
        C (Tensor, optional): the output matrix, of shape :math:`(M)` or
            :math:`(B, M)` for a scalar output, or :math:`(P, M)` or
            :math:`(B, P, M)` for :math:`P` outputs. Default: ``None``, which
            outputs the states.
        D (Tensor, optional): the feedthrough matrix. For a 2-D :attr:`x`, of
            shape :math:`()`, :math:`(1)`, :math:`(B)`, :math:`(P)` or
            :math:`(B, P)`; for a 3-D one, :math:`()`, :math:`(1)`,
            :math:`(F)`, :math:`(B, F)`, :math:`(P, F)` or :math:`(B, P, F)`.
            When two readings fit, such as :math:`P = B`, the batched one wins.
            Default: ``None``, no feedthrough.
        zi (Tensor, optional): the initial state, of shape :math:`(B, M)` or
            :math:`(M)`. Default: ``None``, all zero.
        unroll_factor (int, optional): see :func:`state_space_recursion`.
            Default: ``None``.
        out_idx (int, optional): output only this state, instead of using
            :attr:`C`. Default: ``None``.

    Returns:
        Tensor or tuple of Tensor: the outputs, of shape :math:`(B, N)`,
        :math:`(B, N, P)`, or :math:`(B, N, M)` without :attr:`C`, and with
        :attr:`zi`, the final state :math:`\mathbf{h}[N]`, of shape
        :math:`(B, M)`.

    Raises:
        ValueError: if both :attr:`C` and :attr:`out_idx` are given, if
            :attr:`A`, :attr:`B`, :attr:`C`, :attr:`D`, :attr:`zi` and
            :attr:`x` don't share a dtype and a device, if
            :attr:`B`, :attr:`C` or :attr:`D` has an unsupported shape, if
            :attr:`D` gives a different number of outputs from :attr:`C`,
            :attr:`out_idx` or the states, or if the other shapes do not match.

    Note:
        This computes what :func:`scipy.signal.dlsim` does with ``x0=zi``,
        without the time vector or the state sequence.

    Example::

        >>> from philtorch.lti import state_space
        >>> # Fibonacci numbers: y[n] = C h[n] with A = [[1, 1], [1, 0]],
        >>> # C = [1, 0] and an impulse x[n].
        >>> A = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        >>> state_space(A, x, C=torch.tensor([1.0, 0.0]))
        tensor([[0., 1., 1., 2., 3., 5., 8.]])
    """
    _ssm.check_arguments(x, C, out_idx)
    _check_shared(A=A, B=B, C=C, D=D, zi=zi, x=x)
    # A first, so that a fault in A isn't reported as one in B, sized by A.
    _check_A(A, x)

    batch_size = x.size(0)
    M = A.size(-1)
    zi, return_zf = _ssm.initial_state(x, zi, M)

    Bx = _ssm_B(B, x, batch_size, M)

    h, zf = _ssm.states(
        partial(state_space_recursion, A, zi, Bx, unroll_factor=unroll_factor),
        zi,
        out_idx=out_idx,
        final_state=return_zf,
    )

    y = _ssm_C_D(h, x, C, D, batch_size, M)

    if return_zf:
        return y, zf
    return y


def _check_A(A: Tensor, x: Tensor) -> None:
    """The checks of A, (M, M) or (B, M, M), against x."""
    if A.dim() not in (2, 3):
        raise ValueError(f"State matrix A must be 2D or 3D, got {A.shape}")
    if A.size(-2) != A.size(-1):
        raise ValueError(f"State matrix A must be square, got {A.shape}")
    if A.dim() == 3 and x.size(0) != A.size(0):
        raise ValueError(
            f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {x.size(0)}"
        )


def _ssm_B(B, x, batch_size, M):
    r"""Return :math:`B \mathbf{x}[n]` for every step.

    This takes the shapes :func:`state_space` accepts, and returns :attr:`x`
    itself without :attr:`B`.
    """
    if B is None:
        return x

    # Matching on x.dim() first makes every case unambiguous, e.g. an (M, F)
    # matrix is never mistaken for a batched (B, M) vector when B == M.
    features = x.size(-1)
    match x.dim(), tuple(B.shape):
        case 2, (m,) if m == M:
            return x.unsqueeze(-1) * B
        case 2, (b, m) if (b, m) == (batch_size, M):
            return x.unsqueeze(-1) * B.unsqueeze(1)
        case 3, (m, f) if (m, f) == (M, features):
            return x @ B.T
        case 3, (b, m, f) if (b, m, f) == (batch_size, M, features):
            return x @ B.mT
    raise _ssm_B_shape_error(B, x, batch_size, M)


def _ssm_B_shape_error(B, x, batch_size, M):
    if x.dim() == 2:
        allowed = f"({M},) or ({batch_size}, {M})"
    else:
        features = x.size(-1)
        allowed = f"({M}, {features}) or ({batch_size}, {M}, {features})"
    return ValueError(
        f"Input matrix B must be of shape {allowed} for {x.dim()}D input x, got {tuple(B.shape)}"
    )


def _ssm_D(D, x, batch_size):
    r"""Return :math:`D \mathbf{x}[n]` for every step.

    This takes the shapes :func:`state_space` accepts.
    """
    features = x.size(-1)
    match x.dim(), tuple(D.shape):
        case 2, (b,) if b == batch_size:
            return D.unsqueeze(1) * x
        case 2, () | (1,):
            return x * D
        case 2, (_,):
            return x.unsqueeze(-1) * D
        case 2, (b, _) if b == batch_size:
            return x.unsqueeze(-1) * D.unsqueeze(1)
        case 3, (f,) if f == features:
            return x @ D
        case 3, () | (1,):
            return x * D
        case 3, (b, f) if (b, f) == (batch_size, features):
            return (x @ D.unsqueeze(-1)).squeeze(-1)
        case 3, (_, f) if f == features:
            return x @ D.T
        case 3, (b, _, f) if (b, f) == (batch_size, features):
            return x @ D.mT
        case 2, _:
            allowed = f"(), (1,), ({batch_size},), (P,), or ({batch_size}, P)"
        case _:
            allowed = (
                f"(), (1,), ({features},), ({batch_size}, {features}), "
                f"(P, {features}), or ({batch_size}, P, {features})"
            )
    raise ValueError(
        f"Input matrix D must be of shape {allowed} for {x.dim()}D input x, got {tuple(D.shape)}"
    )


def _ssm_C_D(h, x, C, D, batch_size, M):
    r"""Return :math:`C \mathbf{h}[n] + D \mathbf{x}[n]` for every step.

    This takes the shapes :func:`state_space` accepts. Where a batched and an
    unbatched shape coincide, such as :math:`P = B`, the batched reading
    wins.
    """
    Dx = None if D is None else _ssm_D(D, x, batch_size)

    if C is not None:
        match tuple(C.shape):
            case (m,) if m == M:
                Ch = h @ C
            case (b, m) if (b, m) == (batch_size, M):
                Ch = (h @ C.unsqueeze(-1)).squeeze(-1)
            case (_, m) if m == M:
                Ch = h @ C.T
            case (b, _, m) if (b, m) == (batch_size, M):
                Ch = h @ C.mT
            case _:
                raise ValueError(
                    f"Output matrix C must be of shape ({M},), ({batch_size}, {M}), "
                    f"(P, {M}), or ({batch_size}, P, {M}), got {tuple(C.shape)}"
                )
    else:
        Ch = h

    return _ssm.outputs(Ch, Dx)


def diag_state_space(
    x: Tensor,
    L: Tensor | None = None,
    V: Tensor | None = None,
    Vinv: Tensor | None = None,
    A: Tensor | None = None,
    B: Tensor | None = None,
    C: Tensor | None = None,
    D: Tensor | None = None,
    zi: Tensor | None = None,
    out_idx: int | None = None,
    unroll_factor: int | None = None,
):
    r"""Compute a state-space model's outputs through its eigendecomposition.

    With :math:`A = V \Lambda V^{-1}` and
    :math:`\Lambda = \operatorname{diag}(\boldsymbol{\lambda})`, the model of
    :func:`state_space` in the eigenvectors' coordinates,
    :math:`\hat{\mathbf{h}} = V^{-1} \mathbf{h}`, is

    .. math::
        \hat{\mathbf{h}}[n + 1] &= \Lambda \hat{\mathbf{h}}[n]
            + \hat{B} \mathbf{x}[n], \\
        \mathbf{y}[n] &= \hat{C} \hat{\mathbf{h}}[n] + D \mathbf{x}[n],

    with :math:`\hat{B} = V^{-1} B` and :math:`\hat{C} = C V`. As
    :math:`\Lambda` is diagonal, the states are :math:`M` independent
    first-order recurrences. The outputs are those of :func:`state_space` for a
    diagonalizable :math:`A`.

    Give the decomposition in one of these ways:

    - :attr:`A` alone: the eigenvalues come from
      :func:`torch.linalg.eigvals` and the eigenvectors from
      :func:`~philtorch.mat.find_eigenvectors`.
    - :attr:`L` with :attr:`A`, :attr:`V` or :attr:`Vinv`: the missing
      matrices are computed from it.
    - :attr:`L` with both :attr:`V` and :attr:`Vinv`: used as given.
    - :attr:`L` alone: :math:`A` is diagonal and :math:`V = I`.

    Args:
        x (Tensor): inputs, of shape :math:`(B, N, F)` or :math:`(B, N)`.
        L (Tensor, optional): the eigenvalues :math:`\boldsymbol{\lambda}`,
            of shape :math:`(M)` or :math:`(B, M)`. Default: ``None``.
        V (Tensor, optional): the eigenvectors as columns, of shape
            :math:`(M, M)` or :math:`(B, M, M)`. Default: ``None``.
        Vinv (Tensor, optional): the inverse of :attr:`V`, of the same shape.
            Default: ``None``.
        A (Tensor, optional): the state matrix, of shape :math:`(M, M)` or
            :math:`(B, M, M)`. Default: ``None``.
        B (Tensor, optional): see :func:`state_space`, except that with
            :attr:`L` alone a 2-D :attr:`x` without :attr:`B` drives every
            state. Default: ``None``.
        C (Tensor, optional): see :func:`state_space`. Default: ``None``.
        D (Tensor, optional): see :func:`state_space`. Default: ``None``.
        zi (Tensor, optional): see :func:`state_space`. Default: ``None``.
        out_idx (int, optional): see :func:`state_space`. Default: ``None``.
        unroll_factor (int, optional): see :func:`linear_recurrence`.
            Default: ``None``.

    Returns:
        Tensor or tuple of Tensor: the outputs, and with :attr:`zi` the final
        state, as in :func:`state_space`, in the dtype of :attr:`x`: real for a
        real :attr:`x`, even with complex eigenvalues.

    Raises:
        ValueError: if neither :attr:`L` nor :attr:`A` is given, if
            :attr:`V` or :attr:`Vinv` is given without :attr:`L`, if both
            :attr:`C` and :attr:`out_idx` are given, if :attr:`A`, :attr:`B`,
            :attr:`C`, :attr:`D` and :attr:`zi` don't have the dtype and
            device of :attr:`x`, if :attr:`L`, :attr:`V` and :attr:`Vinv`
            don't share a dtype and a device, or have neither the dtype of
            :attr:`x` nor, with a real :attr:`A`, its complex dtype, if
            :attr:`D` gives a different number of outputs from :attr:`C`,
            :attr:`out_idx` or the states, or if the shapes do not match.

    Example::

        >>> from philtorch.lti import diag_state_space, state_space
        >>> # The impulse response of A = [[0.5, 0.2], [0, -0.3]] and
        >>> # C = [1, 1], through the eigendecomposition of A and directly.
        >>> A = torch.tensor([[0.5, 0.2], [0.0, -0.3]])
        >>> C = torch.tensor([1.0, 1.0])
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        >>> diag_state_space(x, A=A, C=C)
        tensor([[0.0000, 1.0000, 0.5000, 0.2500, 0.1250, 0.0625]])
        >>> state_space(A, x, C=C)
        tensor([[0.0000, 1.0000, 0.5000, 0.2500, 0.1250, 0.0625]])
    """
    _ssm.check_arguments(x, C, out_idx)
    _check_shared(A=A, B=B, C=C, D=D, zi=zi, x=x)
    _check_shared(L=L, V=V, Vinv=Vinv)
    if L is not None:
        # Without A, the decomposition is the model; a real A's may be complex.
        _check_shared(precision=A is not None and not A.is_complex(), L=L, x=x)
    # A real model has real outputs, even through complex eigenvalues.
    real = not x.is_complex()

    batch_size = x.size(0)

    if L is None:
        if A is None:
            raise ValueError("Either L or A must be provided")
        if V is not None or Vinv is not None:
            raise ValueError("V and Vinv need L")
        _check_A(A, x)
        L = torch.linalg.eigvals(A)
        V = Vinv = None
        M = A.size(-1)
    else:
        match L.shape:
            case (_,):
                M = L.size(0)
                if V is not None:
                    if V.dim() != 2:
                        raise ValueError(f"V must be 2D, got {V.shape}")
                    if not (V.size(0) == V.size(1) == M):
                        raise ValueError(f"V must be square with size {M}, got {V.shape}")
                if Vinv is not None:
                    if Vinv.dim() != 2:
                        raise ValueError(f"Vinv must be 2D, got {Vinv.shape}")
                    if not (Vinv.size(0) == Vinv.size(1) == M):
                        raise ValueError(f"Vinv must be square with size {M}, got {Vinv.shape}")

                if A is not None:
                    if A.dim() != 2:
                        raise ValueError(f"A must be 2D, got {A.shape}")
                    if not (A.size(0) == A.size(1) == M):
                        raise ValueError(f"A must be square with size {M}, got {A.shape}")

            case (L_batch, _) if L_batch == batch_size:
                M = L.size(1)
                if V is not None:
                    if V.dim() != 3:
                        raise ValueError(f"V must be 3D, got {V.shape}")
                    if not (V.size(0) == batch_size and V.size(1) == V.size(2) == M):
                        raise ValueError(
                            f"V must be a batch of square matrices with size {M}, got {V.shape}"
                        )
                if Vinv is not None:
                    if Vinv.dim() != 3:
                        raise ValueError(f"Vinv must be 3D, got {Vinv.shape}")
                    if not (Vinv.size(0) == batch_size and Vinv.size(1) == Vinv.size(2) == M):
                        raise ValueError(
                            f"Vinv must be a batch of square matrices with size {M}, "
                            f"got {Vinv.shape}"
                        )

                if A is not None:
                    if A.dim() != 3:
                        raise ValueError(f"A must be 3D, got {A.shape}")
                    if not (A.size(0) == batch_size and A.size(1) == A.size(2) == M):
                        raise ValueError(
                            f"A must be a batch of square matrices with size {M}, got {A.shape}"
                        )

            case _:
                raise ValueError(f"L must be a vector or a batch of vectors, got {L.shape}")

    if B is None and x.dim() == 3 and x.size(-1) != M:
        raise ValueError(
            f"Last dimension of A must match last dimension of x, got A: {M}, x: {x.size(-1)}"
        )

    match (V, Vinv, A):
        case (None, None, None):
            # A = diag(L) and V = I: V and Vinv stay None, meaning the identity.
            pass
        case (None, None, _):
            V = find_eigenvectors(A, L)
            Vinv = torch.linalg.inv(V)
        case (None, _, _):
            V = torch.linalg.inv(Vinv)
        case (_, None, _):
            Vinv = torch.linalg.inv(V)
        case (_, _, _):
            pass

    zi, return_zf = _ssm.initial_state(x, zi, M)

    # A real model with complex eigenvalues runs in the eigenvalues' dtype.
    x_orig, zi_orig = x, zi
    x, zi, B = (t if t is None else t.to(L.dtype) for t in (x, zi, B))

    # Vinv is None (the identity), or 2-D or 3-D: as given and checked, or as
    # computed.
    if Vinv is None:
        Vinvzi = zi
    elif Vinv.dim() == 2:
        Vinvzi = zi @ Vinv.T
    else:
        Vinvzi = torch.linalg.vecdot(Vinv.conj(), zi.unsqueeze(1))

    if Vinv is None:
        if B is None and x.dim() == 2:  # every state is driven by x
            VinvBx = x.unsqueeze(-1).expand(-1, -1, M)
        else:
            VinvBx = _ssm_B(B, x, batch_size, M)
    elif B is not None:
        # Same dispatch as _ssm_B, but Vinv is folded into B before touching x.
        features = x.size(-1)
        match x.dim(), tuple(B.shape):
            case 2, (m,) if m == M:
                VinvB = Vinv @ B
                if VinvB.dim() == 2:
                    VinvB = VinvB.unsqueeze(1)
                VinvBx = x.unsqueeze(-1) * VinvB
            case 2, (b, m) if (b, m) == (batch_size, M):
                VinvB = (Vinv @ B.unsqueeze(-1)).squeeze(-1)
                VinvBx = x.unsqueeze(-1) * VinvB.unsqueeze(1)
            case 3, (m, f) if (m, f) == (M, features):
                VinvBx = x @ (Vinv @ B).mT
            case 3, (b, m, f) if (b, m, f) == (batch_size, M, features):
                VinvBx = x @ (Vinv @ B).mT
            case _:
                raise _ssm_B_shape_error(B, x, batch_size, M)
    elif x.dim() == 2 and Vinv.dim() == 2:
        VinvBx = x.unsqueeze(-1) * Vinv[:, 0]
    elif x.dim() == 2 and Vinv.dim() == 3:
        VinvBx = x.unsqueeze(-1) * Vinv[:, None, :, 0]
    else:  # x is 3-D, as checked
        VinvBx = x @ Vinv.mT

    recur_runner = partial(linear_recurrence, unroll_factor=unroll_factor)

    Vinvh = recur_runner(
        L.broadcast_to((batch_size, M)).flatten(0, 1),
        Vinvzi.flatten(),
        VinvBx.mT.flatten(0, 1),
    ).unflatten(0, (batch_size, M))

    def run(out_idx, h_hat=Vinvh):
        if V is None:
            h = h_hat.mT if out_idx is None else h_hat[:, out_idx]
        elif out_idx is None:
            h = (V @ h_hat).mT
        else:
            h = (V[..., out_idx, None, :] @ h_hat).squeeze(-2)
        # A real system's states are real: their imaginary part is rounding.
        return h.real if real else h

    x, zi = x_orig, zi_orig
    if C is None:
        h, zf = _ssm.states(run, zi, out_idx=out_idx, final_state=return_zf)
        y = _ssm_C_D(h, x, None, D, batch_size, M)
    else:
        # y[n] = C h[n] = C_hat h_hat[n], with C_hat = C V: the states stay in
        # the eigenbasis, and the shift moves C h, of P values per step,
        # rather than the M states. h[0] = zi is in the original basis.
        N = Vinvh.size(-1)
        zf = (run(None, Vinvh[..., -1:])[:, 0] if N else zi) if return_zf else None
        Czi = _ssm_C_D(zi.unsqueeze(1), x, C, None, batch_size, M)
        C_hat = C.to(L.dtype)
        if V is not None:
            # A (B, M) C is a row per signal, as _ssm_C_D reads it.
            per_signal = V.dim() == 3 and C.dim() == 2 and C.size(0) == batch_size
            C_hat = (C_hat.unsqueeze(-2) @ V).squeeze(-2) if per_signal else C_hat @ V
        Ch = _ssm_C_D(Vinvh.mT[:, :-1], x, C_hat, None, batch_size, M)
        Ch = torch.cat([Czi, Ch.real if real else Ch], dim=1)[:, :N]
        y = _ssm.outputs(Ch, None if D is None else _ssm_D(D, x, batch_size))
    return (y, zf) if return_zf else y
