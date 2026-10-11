"""State-space models with time-varying coefficients."""

from functools import partial

import torch
from torch import Tensor

from .. import _ssm


def state_space_recursion(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    *,
    unroll_factor: int = 1,
    out_idx: int | None = None,
) -> Tensor:
    r"""Compute the states of a linear time-varying recurrence.

    This computes

    .. math::
        \mathbf{h}[n + 1] = A[n] \mathbf{h}[n] + \mathbf{x}[n],
        \quad n = 0, \dots, N - 1,

    starting from :math:`\mathbf{h}[0] = \mathbf{z}_i`, and returns
    :math:`\mathbf{h}[1], \dots, \mathbf{h}[N]`. A 2-D input feeds the
    first state only: :math:`\mathbf{x}[n] = x[n] \mathbf{e}_1`.

    With ``unroll_factor=1``, this runs a kernel chosen by device, dtype and
    :math:`M`: native ones on CPU and CUDA, and a Triton scan on CUDA for the
    rest. Where no kernel takes the input, such as on MPS, it runs a loop
    over time steps. On any device, an :attr:`unroll_factor` greater than 1
    and less than :math:`N` runs a block-unrolled PyTorch recursion instead,
    and one of :math:`N` or more runs a plain PyTorch loop.

    Args:
        A (Tensor): state matrices :math:`A[n]`, of shape :math:`(N, M, M)` to
            share them or :math:`(B, N, M, M)`.
        zi (Tensor): the initial state :math:`\mathbf{h}[0]`, of shape
            :math:`(B, M)`.
        x (Tensor): inputs, of shape :math:`(B, N, M)` or :math:`(B, N)`.
        unroll_factor (int, optional): ``1`` for the kernels described
            above, a value less than :math:`N` for the block length of the
            unrolled recursion, or :math:`N` or more for a plain loop.
            Default: ``1``.
        out_idx (int, optional): return only this state. Default: ``None``.

    Returns:
        Tensor: the states :math:`\mathbf{h}[1], \dots, \mathbf{h}[N]`, of
        shape :math:`(B, N, M)`, or :math:`(B, N)` with :attr:`out_idx`.

    Raises:
        ValueError: if the shapes do not match, or :attr:`unroll_factor` is
            less than 1.

    Note:
        These are the states after each input, :math:`\mathbf{h}[1]` to
        :math:`\mathbf{h}[N]`: the state that :func:`state_space` uses at
        step :math:`n` is :math:`\mathbf{h}[n]`, one step earlier, and its
        final state is :math:`\mathbf{h}[N]`.

    Example::

        >>> from philtorch.lpv import state_space_recursion
        >>> A = torch.tensor([[1.0, 1.0], [1.0, 0.0]]).expand(5, 2, 2)
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        >>> state_space_recursion(A, torch.zeros(1, 2), x, out_idx=0)
        tensor([[1., 1., 2., 3., 5.]])
    """
    if x.dim() not in (2, 3):
        raise ValueError(f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}")
    if A.dim() not in (3, 4):
        raise ValueError(f"State matrix A must be 3D or 4D, got {A.shape}")
    if A.size(-2) != A.size(-1):
        raise ValueError(f"State matrix A must be square, got {A.shape}")
    if A.size(-3) != x.size(1):
        raise ValueError(
            f"State matrix A must have the same time dimension as x, "
            f"got A: {A.size(-3)}, x: {x.size(1)}"
        )
    if A.dim() == 4 and x.size(0) != A.size(0):
        raise ValueError(
            f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {x.size(0)}"
        )
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

    return _ssm.recursion(_ssm.from_lpv(A), zi, x, unroll_factor=unroll_factor, out_idx=out_idx)


def state_space(
    A: Tensor,
    x: Tensor,
    B: Tensor | None = None,
    C: Tensor | None = None,
    D: Tensor | None = None,
    zi: Tensor | None = None,
    unroll_factor: int = 1,
    out_idx: int | None = None,
    # **kwargs,
):
    r"""Compute the outputs of a linear time-varying state-space model.

    This computes

    .. math::
        \mathbf{h}[n + 1] &= A[n] \mathbf{h}[n] + B[n] \mathbf{x}[n], \\
        \mathbf{y}[n] &= C[n] \mathbf{h}[n] + D[n] \mathbf{x}[n],

    starting from :math:`\mathbf{h}[0] = \mathbf{z}_i`. The recursion runs as
    in :func:`state_space_recursion`.

    Each of :attr:`B`, :attr:`C` and :attr:`D` may be constant or time-varying,
    and shared or one per signal: its base shape below can be prefixed with
    :math:`N` for time-varying values, :math:`B` for one per signal, or
    :math:`(B, N)` for both; the one exception is a scalar :attr:`D` with a
    3-D :attr:`x`. When two readings fit, such as :math:`N = B`, the
    time-varying one is tried first, then the per-signal one.

    Args:
        A (Tensor): state matrices :math:`A[n]`, of shape :math:`(N, M, M)` or
            :math:`(B, N, M, M)`.
        x (Tensor): inputs, of shape :math:`(B, N, F)`, or :math:`(B, N)` for
            a scalar input.
        B (Tensor, optional): the input matrices, of base shape :math:`(M)`
            for a 2-D :attr:`x` or :math:`(M, F)` for a 3-D one.
            Default: ``None``: a 2-D :attr:`x` feeds the first state, and a 3-D
            one needs :math:`F = M`.
        C (Tensor, optional): the output matrices, of base shape :math:`(M)`
            for a scalar output or :math:`(P, M)` for :math:`P` outputs.
            Default: ``None``, which outputs the states.
        D (Tensor, optional): the feedthrough matrices, of base shape
            :math:`()` or :math:`(P)` for a 2-D :attr:`x`, where :math:`(1)`
            is also accepted for a shared scalar, and :math:`(F)` or
            :math:`(P, F)` for a 3-D one. For a 3-D :attr:`x`, a scalar
            :math:`()` or :math:`(1)` is only accepted constant and shared,
            as :math:`D` times the identity, so the output size must equal
            :math:`F`; vary the feedthrough over time or per signal with the
            :math:`(F)` or :math:`(P, F)` base shapes. Default: ``None``, no
            feedthrough.
        zi (Tensor, optional): the initial state, of shape :math:`(B, M)` or
            :math:`(M)`. Default: ``None``, all zero.
        unroll_factor (int, optional): see :func:`state_space_recursion`.
            Default: ``1``.
        out_idx (int, optional): output only this state, instead of using
            :attr:`C`. Default: ``None``.

    Returns:
        Tensor or tuple of Tensor: the outputs, of shape :math:`(B, N)`,
        :math:`(B, N, P)`, or :math:`(B, N, M)` without :attr:`C`, and with
        :attr:`zi`, the final state :math:`\mathbf{h}[N]`, of shape
        :math:`(B, M)`.

    Raises:
        ValueError: if both :attr:`C` and :attr:`out_idx` are given, or
            :attr:`B`, :attr:`C` or :attr:`D` has an unsupported shape.
        AssertionError: if the other shapes do not match.

    Example::

        >>> from philtorch.lpv import state_space
        >>> # An accumulator with a growing input gain:
        >>> # h[n + 1] = h[n] + (n + 1) x[n]
        >>> A = torch.ones(4, 1, 1)
        >>> B = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # (N, M)
        >>> state_space(A, torch.ones(1, 4), B=B, C=torch.tensor([1.0]))
        tensor([[0., 1., 3., 6.]])
    """
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"

    assert A.dim() in (3, 4), f"State matrix A must be 3D or 4D, got {A.shape}"
    assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
    assert A.size(-3) == x.size(1), (
        f"State matrix A must have the same time dimension as x, "
        f"got A: {A.size(-3)}, x: {x.size(1)}"
    )

    if not (C is None or out_idx is None):
        raise ValueError("C and out_idx cannot be used together. Use either C or out_idx.")

    batch_size, N, *_ = x.shape
    M = A.size(-1)

    return_zf = True
    if zi is None:
        return_zf = False
        zi = x.new_zeros(batch_size, M)
    elif zi.dim() == 1:
        zi = zi.unsqueeze(0).expand(batch_size, -1)

    if x.dim() == 2:
        features = -1
    else:
        features = x.size(-1)

    if B is not None:
        match B.shape:
            case (BM,) if BM == M:
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when B is of shape {(M,)}, got {x.shape}"
                )
                Bx = x.unsqueeze(-1) * B
            case (BM, F) if BM == M and F == features:
                Bx = x @ B.T
            case (BN, BM) if BN == N and BM == M:
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when B is of shape {batch_size, M}, got {x.shape}"
                )
                Bx = x.unsqueeze(-1) * B
            case (B_batch, BM) if B_batch == batch_size and BM == M:
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when B is of shape {batch_size, M}, got {x.shape}"
                )
                Bx = x.unsqueeze(-1) * B.unsqueeze(1)
            case (BN, BM, F) if BN == N and BM == M and F == features:
                Bx = torch.linalg.vecdot(B.conj(), x.unsqueeze(-2))
            case (B_batch, BN, BM) if B_batch == batch_size and BM == M and BN == N:
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when B is of shape "
                    f"{batch_size, N, M}, got {x.shape}"
                )
                Bx = x.unsqueeze(-1) * B
            case (B_batch, BM, F) if B_batch == batch_size and BM == M and F == features:
                Bx = torch.linalg.vecdot(
                    B.unsqueeze(1).conj(), x.unsqueeze(-2)
                )  # (batch_size, N, M)
            case (B_batch, BN, BM, F) if (
                B_batch == batch_size and BM == M and BN == N and F == features
            ):
                Bx = torch.linalg.vecdot(B.conj(), x.unsqueeze(-2))
            case _:
                raise ValueError(
                    f"Input matrix B must be of shape ({M},), ({batch_size},), "
                    f"({M, features}), ({N, M}), ({batch_size, M}), "
                    f"({N, M, features}), ({batch_size, N, M}), "
                    f"({batch_size, M, features}), or ({batch_size, N, M, features}), "
                    f"got {B.shape}"
                )
    else:
        Bx = x

    h, zf = _ssm.states(
        partial(state_space_recursion, A, zi, Bx, unroll_factor=unroll_factor),
        zi,
        out_idx=out_idx,
        final_state=return_zf,
    )

    if x.dim() == 2:
        features = -1
    else:
        features = x.size(-1)

    if D is not None:
        match D.shape:
            case (F,) if F == features:
                Dx = x @ D
            case (DN,) if DN == N:
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when D is of shape {(N,)}, got {x.shape}"
                )
                Dx = D * x
            case (D_batch,) if D_batch == batch_size:
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when D is of shape {(batch_size,)}, got {x.shape}"
                )
                Dx = D.unsqueeze(1) * x
            case (1,) | ():
                Dx = x * D
            case (_,):
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when D is of shape {D.shape}, got {x.shape}"
                )
                Dx = x.unsqueeze(-1) * D
            case (DN, F) if DN == N and F == features:
                Dx = torch.linalg.vecdot(D.conj(), x)
            case (D_batch, F) if D_batch == batch_size and F == features:
                Dx = torch.linalg.vecdot(D.conj().unsqueeze(1), x)
            case (DN, _) if DN == N:
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when D is of shape ({N, features}), got {x.shape}"
                )
                Dx = D * x.unsqueeze(-1)
            case (D_batch, DN) if D_batch == batch_size and DN == N:
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when D is of shape {batch_size, N}, got {x.shape}"
                )
                Dx = D * x
            case (D_batch, _) if D_batch == batch_size:
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when D is of shape "
                    f"({batch_size, features}), got {x.shape}"
                )
                Dx = D.unsqueeze(1) * x.unsqueeze(-1)
            case (_, F) if F == features:
                assert x.dim() == 3, (
                    f"Input signal x must be 3D when D is of shape {D.shape}, got {x.shape}"
                )
                Dx = x @ D.T
            case (D_batch, DN, F) if D_batch == batch_size and DN == N and F == features:
                Dx = torch.linalg.vecdot(D.conj(), x)
            case (D_batch, DN, _) if D_batch == batch_size and DN == N:
                assert x.dim() == 2, (
                    f"Input signal x must be 2D when D is of shape "
                    f"({batch_size, N, features}), got {x.shape}"
                )
                Dx = D * x.unsqueeze(-1)
            case (DN, _, F) if DN == N and F == features:
                Dx = torch.linalg.vecdot(D.conj(), x.unsqueeze(-2))
            case (D_batch, _, F) if D_batch == batch_size and F == features:
                Dx = x @ D.mT
            case (D_batch, DN, _, F) if D_batch == batch_size and DN == N and F == features:
                Dx = torch.linalg.vecdot(D.conj(), x.unsqueeze(-2))
            case _:
                raise ValueError(
                    f"Input matrix D must be of shape (), (1,), ({N},), "
                    f"({batch_size},), ({features},), ({N, features}), "
                    f"({batch_size, N}), ({batch_size, features}), "
                    f"({features, features}), ({batch_size, N, features}), "
                    f"({N, features, features}), "
                    f"({batch_size, features, features}), "
                    f"or ({batch_size, N, features, features}), got {D.shape}"
                )
    else:
        Dx = None

    if C is not None:
        match C.shape:
            case (CM,) if CM == M:
                Ch = h @ C
            case (CN, CM) if CN == N and CM == M:
                Ch = torch.linalg.vecdot(C.conj(), h)
            case (C_batch, CM) if C_batch == batch_size and CM == M:
                Ch = torch.linalg.vecdot(C.unsqueeze(1).conj(), h)
            case (_, CM) if CM == M:
                Ch = h @ C.T
            case (CN, _, CM) if CN == N and CM == M:
                Ch = torch.linalg.vecdot(C.conj(), h.unsqueeze(-2))
            case (C_batch, CN, CM) if C_batch == batch_size and CM == M and CN == N:
                Ch = torch.linalg.vecdot(C.conj(), h)
            case (C_batch, _, CM) if C_batch == batch_size and CM == M:
                Ch = h @ C.mT
            case (C_batch, CN, _, CM) if C_batch == batch_size and CM == M and CN == N:
                Ch = torch.linalg.vecdot(C.conj(), h.unsqueeze(-2))
            case _:
                raise ValueError(
                    f"Output matrix C must be of shape ({(M,)}), ({N, M}), "
                    f"({batch_size, M}), ({batch_size, N, M}), "
                    f"or ({batch_size, N, features}), got {C.shape}"
                )
    else:
        Ch = h

    if Dx is not None:
        y = Ch + Dx
    else:
        y = Ch

    if return_zf:
        return y, zf
    return y
