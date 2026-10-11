"""State-space models with time-varying coefficients."""

import torch
import torch.nn.functional as F
from torch import Tensor

from ..mat import matrices_cumdot


def _recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    out_idx: int | None = None,
) -> Tensor:
    r"""Run the time-varying matrix recurrence with a loop over time steps.

    This computes
    :math:`\mathbf{h}[n + 1] = A[n] \mathbf{h}[n] + \mathbf{x}[n]` from
    :math:`\mathbf{h}[0] = \mathbf{z}_i` and returns
    :math:`\mathbf{h}[1], \dots, \mathbf{h}[N]`.

    Args:
        A (Tensor): state matrices, of shape :math:`(N, M, M)` or
            :math:`(B, N, M, M)`.
        zi (Tensor): the initial state, of shape :math:`(B, M)`.
        x (Tensor): inputs, of shape :math:`(B, N, M)`, or :math:`(B, N)` to
            feed the first state only.
        out_idx (int, optional): return only this state. Default: ``None``.

    Returns:
        Tensor: the states, of shape :math:`(B, N, M)`, or :math:`(B, N)`
        with :attr:`out_idx`.
    """
    assert x.size(1) == A.size(-3), (
        f"State matrix A must have the same time dimension as x, "
        f"got A: {A.size(-3)}, x: {x.size(1)}"
    )
    results = []
    AT = A.mT
    if x.dim() == 2:
        M = A.size(-1)
        x = torch.cat([x.unsqueeze(-1), x.new_zeros(*x.shape, M - 1)], dim=-1)  # (batch, time, M)
    if A.dim() == 3:
        h = zi
        for xn, AnT in zip(x.unbind(1), AT.unbind(0)):
            h = torch.addmm(xn, h, AnT)
            results.append(h if out_idx is None else h[:, out_idx])
        output = torch.stack(results, dim=1)
    else:
        h = zi.unsqueeze(1)
        for xn, AnT in zip(x.unbind(1), AT.unbind(1)):
            h = torch.baddbmm(xn.unsqueeze(1), h, AnT)
            results.append(h if out_idx is None else h[:, :, out_idx])
        output = torch.cat(results, dim=1)

    return output


def _kernel_recursion(A: Tensor, zi: Tensor, x: Tensor, out_idx: int | None = None) -> Tensor:
    """Run the time-varying matrix recurrence with
    :func:`philtorch._recur.recurrence`'s kernels, or with :func:`_recursion_loop`
    where none takes the input.

    Takes the same arguments and returns the same as :func:`_recursion_loop`.
    """
    # Imported here, so that importing philtorch doesn't load Triton.
    from .._recur import recurrence

    M = A.size(-1)
    states = F.pad(x.unsqueeze(-1), (0, M - 1)) if x.dim() == 2 else x
    try:
        h = recurrence(A[None] if A.dim() == 3 else A, zi, states)
    except NotImplementedError:
        return _recursion_loop(A, zi, x, out_idx=out_idx)
    return h if out_idx is None else h[..., out_idx]


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

    batch_size, N = x.size(0), x.size(1)
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

    if unroll_factor < 1:
        raise ValueError("Unroll factor must be >= 1")
    else:
        block_size = unroll_factor

    runner = _kernel_recursion if block_size == 1 else _recursion_loop

    # boundary condition
    if block_size == 1 or block_size >= N:
        return runner(A, zi, x, out_idx=out_idx)

    remainder = N % block_size
    if remainder != 0:
        x = F.pad(x, (0, 0) * (x.dim() - 2) + (0, block_size - remainder))
        A = F.pad(A, (0,) * 4 + (0, block_size - remainder))  # pad
        N = x.size(1)  # Update T after padding

    unrolled_x = x.unflatten(1, (-1, block_size))
    unrolled_x_flatten = unrolled_x.flatten(2, -1)
    unrolled_A = A.unflatten(-3, (-1, block_size))

    A_cums = matrices_cumdot(unrolled_A[..., 1:, :, :].flip(-3)).flip(-3)
    A_last_cum = A_cums[..., 0, :, :] @ unrolled_A[..., 0, :, :]
    A_cums_plus_I = torch.cat(
        [
            A_cums,
            torch.eye(M, device=A.device, dtype=A.dtype).broadcast_to(
                A_cums.shape[:-3] + (1, M, M)
            ),
        ],
        dim=-3,
    )
    mat1 = (
        A_cums_plus_I.mT.flatten(-3, -2)
        if x.dim() == 3
        else A_cums_plus_I[..., 0]  # assume x -> x * [1, 0, 0, ...] in the 2D case
    )
    z = torch.squeeze(unrolled_x_flatten.unsqueeze(-2) @ mat1, -2)

    initials = torch.cat(
        [
            zi.unsqueeze(1),
            state_space_recursion(A_last_cum, zi, z, unroll_factor=unroll_factor),
        ],
        dim=1,
    )

    output = runner(
        (
            unrolled_A[:, :, :-1].flatten(0, 1)
            if unrolled_A.dim() == 5
            else unrolled_A[:, :-1].repeat(batch_size, 1, 1, 1)
        ),
        initials[:, :-1].flatten(0, 1),
        unrolled_x[:, :, :-1].flatten(0, 1),
        out_idx=out_idx,
    ).unflatten(0, (batch_size, -1))

    # concat the first M - 1 outputs with the last one
    if out_idx is None:
        output = torch.cat([output, initials[:, 1:, None, :]], dim=2).flatten(1, 2)
    else:
        output = torch.cat([output, initials[:, 1:, out_idx, None]], dim=2).flatten(1, 2)
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output


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

    if return_zf or out_idx is None:
        h = state_space_recursion(A, zi, Bx, unroll_factor=unroll_factor, out_idx=None)
        zf = h[:, -1, :] if return_zf else None
        h = (
            torch.cat([zi.unsqueeze(1), h[:, :-1]], dim=1)
            if out_idx is None
            else torch.cat([zi[:, None, out_idx], h[:, :-1, out_idx]], dim=1)
        )
    else:
        zf = None
        h = state_space_recursion(A, zi, Bx, unroll_factor=unroll_factor, out_idx=out_idx)
        h = torch.cat([zi[:, None, out_idx], h[:, :-1]], dim=1)

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
