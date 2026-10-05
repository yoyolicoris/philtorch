"""State-space models with time-varying coefficients."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

from .._torchlpc import lpc
from ..lti.ssm import helion_backend_indicator
from ..mat import matrices_cumdot


def extension_backend_indicator(x: Tensor, M: int) -> bool:
    r"""Return whether to dispatch this input to the native extension.

    This is a dispatch heuristic, not a check that a kernel exists: ``True``
    on CPU, for :math:`M \le 2` on every device, and where the ParaRNN
    kernels apply (see :func:`_pararnn_applicable`). Some kernels are still
    missing for a device or dtype, such as on MPS, and calling one raises an
    error; the README lists them.

    Args:
        x (Tensor): the input, whose device and dtype are checked.
        M (int): the state size.

    Returns:
        bool: whether to call the native extension.
    """
    return x.is_cpu or M <= 2 or _pararnn_applicable(x, M)


def _pararnn_applicable(x: Tensor, M: int) -> bool:
    """Return whether the vendored ParaRNN kernels support this input.

    They need a real floating-point input on CUDA, with :math:`M = 2` or
    :math:`3`.
    """
    return x.is_cuda and x.is_floating_point() and M in (2, 3)


class MatrixRecurrence(Function):
    r"""Autograd function for the matrix recurrence of :func:`_ext_ss_recur`.

    The forward pass runs the ParaRNN kernels where they apply, the native
    ``recur2`` kernel for :math:`M = 2`, the Helion kernel on CUDA when it
    is available, and the ``recurN`` kernel otherwise. The backward pass
    runs the recurrence backward in time with :math:`A[n]^H`, and the JVP
    runs it forward on the tangents.
    """

    @staticmethod
    def forward(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
        if _pararnn_applicable(x, x.size(-1)):
            if A.ndim == 3:
                A = A.repeat(x.size(0), 1, 1, 1)
            jac = F.pad(-A, (0, 0, 0, 0, 1, 0))
            rhs = torch.cat([zi.unsqueeze(1), x], dim=1)
            return (
                torch.ops.parallel_reduce_cuda.parallel_reduce_block_diag_3x3_cuda
                if x.size(-1) == 3
                else torch.ops.parallel_reduce_cuda.parallel_reduce_block_diag_2x2_cuda
            )(jac, rhs)[:, 1:]
        elif x.size(-1) == 2:
            return torch.ops.philtorch.recur2(A, zi, x)
        elif helion_backend_indicator(x):
            from .. import hl_recurN

            return hl_recurN(A, zi, x)
        return torch.ops.philtorch.recurN(A, zi, x)

    @staticmethod
    def setup_context(ctx: Any, inputs: list[Any], output: Any) -> Any:
        A, zi, _ = inputs
        y = output
        ctx.save_for_backward(A, zi, y)
        ctx.save_for_forward(A, zi, y)

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        A, zi, y = ctx.saved_tensors
        grad_x = grad_A = grad_zi = None

        AmT = A.mT.conj_physical()
        AmT_rolled = torch.roll(AmT, shifts=-1, dims=-3)

        flipped_grad_x = MatrixRecurrence.apply(
            AmT_rolled.flip(-3),
            torch.zeros_like(zi),
            grad_y.flip(1),
        )

        if ctx.needs_input_grad[1]:
            grad_zi = (AmT[..., 0, :, :] @ flipped_grad_x[:, -1, :, None]).squeeze(-1)

        if ctx.needs_input_grad[2]:
            grad_x = flipped_grad_x.flip(1)

        if ctx.needs_input_grad[0]:
            valid_y = y[:, :-1]
            padded_y = torch.cat([zi.unsqueeze(1), valid_y], dim=1)

            if A.dim() == 3:
                grad_A = flipped_grad_x.flip(1).permute(
                    1, 2, 0
                ) @ padded_y.conj_physical().transpose(0, 1)
            else:
                grad_A = padded_y.conj_physical().unsqueeze(-2) * flipped_grad_x.flip(1).unsqueeze(
                    -1
                )

        return grad_A, grad_zi, grad_x

    @staticmethod
    def jvp(
        ctx: Any, grad_A: torch.Tensor, grad_zi: torch.Tensor, grad_x: torch.Tensor
    ) -> torch.Tensor:
        A, zi, y = ctx.saved_tensors

        fwd_zi = grad_zi if grad_zi is not None else torch.zeros_like(zi)
        fwd_x = grad_x if grad_x is not None else torch.zeros_like(y)

        if grad_A is not None:
            padded_y = torch.cat([zi.unsqueeze(1), y[:, :-1]], dim=1)
            fwd_A = (grad_A @ padded_y.unsqueeze(-1)).squeeze(-1)
            fwd_x = fwd_x + fwd_A

        return MatrixRecurrence.apply(A, fwd_zi, fwd_x)


def _matrix_recurrence(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    if torch.compiler.is_compiling():
        return MatrixRecurrence.forward(A, zi, x)
    return MatrixRecurrence.apply(A, zi, x)


def _recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    out_idx: int | None = None,
) -> Tensor:
    r"""Run the time-varying matrix recurrence with a loop over time steps.

    This computes
    :math:`\mathbf{h}[n] = A[n] \mathbf{h}[n - 1] + \mathbf{x}[n]`.

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


def _ext_ss_recur(A: Tensor, zi: Tensor, x: Tensor, *, out_idx: int | None = None, **_) -> Tensor:
    """Run the time-varying matrix recurrence with the native kernels.

    Takes the same arguments and returns the same as :func:`_recursion_loop`.
    """
    if x.dim() == 2 and A.size(-1) == 1:
        y = lpc(x, -A[..., 0].broadcast_to(x.shape + (1,)), zi).unsqueeze(-1)
    elif A.size(-1) == 1:
        y = lpc(x.squeeze(-1), -A[..., 0].broadcast_to(x.shape), zi).unsqueeze(-1)
    else:
        x = (
            torch.cat([x.unsqueeze(-1), x.new_zeros(*x.shape, A.size(-1) - 1)], dim=-1)
            if x.dim() == 2
            else x
        )
        y = _matrix_recurrence(A, zi, x)
    if out_idx is not None and y.dim() == 3:
        y = y[:, :, out_idx]
    return y


def _select_recursion_runner(x: Tensor, M: int, unroll_factor: int):
    if unroll_factor == 1 and extension_backend_indicator(x, M):
        return _ext_ss_recur
    return _recursion_loop


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
        \mathbf{h}[n] = A[n] \mathbf{h}[n - 1] + \mathbf{x}[n],
        \quad n = 0, \dots, N - 1,

    starting from :math:`\mathbf{h}[-1] = \mathbf{z}_i`. A 2-D input feeds
    the first state only: :math:`\mathbf{x}[n] = x[n] \mathbf{e}_1`.

    With ``unroll_factor=1``, this calls the native extension on CPU, on
    other devices when :math:`M \le 2`, and on CUDA for real floating-point
    inputs with :math:`M = 3`; otherwise it runs a loop over time steps. On
    other devices the extension may lack a kernel for the device or dtype,
    such as on MPS, which raises an error; the README lists them. On any
    device, an :attr:`unroll_factor` greater than 1 and less than :math:`N`
    runs a block-unrolled PyTorch recursion instead, and one of :math:`N` or
    more runs a plain PyTorch loop.

    Args:
        A (Tensor): state matrices :math:`A[n]`, of shape :math:`(N, M, M)` to
            share them or :math:`(B, N, M, M)`.
        zi (Tensor): the initial state :math:`\mathbf{h}[-1]`, of shape
            :math:`(B, M)`.
        x (Tensor): inputs, of shape :math:`(B, N, M)` or :math:`(B, N)`.
        unroll_factor (int, optional): ``1`` for the dispatch described
            above, a value less than :math:`N` for the block length of the
            unrolled recursion, or :math:`N` or more for a plain loop.
            Default: ``1``.
        out_idx (int, optional): return only this state. Default: ``None``.

    Returns:
        Tensor: the states :math:`\mathbf{h}[0], \dots, \mathbf{h}[N - 1]`,
        of shape :math:`(B, N, M)`, or :math:`(B, N)` with :attr:`out_idx`.

    Raises:
        AssertionError: if the shapes do not match.
        ValueError: if :attr:`unroll_factor` is less than 1.

    Note:
        Unlike :func:`state_space`, the state at step :math:`n` already
        includes :math:`\mathbf{x}[n]`.

    Example::

        >>> from philtorch.lpv import state_space_recursion
        >>> A = torch.tensor([[1.0, 1.0], [1.0, 0.0]]).expand(5, 2, 2)
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        >>> state_space_recursion(A, torch.zeros(1, 2), x, out_idx=0)
        tensor([[1., 1., 2., 3., 5.]])
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

    if A.dim() == 4:
        assert x.size(0) == A.size(0), (
            f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {x.size(0)}"
        )

    if x.dim() == 3:
        assert A.size(-1) == x.size(-1), (
            f"Last dimension of A must match last dimension of x, "
            f"got A: {A.size(-1)}, x: {x.size(-1)}"
        )

    batch_size, N = x.size(0), x.size(1)
    M = A.size(-1)

    assert zi.dim() == 2, f"Initial conditions zi must be 2D, got {zi.shape}"
    assert zi.size(0) == batch_size, (
        f"Batch size of zi must match batch size of x, got zi: {zi.size(0)}, x: {batch_size}"
    )
    assert zi.size(1) == M, (
        f"Last dimension of zi must match last dimension of A, got zi: {zi.size(1)}, A: {M}"
    )

    if unroll_factor < 1:
        raise ValueError("Unroll factor must be >= 1")
    else:
        block_size = unroll_factor

    runner = _select_recursion_runner(x, M, block_size)

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
    :math:`(B, N)` for both. When two readings fit, such as :math:`N = B`,
    the time-varying one is tried first, then the per-signal one.

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
            :math:`()` or :math:`(P)` for a 2-D :attr:`x`, and :math:`()`,
            :math:`(F)` or :math:`(P, F)` for a 3-D one; :math:`(1)` is also
            accepted for a shared scalar. Default: ``None``, no feedthrough.
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
