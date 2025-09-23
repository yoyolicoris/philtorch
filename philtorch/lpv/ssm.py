import torch
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Union, Tuple, Any, List
from torch import Tensor
from torchlpc import sample_wise_lpc

from ..mat import matrices_cumdot
from .. import EXTENSION_LOADED


def extension_backend_indicator(x: Tensor, M: int) -> bool:
    """Return True when a compiled extension backend is preferable.

    The indicator prefers the extension when it is available and when the
    problem size or device favors the extension implementation.
    """
    return EXTENSION_LOADED and (x.is_cpu or (M <= 2))


class MatrixRecurrence(Function):
    """Autograd Function for matrix recurrence using the compiled ops."""

    @staticmethod
    def forward(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
        if x.size(-1) == 2:
            return torch.ops.philtorch.recur2(A, zi, x)
        return torch.ops.philtorch.recurN(A, zi, x)

    @staticmethod
    def setup_context(ctx: Any, inputs: List[Any], output: Any) -> Any:
        A, zi, _ = inputs
        y = output
        ctx.save_for_backward(A, zi, y)
        ctx.save_for_forward(A, zi, y)

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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
                grad_A = padded_y.conj_physical().unsqueeze(-2) * flipped_grad_x.flip(
                    1
                ).unsqueeze(-1)

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


def _recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    out_idx: Optional[int] = None,
) -> Tensor:
    """Pure-Python state-space recursion loop.

    This helper runs the recurrence in Python for cases where the compiled
    extension is not used. It supports time-varying A (with a time axis at
    -3) and both batched and unbatched forms.

    Args:
        A (Tensor): State matrices with shape (N, M, M) or (B, N, M, M).
        zi (Tensor): Initial states with shape (B, M).
        x (Tensor): Input sequence with shape (B, N, M) or (B, N).
        out_idx (int, optional): If provided, return only this state index
            per time step (reduces last dim).

    Returns:
        Tensor: Sequence of states with shape (B, N, M) or (B, N) if
            ``out_idx`` is provided.
    """
    assert x.size(1) == A.size(
        -3
    ), f"State matrix A must have the same time dimension as x, got A: {A.size(-3)}, x: {x.size(1)}"
    results = []
    AT = A.mT
    if x.dim() == 2:
        M = A.size(-1)
        x = torch.cat(
            [x.unsqueeze(-1), x.new_zeros(*x.shape, M - 1)], dim=-1
        )  # (batch, time, M)
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


def _ext_ss_recur(
    A: Tensor, zi: Tensor, x: Tensor, *, out_idx: Optional[int] = None, **_
) -> Tensor:
    """Call the compiled extension for state-space recursion when available.

    This wrapper delegates to a compiled backend (if built) for better
    performance.

    Args:
        A (Tensor): State matrices.
        zi (Tensor): Initial states.
        x (Tensor): Input sequence.
        out_idx (int, optional): If set and the returned y has a state
            dimension, selects the single output index to return.

    Returns:
        Tensor: Output states (B, N, M) or (B, N) if ``out_idx`` is set.
    """
    assert (
        EXTENSION_LOADED
    ), "Extension not loaded. Please ensure philtorch is built with extensions."
    if x.dim() == 2 and A.size(-1) == 1:
        y = sample_wise_lpc(x, -A[..., 0].broadcast_to(x.shape + (1,)), zi).unsqueeze(
            -1
        )
    elif A.size(-1) == 1:
        y = sample_wise_lpc(
            x.squeeze(-1), -A[..., 0].broadcast_to(x.shape), zi
        ).unsqueeze(-1)
    else:
        x = (
            torch.cat([x.unsqueeze(-1), x.new_zeros(*x.shape, A.size(-1) - 1)], dim=-1)
            if x.dim() == 2
            else x
        )
        y = MatrixRecurrence.apply(A, zi, x)
    if out_idx is not None and y.dim() == 3:
        y = y[:, :, out_idx]
    return y


def state_space_recursion(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    *,
    unroll_factor: int = 1,
    out_idx: Optional[int] = None,
) -> Tensor:
    """Compute the internal state sequence for a time-varying discrete
    state-space model.

    This is a Pure-Python implementation of the state recursion.
    The function supports block unrolling to accelerate long recurrences and
    can operate on batched or unbatched state matrices.

    Args:
        A (Tensor): State matrices with shape (N, M, M) or (B, N, M, M).
        zi (Tensor): Initial states with shape (B, M).
        x (Tensor): Input sequence with shape (B, N, M) or (B, N).
        unroll_factor (int): Block unroll factor to accelerate long
            recurrences (default 1 = no unrolling).
        out_idx (int, optional): If provided, return only this state index
            per time step (reduces last dim).

    Returns:
        Tensor: State sequence (B, N, M) or (B, N) if ``out_idx`` is provided.
    """
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"
    assert A.dim() in (3, 4), f"State matrix A must be 3D or 4D, got {A.shape}"
    assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
    assert A.size(-3) == x.size(
        1
    ), f"State matrix A must have the same time dimension as x, got A: {A.size(-3)}, x: {x.size(1)}"

    if A.dim() == 4:
        assert x.size(0) == A.size(
            0
        ), f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {x.size(0)}"

    if x.dim() == 3:
        assert A.size(-1) == x.size(
            -1
        ), f"Last dimension of A must match last dimension of x, got A: {A.size(-1)}, x: {x.size(-1)}"

    batch_size, N = x.size(0), x.size(1)
    M = A.size(-1)

    assert zi.dim() == 2, f"Initial conditions zi must be 2D, got {zi.shape}"
    assert (
        zi.size(0) == batch_size
    ), f"Batch size of zi must match batch size of x, got zi: {zi.size(0)}, x: {batch_size}"
    assert (
        zi.size(1) == M
    ), f"Last dimension of zi must match last dimension of A, got zi: {zi.size(1)}, A: {M}"

    if unroll_factor < 1:
        raise ValueError("Unroll factor must be >= 1")
    else:
        block_size = unroll_factor

    # boundary condition
    if block_size == 1 or block_size >= N:
        return _recursion_loop(A, zi, x, out_idx=out_idx)

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

    output = _recursion_loop(
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
        output = torch.cat([output, initials[:, 1:, out_idx, None]], dim=2).flatten(
            1, 2
        )
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output


def state_space(
    A: Tensor,
    x: Tensor,
    B: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    D: Optional[Tensor] = None,
    zi: Optional[Tensor] = None,
    unroll_factor: Optional[int] = None,
    out_idx: Optional[int] = None,
    # **kwargs,
):
    """Compute outputs from a discrete parameter-varying state-space model.

    This utility evaluates the standard state-space equations with time-varying
    state matrices:

        h_{t+1} = A_t @ h_t + B_t x_t
        y_t = C_t h_t + D_t x_t

    The function accepts many broadcastable shapes for B, C and D to support
    common use-cases. If ``zi`` is provided the function also returns the
    final state ``zf`` as a second value (y, zf).

    Args:
        A (Tensor): State matrices with shape (N, M, M) or (B, N, M, M).
        x (Tensor): Input sequence with shape (B, N, F) or (B, N).
        B (Tensor, optional): Input matrices with broadcastable shape.
        C (Tensor, optional): Output matrices with broadcastable shape.
        D (Tensor, optional): Feedthrough matrices with broadcastable shape.
        zi (Tensor, optional): Initial states with shape (B, M). If not
            provided, assumed to be zero. If provided, the final state
            ``zf`` is also returned.
        unroll_factor (int, optional): Block unroll factor to accelerate
            long recurrences if the pure-Python backend is used.
            If ``None``, it is set to ``round(N**0.5)``.
        out_idx (int, optional): If set and the returned y has a state
            dimension, selects the single output index to return.
    Returns:
        Output sequence (B, N, F) or (B, N) if ``out_idx`` is set.
        If ``zi`` is provided, also returns the final state (B, M).
    """
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"

    assert A.dim() in (3, 4), f"State matrix A must be 3D or 4D, got {A.shape}"
    assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
    assert A.size(-3) == x.size(
        1
    ), f"State matrix A must have the same time dimension as x, got A: {A.size(-3)}, x: {x.size(1)}"

    if not (C is None or out_idx is None):
        raise ValueError(
            "C and out_idx cannot be used together. Use either C or out_idx."
        )

    batch_size, N, *_ = x.shape
    M = A.size(-1)

    return_zf = True
    if zi is None:
        return_zf = False
        zi = x.new_zeros(batch_size, M)
    elif zi.dim() == 1:
        zi = zi.unsqueeze(0).expand(batch_size, -1)

    if unroll_factor is None:
        unroll_factor = round(N**0.5)

    if x.dim() == 2:
        features = -1
    else:
        features = x.size(-1)

    if B is not None:
        match B.shape:
            case (BM,) if BM == M:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when B is of shape {M,}, got {x.shape}"
                Bx = x.unsqueeze(-1) * B
            case (BM, F) if BM == M and F == features:
                Bx = x @ B.T
            case (BN, BM) if BN == N and BM == M:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when B is of shape {batch_size, M}, got {x.shape}"
                Bx = x.unsqueeze(-1) * B
            case (B_batch, BM) if B_batch == batch_size and BM == M:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when B is of shape {batch_size, M}, got {x.shape}"
                Bx = x.unsqueeze(-1) * B.unsqueeze(1)
            case (BN, BM, F) if BN == N and BM == M and F == features:
                Bx = torch.linalg.vecdot(B.conj(), x.unsqueeze(-2))
            case (B_batch, BN, BM) if B_batch == batch_size and BM == M and BN == N:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when B is of shape {batch_size, N, M}, got {x.shape}"
                Bx = x.unsqueeze(-1) * B
            case (B_batch, BM, F) if (
                B_batch == batch_size and BM == M and F == features
            ):
                Bx = torch.linalg.vecdot(
                    B.unsqueeze(1).conj(), x.unsqueeze(-2)
                )  # (batch_size, N, M)
            case (B_batch, BN, BM, F) if (
                B_batch == batch_size and BM == M and BN == N and F == features
            ):
                Bx = torch.linalg.vecdot(B.conj(), x.unsqueeze(-2))
            case _:
                raise ValueError(
                    f"Input matrix B must be of shape ({M},), ({batch_size},), ({M, features}), ({N, M}), ({batch_size, M}), ({N, M, features}), ({batch_size, N, M}), ({batch_size, M, features}), or ({batch_size, N, M, features}), got {B.shape}"
                )
    else:
        Bx = x

    recur_runner = (
        _ext_ss_recur if extension_backend_indicator(x, M) else state_space_recursion
    )

    if return_zf or out_idx is None:
        h = recur_runner(A, zi, Bx, unroll_factor=unroll_factor, out_idx=None)
        zf = h[:, -1, :] if return_zf else None
        h = (
            torch.cat([zi.unsqueeze(1), h[:, :-1]], dim=1)
            if out_idx is None
            else torch.cat([zi[:, None, out_idx], h[:, :-1, out_idx]], dim=1)
        )
    else:
        zf = None
        h = recur_runner(A, zi, Bx, unroll_factor=unroll_factor, out_idx=out_idx)
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
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when D is of shape {N,}, got {x.shape}"
                Dx = D * x
            case (D_batch,) if D_batch == batch_size:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when D is of shape {batch_size,}, got {x.shape}"
                Dx = D.unsqueeze(1) * x
            case (1,) | ():
                Dx = x * D
            case (_,):
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when D is of shape {D.shape}, got {x.shape}"
                Dx = x.unsqueeze(-1) * D
            case (DN, F) if DN == N and F == features:
                Dx = torch.linalg.vecdot(D.conj(), x)
            case (D_batch, F) if D_batch == batch_size and F == features:
                Dx = torch.linalg.vecdot(D.conj().unsqueeze(1), x)
            case (DN, _) if DN == N:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when D is of shape ({N, features}), got {x.shape}"
                Dx = D * x.unsqueeze(-1)
            case (D_batch, DN) if D_batch == batch_size and DN == N:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when D is of shape {batch_size, N}, got {x.shape}"
                Dx = D * x
            case (D_batch, _) if D_batch == batch_size:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when D is of shape ({batch_size, features}), got {x.shape}"
                Dx = D.unsqueeze(1) * x.unsqueeze(-1)
            case (_, F) if F == features:
                assert (
                    x.dim() == 3
                ), f"Input signal x must be 3D when D is of shape {D.shape}, got {x.shape}"
                Dx = x @ D.T
            case (D_batch, DN, F) if (
                D_batch == batch_size and DN == N and F == features
            ):
                Dx = torch.linalg.vecdot(D.conj(), x)
            case (D_batch, DN, _) if D_batch == batch_size and DN == N:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when D is of shape ({batch_size, N, features}), got {x.shape}"
                Dx = D * x.unsqueeze(-1)
            case (DN, _, F) if DN == N and F == features:
                Dx = torch.linalg.vecdot(D.conj(), x.unsqueeze(-2))
            case (D_batch, _, F) if D_batch == batch_size and F == features:
                Dx = x @ D.mT
            case (D_batch, DN, _, F) if (
                D_batch == batch_size and DN == N and F == features
            ):
                Dx = torch.linalg.vecdot(D.conj(), x.unsqueeze(-2))
            case _:
                raise ValueError(
                    f"Input matrix D must be of shape (), (1,), ({N},), ({batch_size},), ({features},), ({N, features}), ({batch_size, N}), ({batch_size, features}), ({features, features}), ({batch_size, N, features}), ({N, features, features}), ({batch_size, features, features}), or ({batch_size, N, features, features}), got {D.shape}"
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
                    f"Output matrix C must be of shape ({M,}), ({N, M}), ({batch_size, M}), ({batch_size, N, M}), or ({batch_size, N, features}), got {C.shape}"
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
