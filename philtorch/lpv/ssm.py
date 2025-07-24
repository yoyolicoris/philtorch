import torch
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Union, Tuple, Any, List
from torch import Tensor

from ..mat import matrices_cumdot
from ..lti.ssm import _ssm_C_D
from .. import EXTENSION_LOADED


class SecondOrderRecurrence(Function):
    @staticmethod
    def forward(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
        return torch.ops.philtorch.recur2(A, zi, x)

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

        flipped_grad_x = SecondOrderRecurrence.apply(
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
            grad_A = padded_y.conj_physical().unsqueeze(-2) * flipped_grad_x.flip(
                1
            ).unsqueeze(-1)
            if A.dim() == 3:
                grad_A = grad_A.sum(0)

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

        return SecondOrderRecurrence.apply(A, fwd_zi, fwd_x)


def _recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    out_idx: Optional[int] = None,
) -> Tensor:
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


def state_space_recursion(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    *,
    unroll_factor: int = 1,
    out_idx: Optional[int] = None,
) -> Tensor:
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

    # if M == 2 and x.is_cuda and EXTENSION_LOADED:
    #     # Special case for 2D state space, use the extension
    #     if x.dim() == 2:
    #         x = torch.stack([x, torch.zeros_like(x)], dim=-1)
    #     output = SecondOrderRecurrence.apply(A, zi, x)
    #     if out_idx is not None:
    #         output = output[:, :, out_idx]
    #     return output

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

    if B is not None:
        match B.shape:
            case (BM,) if BM == M:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when B is of shape {M,}, got {x.shape}"
                Bx = x.unsqueeze(-1) * B
            case (B_batch, BM) if B_batch == batch_size and BM == M:
                assert (
                    x.dim() == 2
                ), f"Input signal x must be 2D when B is of shape {batch_size, M}, got {x.shape}"
                Bx = x.unsqueeze(-1) * B.unsqueeze(1)
            case (BM, _) if BM == M:
                Bx = x @ B.T
            case (B_batch, BM, _) if B_batch == batch_size and BM == M:
                Bx = torch.linalg.vecdot(
                    B.unsqueeze(1).conj(), x.unsqueeze(-2)
                )  # (batch_size, N, M)
            case _:
                raise ValueError(
                    f"Input matrix B must be of shape ({M,}), ({M}, features), ({batch_size}, {M}), or ({batch_size}, {M}, features), got {B.shape}"
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
        h = state_space_recursion(
            A, zi, Bx, unroll_factor=unroll_factor, out_idx=out_idx
        )
        h = torch.cat([zi[:, None, out_idx], h[:, :-1]], dim=1)

    y = _ssm_C_D(h, x, C, D, batch_size, M)

    if return_zf:
        return y, zf
    return y
