import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from torch import Tensor

from ..prototype.utils import a2companion, matrix_power_accumulate


def _recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    out_idx: Optional[int] = None,
) -> Tensor:
    assert x.dim() in (2, 3), "Input signal must be 2D or 3D (batch, time, [features])"
    assert A.dim() in (2, 3), "State matrix A must be 2D or 3D"
    assert A.shape[-2] == A.shape[-1], "State matrix A must be square"
    if A.dim() == 3:
        assert x.shape[0] == A.shape[0], "Batch size of A must match batch size of x"

    if x.dim() == 3:
        assert (
            A.shape[-1] == x.shape[-1]
        ), "Last dimension of A must match last dimension of x"
    assert zi.dim() == 2, "Initial conditions zi must be 2D"
    assert zi.shape[0] == x.shape[0], "Batch size of zi must match batch size of x"
    assert (
        zi.shape[1] == A.shape[-1]
    ), "Last dimension of zi must match last dimension of A"

    results = []
    AT = A.mT
    if A.dim() == 2:
        h = zi
        for xn in x.unbind(1):
            h = torch.addmm(xn, h, AT)
            results.append(h if out_idx is None else h[:, out_idx])
        output = torch.stack(results, dim=1)
    else:
        h = zi.unsqueeze(1)
        for xn in x.unbind(1):
            h = torch.baddbmm(xn.unsqueeze(1), h, AT)
            results.append(h if out_idx is None else h[:, :, out_idx])
        output = torch.cat(results, dim=1)

    return output


def state_space_recursion(
    A: Tensor,
    x: Tensor,
    zi: Optional[Tensor] = None,
    *,
    unroll_factor: Optional[int] = None,
    out_idx: Optional[int] = None,
) -> Tensor:
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"
    assert A.dim() in (2, 3), f"State matrix A must be 2D or 3D, got {A.shape}"
    assert A.shape[-2] == A.shape[-1], f"State matrix A must be square, got {A.shape}"
    if A.dim() == 3:
        assert (
            x.shape[0] == A.shape[0]
        ), f"Batch size of A must match batch size of x, got A: {A.shape[0]}, x: {x.shape[0]}"

    if x.dim() == 3:
        assert (
            A.shape[-1] == x.shape[-1]
        ), f"Last dimension of A must match last dimension of x, got A: {A.shape[-1]}, x: {x.shape[-1]}"

    batch_size, N, *_ = x.shape
    M = A.shape[-1]

    if zi is None:
        zi = x.new_zeros(batch_size, M)
    elif zi.dim() == 1:
        zi = zi.unsqueeze(0).expand(batch_size, -1)
    else:
        assert zi.dim() == 2, f"Initial conditions zi must be 2D, got {zi.shape}"
        assert (
            zi.shape[0] == batch_size
        ), f"Batch size of zi must match batch size of x, got zi: {zi.shape[0]}, x: {batch_size}"
        assert (
            zi.shape[1] == M
        ), f"Last dimension of zi must match last dimension of A, got zi: {zi.shape[1]}, A: {M}"

    if unroll_factor is None:
        block_size = int(N**0.5)
    elif unroll_factor < 1:
        raise ValueError("Unroll factor must be >= 1")
    else:
        block_size = unroll_factor

    # boundary condition
    if block_size == 1 or block_size >= N:
        return _recursion_loop(A, zi, x, out_idx=out_idx)

    remainder = N % block_size
    if remainder != 0:
        x = F.pad(x, (0, 0) * (x.dim() - 2) + (0, block_size - remainder))
        N = x.shape[1]  # Update T after padding

    unrolled_x = x.unflatten(1, (-1, block_size)).flatten(2, -1)

    A_powers = matrix_power_accumulate(A, block_size)
    A_powered = A_powers[..., -1, :, :]
    A_powers_plus_I = torch.cat(
        [
            A_powers[..., :-1, :, :].flip(-3),
            torch.eye(M, device=A.device, dtype=A.dtype)
            .broadcast_to(A.shape)
            .unsqueeze(-3),
        ],
        dim=-3,
    )

    mat1 = (
        A_powers_plus_I.transpose(-2, -1).flatten(-3, -2)
        if x.dim() == 3
        else A_powers_plus_I[..., 0]  # assume x -> x * [1, 0, 0, ...] in the 2D case
    )
    z = unrolled_x @ mat1

    initials = torch.cat(
        [
            zi.unsqueeze(1),
            state_space_recursion(A_powered, z, zi, unroll_factor=unroll_factor),
        ],
        dim=1,
    )

    # prepare the augmented matrix and input for all the remaining steps
    aug_x = torch.cat(
        [initials[:, :-1], unrolled_x[..., : -(1 if x.dim() == 2 else M)]], dim=2
    )

    if out_idx is None:
        mat2 = A_powers[..., :-1, :, :].flatten(-3, -2)
        if x.dim() == 3:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, :, :],
                        A_powers_plus_I.new_zeros(
                            A_powers_plus_I.shape[:-3] + (block_size - 2, M, M)
                        ),
                    ],
                    dim=-3,
                )
                .unfold(-3, block_size - 1, 1)
                .transpose(-2, -1)
                .flip(-4)
                .flatten(-4, -3)
                .flatten(-2, -1)
            )
        else:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, :, 0],
                        A_powers_plus_I.new_zeros(
                            A_powers_plus_I.shape[:-3] + (block_size - 2, M)
                        ),
                    ],
                    dim=-2,
                )
                .unfold(-2, block_size - 1, 1)
                .flip(-3)
                .flatten(-3, -2)
            )
    else:
        mat2 = A_powers[..., :-1, out_idx, :]
        if x.dim() == 3:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, out_idx, :],
                        A_powers_plus_I.new_zeros(
                            A_powers_plus_I.shape[:-3] + (block_size - 2, M)
                        ),
                    ],
                    dim=-2,
                )
                .unfold(-2, block_size - 1, 1)
                .transpose(-2, -1)
                .flip(-3)
                .flatten(-2, -1)
            )
        else:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, out_idx, 0],
                        A_powers_plus_I.new_zeros(
                            A_powers_plus_I.shape[:-3] + (block_size - 2,)
                        ),
                    ],
                    dim=-1,
                )
                .unfold(-1, block_size - 1, 1)
                .flip(-2)
            )

    aug_A = torch.cat([mat2.mT, mat3.mT], dim=-2)
    output = aug_x @ aug_A

    # concat the first M - 1 outputs with the last one
    if out_idx is None:
        output = (
            torch.cat([output, initials[:, 1:, :]], dim=2)
            .unflatten(2, (-1, M))
            .flatten(1, 2)
        )
    else:
        output = torch.cat([output, initials[:, 1:, out_idx, None]], dim=2).flatten(
            1, 2
        )
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output
