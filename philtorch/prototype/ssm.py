import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional

from sympy.ntheory import factorint
from itertools import accumulate
import math


def exp_state_space_recursion(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    *,
    unroll_factor: int = 1,
    out_idx: Optional[int] = None,
) -> Tensor:
    """Compute internal state evolution for an LTI model.

    This is a work-inefficient implementation of the parallel prefix sum algorithm.

    Args:
        A (Tensor): State matrices with shape (M, M) or (B, M, M) when batched.
        zi (Tensor): Initial states with shape (B, M).
        x (Tensor): Input sequence with shape (B, T, ...) or (B, T).
        unroll_factor (int): Block size for recursion acceleration (>=1).
        out_idx (int, optional): If provided, return only this state index.

    Returns:
        Tensor: State sequence (B, T, M) or (B, T) when ``out_idx`` is used.
    """
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"
    assert A.dim() in (2, 3), f"State matrix A must be 2D or 3D, got {A.shape}"
    assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
    if A.dim() == 3:
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

    if x.dim() == 2:
        x = torch.cat(
            [x.unsqueeze(-1), x.new_zeros(*x.shape, M - 1)], dim=-1
        )  # (batch, time, M)
    # x = torch.cat([zi.unsqueeze(1), x], dim=1)  # prepend zi to the input sequence
    # N += 1  # account for the prepended zi

    x = x.mT
    factors = factorint(N, multiple=True)
    dilation = 1
    if A.dim() == 3:
        x = x.reshape(1, -1, N)
        zi = zi.reshape(1, -1)
    for fac in factors:
        A_powered = torch.stack(list(accumulate([A] * fac, torch.matmul)), dim=-3)
        A = A_powered[..., -1, :, :]
        A_powered = A_powered[..., :-1, :, :]

        weight = torch.movedim(A_powered, -3, -1).flip(-1)
        if A.dim() == 3:
            weight = weight.flatten(0, 1)
        x = (
            F.pad(
                F.conv1d(
                    F.pad(
                        torch.cat([zi.unsqueeze(-1), x[..., :-dilation]], dim=-1),
                        ((weight.size(-1) - 1) * dilation, 0),
                    ),
                    weight,
                    dilation=dilation,
                    groups=batch_size if A.dim() == 3 else 1,
                ),
                (dilation - 1, 0),
            )
            + x
        )
        dilation *= fac

    if A.dim() == 3:
        x = x.reshape(-1, M, N)
    x = x.mT

    if out_idx is None:
        output = x
    else:
        output = x[:, :, out_idx]

    return output
