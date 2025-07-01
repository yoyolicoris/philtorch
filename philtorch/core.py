import torch
from torch import Tensor
from torchlpc import sample_wise_lpc
from typing import Optional, Union, Tuple, List
from itertools import accumulate


def lpv_fir(
    b: Tensor, x: Tensor, zi: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of parameter-varying FIR filters to input signal
    Args:
        b (Tensor): Coefficients of the FIR filters, shape (B, T, N+1).
        x (Tensor): Input signal, shape (B, T).
        zi (Tensor, optional): Initial conditions for the filter, shape (B, N).
    Returns:
        Filtered output signal, shape (B, T), and optionally the final state of the filter.
    """
    assert b.dim() == 3, "Numerator coefficients b must be 3D."
    assert x.dim() == 2, "Input signal x must be 2D."

    B, T = x.shape
    assert (
        b.shape[:2] == x.shape
    ), "The first two dimensions of b must match the shape of x."

    if zi is None:
        return_zf = False
        zi = x.new_zeros((B, b.shape[2] - 1))
    else:
        assert zi.dim() == 2, "Initial conditions zi must be 2D."
        assert zi.shape[0] == B, "The first dimension of zi must match the batch size."
        assert (
            zi.shape[1] == b.shape[2] - 1
        ), "The second dimension of zi must match the filter order."

        return_zf = True

    unfolded_x = torch.cat([zi.flip(1), x], dim=1).unfold(1, b.shape[2], 1)
    y = torch.linalg.vecdot(unfolded_x, b.flip(2))

    if return_zf:
        return y, unfolded_x[:, -1, :-1].flip(1)
    return y


def lpv_allpole(
    a: Tensor, x: Tensor, zi: Optional[Tensor] = None
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of parameter-varying all-pole filters to input signal
    Args:
        a (Tensor): Coefficients of the all-pole filters, shape (B, T, N).
        x (Tensor): Input signal, shape (B, T).
        zi (Tensor, optional): Initial conditions for the filter, shape (B, N).
    Returns:
        Filtered output signal, shape (B, T), and optionally the final state of the filter.
    """
    assert a.dim() == 3, "Denominator coefficients a must be 3D."
    assert x.dim() == 2, "Input signal x must be 2D."
    B, T = x.shape
    assert (
        a.shape[:2] == x.shape
    ), "The first two dimensions of a must match the shape of x."

    if zi is None:
        return_zf = False
        zi = x.new_zeros((B, a.shape[2]))
    else:
        assert zi.dim() == 2, "Initial conditions zi must be 2D."
        assert zi.shape[0] == B, "The first dimension of zi must match the batch size."
        assert (
            zi.shape[1] == a.shape[2]
        ), "The second dimension of zi must match the filter order, but got {} instead of {}".format(
            zi.shape[1], a.shape[2]
        )

        return_zf = True

    return sample_wise_lpc(x, a, zi=zi, return_zf=return_zf)


def matrix_power_accumulate(A: Tensor, n: int) -> Tensor:
    """Compute the matrix power of A raised to n, accumulating the result.
    Args:
        A (Tensor): The input matrix, shape (*, N, N) where * can be any number of batch dimensions.
        n (int): The exponent to which the matrix A is raised.
        initial (Optional[Tensor]): Initial value for accumulation, shape (*, N, N).
    Returns:
        Tensor: The accumulated result after raising A to the power of n with shape (*, max(n, 1), N, N).
    """
    assert A.dim() >= 2, "Input matrix A must have at least 2 dimensions."
    assert A.shape[-2] == A.shape[-1], "Input matrix A must be square."
    assert n >= 0, "Exponent n must be non-negative."

    if n == 0:
        return (
            torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
            .broadcast_to(A.shape)
            .unsqueeze(-3)
        )

    # TODO: Use parallel scan
    return torch.stack(
        list(
            accumulate(
                [A] * n,
                torch.matmul,
            )
        ),
        dim=-3,
    )
