import torch
from torch import Tensor
import torch.nn.functional as F


def diag_shift(coef: Tensor, offset: int = 0, discard_end: bool = False) -> Tensor:
    assert coef.dim() >= 2, "Coefficient tensor must have at least 2 dimensions."
    *_, T, M = coef.shape

    padded_coef = F.pad(coef.mT, (offset, M))
    y = (
        padded_coef.flatten(-2, -1)
        .unflatten(-1, (T + M + offset, M))[..., :-1, :]
        .flatten(-2, -1)
        .unflatten(-1, (M, T + M + offset - 1))
    )
    if discard_end:
        y = y[..., : -(M - 1)]
    return y.mT
