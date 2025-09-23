import torch
from torch import Tensor
import torch.nn.functional as F


def diag_shift(coef: Tensor, offset: int = 0, discard_end: bool = False) -> Tensor:
    """Shift coefficient tensor along a diagonal-like layout.

    This helper rearranges a (..., T, M) coefficient tensor into a shifted
    representation that is convenient for converting between direct and transpose-direct
    form.

    Args:
        coef (Tensor): Coefficient tensor with shape (..., T, M).
        offset (int): Additional offset applied to the diagonal shift.
        discard_end (bool): If True, discard trailing padded columns.

    Returns:
        Tensor: Shifted coefficient tensor with the same leading dimensions.
    """
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
        y = y[..., : -(M - 1 + offset)]
    return y.mT
