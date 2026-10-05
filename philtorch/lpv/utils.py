"""Helpers for time-varying filter structures."""

import torch.nn.functional as F
from torch import Tensor


def diag_shift(coef: Tensor, offset: int = 0, discard_end: bool = False) -> Tensor:
    r"""Delay each column of a coefficient tensor by its index.

    This returns ``out[..., n, k] = coef[..., n - k - offset, k]``, zero where
    the index is out of range, which turns direct-form coefficients into
    transposed-form ones.

    Args:
        coef (Tensor): coefficients, of shape :math:`(*, T, M)`.
        offset (int, optional): an extra delay for every column. Default: ``0``.
        discard_end (bool, optional): keep only the first :math:`T` steps if
            ``True``, or all :math:`T + M + \text{offset} - 1` if ``False``.
            Default: ``False``.

    Returns:
        Tensor: the shifted coefficients, of shape :math:`(*, T, M)` or
        :math:`(*, T + M + \text{offset} - 1, M)`.
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
