import torch
from torch import Tensor
from typing import Optional, Union, Tuple

from ..lpv import lfilter as lpv_lfilter


def lfilter(
    b: Tensor, a: Tensor, x: Tensor, zi: Optional[Tensor] = None, form: str = "df2"
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of time-invariant linear filters to input signal.
    Args:
        b (Tensor): Coefficients of the FIR filters, shape (B, N+1) or (N+1).
        a (Tensor): Coefficients of the all-pole filters, shape (B, N) or (N).
        x (Tensor): Input signal, shape (B, T) or (T).
        zi (Tensor, optional): Initial conditions for the filter, shape (B, N) or (N).
        form (str): The filter form to use. Options are 'df2', 'tdf2', 'df1', 'tdf1'.
    Returns:
        Filtered output signal with the same shape as x and optionally the final state of the filter.
    """

    squeeze_first = (
        (x.dim() == 1)
        & (b.dim() == 2)
        & (a.dim() == 2)
        & ((zi is None) or (zi.dim() == 1))
    )
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        raise ValueError("Input signal x must be 1D or 2D.")

    _, T = x.shape

    if b.dim() == 1:
        b = b.unsqueeze(0)
    elif b.dim() == 2:
        pass
    else:
        raise ValueError("Numerator coefficients b must be 1D or 2D.")
    if a.dim() == 1:
        a = a.unsqueeze(0)
    elif a.dim() == 2:
        pass
    else:
        raise ValueError("Denominator coefficients a must be 1D or 2D.")

    if b.shape[1] < a.shape[1] + 1:
        b = torch.cat(
            (b, b.new_zeros((b.shape[0], a.shape[1] + 1 - b.shape[1]))),
            dim=1,
        )
    elif b.shape[1] > a.shape[1] + 1:
        a = torch.cat(
            (a, a.new_zeros((a.shape[0], b.shape[1] - a.shape[1] - 1))),
            dim=1,
        )

    B = max(b.shape[0], a.shape[0], x.shape[0])
    broadcasted_b = b.expand(B, -1)
    broadcasted_a = a.expand(B, -1)
    broadcasted_x = x.expand(B, -1)

    # Use parameter-varying filter implementation, temporarily
    y = lpv_lfilter(
        broadcasted_b.unsqueeze(1).expand(-1, T, -1),
        broadcasted_a.unsqueeze(1).expand(-1, T, -1),
        broadcasted_x,
        zi=zi,
        form=form,
    )
    if isinstance(y, tuple):
        y, zf = y
        if squeeze_first:
            y = y.squeeze(0)
            zf = zf.squeeze(0)
        return y, zf
    return y.squeeze(0) if squeeze_first else y
