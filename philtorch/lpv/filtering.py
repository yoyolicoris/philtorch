import torch
from torch import Tensor
from typing import Optional, Union, Tuple
from functools import reduce, partial

from ..core import lpv_fir, lpv_allpole
from ..utils import chain_functions


def lfilter(
    b: Tensor, a: Tensor, x: Tensor, zi: Optional[Tensor] = None, form: str = "df2"
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of parameter-varying linear filters to input signal.
    Args:
        b (Tensor): Coefficients of the FIR filters, shape (B, T, N+1) or (T, N+1).
        a (Tensor): Coefficients of the all-pole filters, shape (B, T, N) or (T, N).
        x (Tensor): Input signal, shape (B, T) or (T).
        zi (Tensor, optional): Initial conditions for the filter, shape (B, N) or (N).
        form (str): The filter form to use. Options are 'df2', 'tdf2', 'df1', 'tdf1'.
    Returns:
        Filtered output signal with the same time steps as x and optionally the final state of the filter.
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

    if b.dim() == 2:
        b = b.unsqueeze(0)
    elif b.dim() == 3:
        pass
    else:
        raise ValueError("Numerator coefficients b must be 2D or 3D.")
    if a.dim() == 2:
        a = a.unsqueeze(0)
    elif a.dim() == 3:
        pass
    else:
        raise ValueError("Denominator coefficients a must be 2D or 3D.")

    assert (
        b.shape[1] == a.shape[1] == T
    ), "The number of time steps in b and a must match the input signal x."

    if b.shape[2] < a.shape[2] + 1:
        b = torch.cat(
            (b, b.new_zeros((b.size(0), b.shape[1], a.shape[2] + 1 - b.shape[2]))),
            dim=2,
        )
    elif b.shape[2] > a.shape[2] + 1:
        a = torch.cat(
            (a, a.new_zeros((a.size(0), a.shape[1], b.shape[2] - a.shape[2] - 1))),
            dim=2,
        )

    order = a.shape[2]

    B = max(b.size(0), a.size(0), x.size(0))

    return_zf = (zi is not None) and (form in ("df2", "tdf2"))
    if zi is None:
        zi = x.new_zeros((B, order))
    elif zi.dim() == 1:
        zi = zi.unsqueeze(0).expand(B, -1)
    elif zi.dim() == 2:
        assert zi.shape[1] == order, "Initial conditions zi must match filter order."
        B = max(B, zi.size(0))
        zi = zi.expand(B, -1)
    else:
        raise ValueError("Initial conditions zi must be 1D or 2D.")

    broadcasted_b = b.expand(B, -1, -1)
    broadcasted_a = a.expand(B, -1, -1)
    broadcasted_x = x.expand(B, -1)

    match form:
        case "df2":
            filt = chain_functions(
                partial(lpv_allpole, broadcasted_a, zi=zi),
                lambda x, _: x,
                partial(lpv_fir, broadcasted_b, zi=zi),
            )
        case "tdf2":
            raise NotImplementedError(
                "Transposed Direct Form II (tdf2) is not implemented yet."
            )
        case "df1":
            # In Direct Form I, the initial conditions are neglected.
            filt = chain_functions(
                partial(lpv_fir, broadcasted_b),
                partial(lpv_allpole, broadcasted_a),
            )
        case "tdf1":
            raise NotImplementedError(
                "Transposed Direct Form I (tdf1) is not implemented yet."
            )
        case _:
            raise ValueError(
                f"Unknown filter form: {form}. Supported forms are 'df2', 'tdf2', 'df1', 'tdf1'."
            )

    y = filt(broadcasted_x)
    if isinstance(y, tuple):
        y, zf = y
        if squeeze_first:
            y = y.squeeze(0)
            zf = zf.squeeze(0)
        return y if not return_zf else (y, zf)

    if squeeze_first:
        y = y.squeeze(0)
    return y
