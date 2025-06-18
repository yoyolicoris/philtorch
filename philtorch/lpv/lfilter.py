import torch
from torch import Tensor
from torchlpc import sample_wise_lpc
from typing import Optional, Union, Tuple
from functools import reduce, partial

from ..core import lpv_fir, lpv_allpole
from ..utils import chain_functions


def lfilter(
    b: Tensor, a: Tensor, x: Tensor, zi: Optional[Tensor] = None, form: str = "df2"
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    squeeze_first = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_first = True
    elif x.dim() > 2:
        raise ValueError("Input signal x must be 1D or 2D.")

    B, T = x.shape

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
            (b, b.new_zeros((b.shape[0], b.shape[1], a.shape[2] + 1 - b.shape[2]))),
            dim=2,
        )
    elif b.shape[2] > a.shape[2] + 1:
        a = torch.cat(
            (a, a.new_zeros((a.shape[0], a.shape[1], b.shape[2] - a.shape[2] - 1))),
            dim=2,
        )

    order = a.shape[2]
    broadcasted_b = b.expand(B, -1, -1)
    broadcasted_a = a.expand(B, -1, -1)

    return_zf = (zi is not None) and (form in ("df2", "tdf2"))
    if zi is None:
        zi = x.new_zeros((B, order + 1))
    elif zi.dim() == 1:
        zi = zi.unsqueeze(0).expand(B, -1)
    elif zi.dim() == 2:
        assert zi.shape[0] == B, "Initial conditions zi must match batch size B."
        assert zi.shape[1] == order, "Initial conditions zi must match filter order."
    else:
        raise ValueError("Initial conditions zi must be 1D or 2D.")

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

    y = filt(x)
    if isinstance(y, tuple):
        if squeeze_first:
            return y[0].squeeze(0), y[1].squeeze(0)
        return y

    return y.squeeze(0) if squeeze_first else y
