import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from functools import partial

from ..lpv import lfilter as lpv_lfilter
from .ssm import state_space, state_space_recursion
from ..prototype.utils import a2companion
from ..utils import chain_functions
from ..core import lti_fir


def lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Optional[Tensor] = None,
    form: str = "df2",
    backend: str = "ssm",
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of time-invariant linear filters to input signal.
    Args:
        b (Tensor): Coefficients of the FIR filters, shape (B, M+1) or (M+1).
        a (Tensor): Coefficients of the all-pole filters, shape (B, M) or (M).
        x (Tensor): Input signal, shape (B, N) or (N).
        zi (Tensor, optional): Initial conditions for the filter, shape (B, M) or (M).
        form (str): The filter form to use. Options are 'df2', 'tdf2', 'df1', 'tdf1'.
    Returns:
        Filtered output signal with the same shape as x and optionally the final state of the filter.
    """

    squeeze_first = (
        (x.dim() == 1)
        & (b.dim() == 1)
        & (a.dim() == 1)
        & ((zi is None) or (zi.dim() == 1))
    )
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        raise ValueError("Input signal x must be 1D or 2D.")

    assert b.dim() in (1, 2), "Numerator coefficients b must be 1D or 2D."
    assert a.dim() in (1, 2), "Denominator coefficients a must be 1D or 2D."

    if b.size(-1) < a.size(-1) + 1:
        b = F.pad(b, (0, a.size(-1) + 1 - b.size(-1)))
    elif b.shape[1] > a.shape[1] + 1:
        a = F.pad(a, (0, b.size(-1) - a.size(-1) - 1))

    match backend:
        case "ssm":
            y = _ssm_lfilter(b, a, x, zi, form=form, **kwargs)
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    if isinstance(y, tuple):
        y, zf = y
        if squeeze_first:
            y = y.squeeze(0)
            zf = zf.squeeze(0)
        return y, zf
    return y.squeeze(0) if squeeze_first else y


def _ssm_lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Optional[Tensor] = None,
    form: str = "df2",
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of time-invariant linear filters to input signal using state-space model."""
    A = a2companion(a)

    match form:
        case "df2":
            b0 = b[..., :1]  # First coefficient of the FIR filter
            C = b[..., 1:] - b0 * a
            D = b[..., 0]
            filt = partial(
                state_space,
                A,
                B=None,
                C=C,
                D=D,
                zi=zi,
                out_idx=None,
                **kwargs,
            )
        case "tdf2":
            b0 = b[..., :1]  # First coefficient of the FIR filter
            B = b[..., 1:] - b0 * a
            D = b[..., 0]
            filt = partial(
                state_space,
                A.mT.conj(),
                B=B.conj(),
                C=None,
                D=D.conj(),
                zi=zi,
                out_idx=0,
                **kwargs,
            )
        case "df1":
            zi = x.new_zeros((x.size(0), A.size(-1)))
            filt = chain_functions(
                partial(lti_fir, b),
                partial(state_space_recursion, A, zi, out_idx=0, **kwargs),
            )
        case "tdf1":
            zi = x.new_zeros((x.size(0), A.size(-1)))
            filt = chain_functions(
                partial(state_space_recursion, A.mT.conj(), zi, out_idx=0, **kwargs),
                partial(lti_fir, b.conj(), tranpose=True),
            )
        case _:
            raise ValueError(f"Unknown filter form: {form}")

    return filt(x)
