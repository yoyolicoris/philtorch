import torch
from torch import Tensor
from typing import Optional, Union, Tuple
from functools import reduce, partial
from torchlpc import sample_wise_lpc
import torch.nn.functional as F

from ..utils import chain_functions
from ..mat import companion
from .ssm import (
    state_space,
    state_space_recursion,
    _ext_ss_recur,
    extension_backend_indicator,
)
from .utils import diag_shift


def fir(
    b: Tensor, x: Tensor, zi: Optional[Tensor] = None, transpose: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of parameter-varying FIR filters.

    This supports time-varying (parameter-varying) FIR coefficients where the
    coefficients can change at each time step. The tensors `b` and `x` must
    share their leading batch/time dimensions.

    Args:
        b (Tensor): Time-varying FIR coefficients with shape (B, N, M + 1).
        x (Tensor): Input signal with shape (B, N).
        zi (Tensor, optional): Initial conditions with shape (B, M).
        transpose (bool): If True, compute the transpose implementation.

    Returns:
        Filtered output and optionally final state.
    """
    assert b.dim() == 3, "Numerator coefficients b must be 3D."
    assert x.dim() == 2, "Input signal x must be 2D."

    B, T = x.shape
    assert (
        b.shape[:2] == x.shape
    ), "The first two dimensions of b must match the shape of x."

    if zi is None:
        return_zf = False
        zi = x.new_zeros((B, b.size(2) - 1))
    else:
        assert zi.dim() == 2, "Initial conditions zi must be 2D."
        assert zi.size(0) == B, "The first dimension of zi must match the batch size."
        assert (
            zi.size(1) == b.size(2) - 1
        ), "The second dimension of zi must match the filter order."

        return_zf = True

    if transpose:
        shifted_b = diag_shift(b, discard_end=not return_zf)
        y = torch.linalg.vecdot(
            shifted_b.flip(2),
            F.pad(
                x,
                (shifted_b.size(2) - 1, 0 if not return_zf else shifted_b.size(2) - 1),
            ).unfold(1, shifted_b.size(2), 1),
        )
        if return_zf:
            y, zf = torch.split_with_sizes(y, [T, b.size(2) - 1], 1)
            return (
                torch.cat(
                    [
                        y[..., : b.size(2) - 1] + zi,
                        y[..., b.size(2) - 1 :],
                    ],
                    dim=-1,
                ),
                zf,
            )
        return y

    unfolded_x = torch.cat([zi.flip(1), x], dim=1).unfold(1, b.size(2), 1)
    y = torch.linalg.vecdot(unfolded_x.conj(), b.flip(2))

    if return_zf:
        return y, unfolded_x[:, -1, 1:].flip(1)
    return y


def allpole(
    a: Tensor, x: Tensor, zi: Optional[Tensor] = None, transpose: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of parameter-varying all-pole filters.

    Args:
        a (Tensor): Time-varying denominator coefficients with shape (B, N, M).
        x (Tensor): Input signal with shape (B, N).
        zi (Tensor, optional): Initial conditions with shape (B, M).
        transpose (bool): If True, use the transposed implementation.

    Returns:
        Tensor or (Tensor, Tensor): Filtered output and optionally final state.
    """
    assert a.dim() == 3, "Denominator coefficients a must be 3D."
    assert x.dim() == 2, "Input signal x must be 2D."
    B, T = x.shape
    assert (
        a.shape[:2] == x.shape
    ), "The first two dimensions of a must match the shape of x."

    if zi is None:
        return_zf = False
        zi = x.new_zeros((B, a.size(2)))
    else:
        assert zi.dim() == 2, "Initial conditions zi must be 2D."
        assert zi.size(0) == B, "The first dimension of zi must match the batch size."
        assert zi.size(1) == a.size(
            2
        ), "The second dimension of zi must match the filter order, but got {} instead of {}".format(
            zi.size(1), a.size(2)
        )

        return_zf = True

    if transpose:
        a = diag_shift(a.conj(), offset=1, discard_end=not return_zf)
        x = torch.cat(
            [zi + x[:, : a.size(2)], x[:, a.size(2) :]]
            + ([torch.zeros_like(zi)] if return_zf else []),
            dim=1,
        )
        y = sample_wise_lpc(x, a)
        if return_zf:
            return torch.split_with_sizes(y, [T, a.size(2)], 1)
        return y

    return sample_wise_lpc(x, a, zi=zi, return_zf=return_zf)


def lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Optional[Tensor] = None,
    form: str = "df2",
    backend: str = "ssm",
    **kwargs: Optional[dict],
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of parameter-varying linear filters to input signal.
    Args:
        b (Tensor): Coefficients of the FIR filters, shape (B, N, M_b) or (N, M_b).
        a (Tensor): Coefficients of the all-pole filters, shape (B, N, M_a) or (N, M_a).
        x (Tensor): Input signal, shape (B, N) or (N).
        zi (Tensor, optional): Initial conditions for the filter, shape (B, max(M_a, M_b - 1)) or (max(M_a, M_b - 1)).
        form (str): The filter form to use. Options are 'df2', 'tdf2', 'df1', 'tdf1'.
        backend (str): The backend to use for filtering. Options are 'ssm', 'torchlpc'.
        **kwargs: Additional keyword arguments for the backend-specific filtering function.
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

    assert b.dim() in (2, 3), "Numerator coefficients b must be 2D or 3D."
    assert a.dim() in (2, 3), "Denominator coefficients a must be 2D or 3D."

    match backend:
        case "ssm":
            y = _ssm_lfilter(b, a, x, zi=zi, form=form, **kwargs)
        case "torchlpc":
            y = _torchlpc_lfilter(b, a, x, zi=zi, form=form)
        case _:
            raise ValueError(
                f"Unknown backend: {backend}. Supported backends are 'ssm', 'torchlpc'."
            )

    if isinstance(y, tuple):
        y, zf = y
        if squeeze_first:
            y = y.squeeze(0)
            zf = zf.squeeze(0)
        return y, zf

    if squeeze_first:
        y = y.squeeze(0)
    return y


def _torchlpc_lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Optional[Tensor] = None,
    form: str = "df2",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

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

    order = max(a.shape[2], b.shape[2] - 1)

    B = max(b.size(0), a.size(0), x.size(0))

    return_zf = (zi is not None) and (form in ("df2", "tdf2"))
    if zi is None:
        zi = x.new_zeros((B, order))
    elif zi.dim() == 1:
        assert zi.shape[0] == order, "Initial conditions zi must match filter order."
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
                partial(allpole, broadcasted_a, zi=zi[:, : a.shape[2]]),
                lambda x, a_zf: fir(broadcasted_b, x, zi=zi[:, : b.shape[2] - 1])
                + (a_zf,),
                lambda x, b_zf, a_zf: (
                    (
                        x,
                        b_zf if b_zf.size(1) > a_zf.size(1) else a_zf,
                    )
                    if return_zf
                    else x
                ),
            )
        case "tdf2":
            raise NotImplementedError(
                "Transposed Direct Form II (tdf2) is not implemented yet."
            )
        case "df1":
            # In Direct Form I, the initial conditions are neglected.
            filt = chain_functions(
                partial(fir, broadcasted_b),
                partial(allpole, broadcasted_a),
            )
        case "tdf1":
            # In Transposed Direct Form I, the initial conditions are neglected.
            filt = chain_functions(
                partial(
                    allpole,
                    broadcasted_a,
                    transpose=True,
                ),
                partial(fir, broadcasted_b, transpose=True),
            )
        case _:
            raise ValueError(
                f"Unknown filter form: {form}. Supported forms are 'df2', 'tdf2', 'df1', 'tdf1'."
            )

    return filt(broadcasted_x)


def _ssm_lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Optional[Tensor] = None,
    form: str = "df2",
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of time-invariant linear filters to input signal using state-space model."""
    if b.size(-1) < a.size(-1) + 1:
        b = F.pad(b, (0, a.size(-1) + 1 - b.size(-1)))
    elif b.size(-1) > a.size(-1) + 1:
        a = F.pad(a, (0, b.size(-1) - a.size(-1) - 1))

    A = companion(a)

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
                partial(fir, b.broadcast_to((x.size(0), -1, -1))),
                partial(
                    (
                        _ext_ss_recur
                        if extension_backend_indicator(x, A.size(-1))
                        else state_space_recursion
                    ),
                    A,
                    zi,
                    out_idx=0,
                    **kwargs,
                ),
            )
        case "tdf1":
            zi = x.new_zeros((x.size(0), A.size(-1)))
            filt = chain_functions(
                partial(
                    (
                        _ext_ss_recur
                        if extension_backend_indicator(x, A.size(-1))
                        else state_space_recursion
                    ),
                    A.mT.conj(),
                    zi,
                    out_idx=0,
                    **kwargs,
                ),
                partial(
                    fir, b.conj().broadcast_to((x.size(0), -1, -1)), transpose=True
                ),
            )
        case _:
            raise ValueError(f"Unknown filter form: {form}")

    return filt(x)
