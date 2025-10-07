import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from functools import partial

from .ssm import state_space, state_space_recursion, diag_state_space
from ..mat import companion
from ..utils import chain_functions
from ..poly import polydiv
from .recur import linear_recurrence, LTIRecurrence
from .. import EXTENSION_LOADED


def comb_filter(
    a: Tensor, delay: int, x: Tensor, zi: Optional[Tensor] = None, **kwargs
) -> Tensor:
    """Apply a comb filter to the input signal.
    Args:
        a (Tensor): Coefficients of the all-pole filter, shape (B,) or (1,).
        delay (int): Delay of the comb filter.
        x (Tensor): Input signal, shape (B, N).
        zi (Tensor, optional): Initial conditions for the filter, shape (delay,) or (B, delay).
        **kwargs: Additional keyword arguments for `linear_recurrence`.
    Returns:
        Tensor: Filtered output signal, shape (B, N).
    """
    assert a.dim() <= 1, "Denominator coefficients a must be at most 1D."
    assert x.dim() == 2, "Input signal x must be 2D."
    assert delay >= 0, "Delay must be non-negative."
    if a.dim() == 1:
        assert a.size(0) == x.size(
            0
        ), "The first dimension of a must match the batch size of x."

    if delay == 1:
        return linear_recurrence(
            -a, torch.zeros_like(a) if zi is None else zi.squeeze(-1), x, **kwargs
        )

    remainder = x.size(1) % delay
    if remainder != 0:
        x = F.pad(x, (0, delay - remainder))

    folded_x = x.unflatten(1, (-1, delay)).mT
    if a.dim() == 1:
        a = a.repeat_interleave(delay)
    if zi is not None:
        return_zf = True
        if zi.dim() == 1:
            zi = zi.flip(0).repeat(x.size(0))
        else:
            zi = zi.flip(1).flatten()
    else:
        return_zf = False
        zi = torch.zeros_like(a)

    y = (
        (
            linear_recurrence(-a, zi, folded_x.flatten(0, 1), **kwargs)
            if EXTENSION_LOADED
            else LTIRecurrence.apply(-a, zi, folded_x.flatten(0, 1))
        )
        .unflatten(0, (-1, delay))
        .mT.flatten(1, 2)
    )
    if remainder != 0:
        y = y[:, : -(delay - remainder)]
    if return_zf:
        return y, y[:, -delay:].flip(1)
    return y


def lfiltic(b: Tensor, a: Tensor, y: Tensor, x: Optional[Tensor] = None) -> Tensor:
    """Compute the initial conditions for a linear filter given its coefficients and output.
    Args:
        b (Tensor): Coefficients of the FIR filter, shape (..., M+1).
        a (Tensor): Coefficients of the all-pole filter, shape (..., N).
        y (Tensor): Output signal, shape (..., N).
        x (Tensor, optional): Input signal, shape (..., M). If not provided,
            it will be initialized to zeros.
    Returns:
        Tensor: Initial conditions for the filter, shape (..., max(M, N)).
    """
    assert b.dim() >= 1, "Numerator coefficients b must be at least 1D."
    assert a.dim() >= 1, "Denominator coefficients a must be at least 1D."

    n = a.size(-1)
    m = b.size(-1) - 1
    k = max(n, m)

    if x is None:
        x = b.new_zeros(m)

    b_mat = F.pad(b[..., 1:], (0, m - 1), value=0.0).unfold(-1, m, 1)
    a_mat = F.pad(a, (0, n - 1), value=0.0).unfold(-1, n, 1)
    zi_b = (b_mat @ x.unsqueeze(-1)).squeeze(-1)
    zi_a = (a_mat @ y.unsqueeze(-1)).squeeze(-1)
    if zi_b.size(-1) < k:
        zi_b = F.pad(zi_b, (0, k - zi_b.size(-1)), value=0.0)
    if zi_a.size(-1) < k:
        zi_a = F.pad(zi_a, (0, k - zi_a.size(-1)), value=0.0)
    zi = zi_b - zi_a
    return zi


def lfilter_zi(a: Tensor, b: Optional[Tensor] = None, transpose: bool = True) -> Tensor:
    """Compute the initial conditions for a linear filter given its coefficients.
    Args:
        b (Tensor): Coefficients of the FIR filter, shape (..., M+1).
        a (Tensor): Coefficients of the all-pole filter, shape (..., M).
        transpose (bool): When set to `True`, the TDF-II form is used; otherwise it's DF-II. Default to `True`.
    Returns:
        Tensor: Initial conditions for the filter, shape (..., M).
    """
    assert a.dim() >= 1, "Denominator coefficients a must be at least 1D."

    if not transpose:
        n = a.size(-1) + 1
        A = companion(a)
        B = a.new_zeros(n - 1)
        B[0] = 1.0
    elif b is not None:
        assert b.dim() >= 1, "Numerator coefficients b must be at least 1D."
        n = max(b.size(-1), a.size(-1) + 1)
        if b.size(-1) < n:
            b = F.pad(b, (0, n - b.size(-1)), value=0.0)
        if a.size(-1) < n - 1:
            a = F.pad(a, (0, n - 1 - a.size(-1)), value=0.0)
        A = companion(a).mT.conj()
        B = b[..., 1:] - b[..., :1] * a
    else:
        raise ValueError(
            "Numerator coefficients b must be provided for transpose=True."
        )

    IminusA = torch.eye(n - 1, device=a.device, dtype=a.dtype) - A

    if IminusA.ndim == 2 and B.ndim > 1:
        IminusA = IminusA.expand(*B.shape[:-1], -1, -1)
    zi = torch.linalg.solve(IminusA, B)
    return zi


def fir(
    b: Tensor,
    x: Tensor,
    zi: Optional[Tensor] = None,
    transpose: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of time-invariant FIR filters.

    This function supports both direct (convolution) and transposed forms via
    the ``transpose`` flag.  When ``zi`` is provided the function will return
    final states as a second output.

    Args:
        b (Tensor): Filter coefficients of shape (B, M+1) where B is batch size.
        x (Tensor): Input signal of shape (B, N).
        zi (Tensor, optional): Initial conditions with shape (B, M).
        transpose (bool): If True, use the transposed convolution implementation.

    Returns:
        Filtered output of shape (B, N) and,
        if ``zi`` was provided, a second tensor containing the final state.
    """
    assert b.dim() == 2, "Numerator coefficients b must be 2D."
    assert x.dim() == 2, "Input signal x must be 2D."
    B, N = x.shape
    assert b.shape[0] == B, "The first dimension of b must match the batch size of x."
    M = b.size(1) - 1

    if zi is not None:
        assert zi.dim() == 2, "Initial conditions zi must be 2D."
        assert zi.size(0) == B, "The first dimension of zi must match the batch size."
        assert (
            zi.size(1) == M
        ), "The second dimension of zi must match the filter order."

    if transpose:
        y = F.conv_transpose1d(
            x.unsqueeze(0),
            b.unsqueeze(1).conj(),
            stride=1,
            groups=B,
        ).squeeze(0)
        if zi is not None:
            zf = y[:, -M:]
            y = y[:, :-M]
            y = torch.cat([zi + y[:, :M], y[:, M:]], dim=1)
            return y, zf
        return y[:, :-M]

    if zi is None:
        zf = None
        padded_x = F.pad(x, (M, 0))
    else:
        padded_x = torch.cat([zi.flip(1), x], dim=1)
        zf = padded_x[:, -M:].flip(1)

    y = F.conv1d(
        padded_x.unsqueeze(0),
        b.flip(1).unsqueeze(1),
        groups=B,
    ).squeeze(0)
    if zf is not None:
        return y, zf
    return y


def lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Optional[Tensor] = None,
    form: str = "tdf2",
    backend: str = "ssm",
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of time-invariant IIR filters.

    This is a convenience wrapper that dispatches to different backends
    (e.g. state-space) and supports several filter structures (direct/transpose
    forms). If all inputs are 1-D and zi matches that shape, the leading batch
    dimension is squeezed from the result.

    Args:
        b (Tensor): FIR coefficients with shape (B, M_b+1) or (M_b+1,).
        a (Tensor): IIR denominator coefficients with shape (B, M_a) or (M_a,).
        x (Tensor): Input signal with shape (B, N) or (N,).
        zi (Tensor, optional): Initial conditions with shape (B, M_{zi}) or (M_{zi},).
                            When `backend` = `ssm`, M_{zi} = `max(M_b, M_a)`.
                            When `backend` = `diag_ssm`, M_{zi} = M_a.
                            This parameter has no use when `form` is either `df1` or `tdf1`.
        form (str): Filter form, one of {'df2','tdf2','df1','tdf1'}. Default is `tdf2`.
        backend (str): Backend to execute the filter ('ssm', 'diag_ssm', ...). Default is `ssm`.

    Returns:
        Filtered output and optionally final state if `zi` is given and `form` is either `tdf2` or `df2`.
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

    match backend:
        case "ssm":
            y = _ssm_lfilter(b, a, x, zi, form=form, **kwargs)
        case "diag_ssm":
            y = _diag_ssm_lfilter(b, a, x, zi, form=form, **kwargs)
        case "fs":
            y = _fs_lfilter(b, a, x, zi, **kwargs)
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
                partial(fir, b.broadcast_to((x.size(0), -1))),
                partial(state_space_recursion, A, zi, out_idx=0, **kwargs),
            )
        case "tdf1":
            zi = x.new_zeros((x.size(0), A.size(-1)))
            filt = chain_functions(
                partial(state_space_recursion, A.mT.conj(), zi, out_idx=0, **kwargs),
                partial(fir, b.conj().broadcast_to((x.size(0), -1)), transpose=True),
            )
        case _:
            raise ValueError(f"Unknown filter form: {form}")

    return filt(x)


def _diag_ssm_lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Optional[Tensor] = None,
    form: str = "df2",
    delayed_form: bool = False,
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply a batch of time-invariant linear filters to input signal using state-space model."""

    if b.size(-1) > a.size(-1) + 1:
        zi = None
        if delayed_form:
            q, r = polydiv(b, F.pad(a, (1, 0), value=1.0))
            direct_filt = partial(fir, q)
            delay = b.size(-1) - a.size(-1)
            b = r
        else:
            rev_q, rev_r = polydiv(b.flip(-1), F.pad(a.flip(-1), (0, 1), value=1.0))
            direct_filt = partial(fir, rev_q.flip(-1))
            delay = 0
            b = rev_r.flip(-1)
    else:
        direct_filt = None
        delay = 0

    if b.size(-1) < a.size(-1) + 1:
        b = F.pad(b, (0, a.size(-1) + 1 - b.size(-1)))

    A = companion(a)

    match form:
        case "df2":
            b0 = b[..., :1]  # First coefficient of the FIR filter
            C = b[..., 1:] - b0 * a
            D = b[..., 0]
            filt = partial(
                diag_state_space,
                A=A,
                B=None,
                C=C,
                D=D,
                zi=zi,
                **kwargs,
            )
        case "tdf2":
            b0 = b[..., :1]  # First coefficient of the FIR filter
            B = b[..., 1:] - b0 * a
            D = b[..., 0]
            filt = partial(
                diag_state_space,
                A=A.mT.conj(),
                B=B.conj(),
                C=None,
                D=D.conj(),
                zi=zi,
                out_idx=0,
                **kwargs,
            )
        case _:
            raise ValueError(f"Unknown filter form: {form}")

    results = filt(x)
    if isinstance(results, tuple):
        return results
    y = results

    if delay > 0:
        y = F.pad(y[:, :-delay], (delay, 0))

    if direct_filt is not None:
        y = y + direct_filt(x)

    return y


def _fs_lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    zi: Optional[Tensor] = None,
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if b.size(-1) > a.size(-1) + 1:
        rev_q, rev_r = polydiv(b.flip(-1), F.pad(a.flip(-1), (0, 1), value=1.0))
        direct_filt = partial(fir, rev_q.flip(-1))
        b = rev_r.flip(-1)
    else:
        direct_filt = None

    if b.size(-1) < a.size(-1) + 1:
        b = F.pad(b, (0, a.size(-1) + 1 - b.size(-1)))

    C = b[..., 1:] - b[..., :1] * a
    D = b[..., :1]

    A = companion(a)
    Apower = torch.linalg.matrix_power(A, x.size(-1))
    # C = C - C @ Apower
    Chat = (C.unsqueeze(1) @ Apower).squeeze(1)

    B_transfer = torch.fft.rfft(F.pad(C - Chat, (1, 0), value=0.0), n=x.size(-1))
    A_transfer = torch.fft.rfft(F.pad(a, (1, 0), value=1.0), n=x.size(-1))

    # A_transfer = A_transfer.abs().clamp_min(1e-7) * torch.exp(1j * A_transfer.angle())

    # B_hat_transfer = torch.fft.rfft(F.pad(Chat, (1, 0), value=0.0), n=x.size(-1) * 2)
    # signs = torch.tensor([1.0, -1.0] * x.size(-1), device=x.device, dtype=x.dtype)[
    # : x.size(-1) + 1
    # ]
    # H = (B_transfer - B_hat_transfer * signs) / A_transfer
    # y = (
    #     torch.fft.irfft(torch.fft.rfft(x, n=x.size(-1) * 2) * H, n=x.size(-1) * 2)[
    #         ..., : x.size(-1)
    #     ]
    #     + D * x
    # )

    H = B_transfer / A_transfer
    h = torch.fft.irfft(H, n=x.size(-1))
    y = (
        torch.fft.irfft(
            torch.fft.rfft(x, n=x.size(-1) * 2 - 1)
            * torch.fft.rfft(h, n=x.size(-1) * 2 - 1),
            n=x.size(-1) * 2 - 1,
        )[..., : x.size(-1)]
        + D * x
    )

    if direct_filt is not None:
        y = y + direct_filt(x)
    return y


def filtfilt(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    padmode: Optional[str] = "replicate",
    padlen: Optional[int] = None,
    method: str = "pad",
    irlen: Optional[int] = None,
    form: str = "tdf2",
    **kwargs,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply zero-phase filtering by processing the input signal in both forward and backward directions.

    This function uses the `lfilter` function twice: first in the forward direction,
    then in the reverse direction, to achieve zero-phase distortion. Padding is applied
    to reduce edge effects.

    Args:
        b (Tensor): FIR coefficients with shape (B, M_b+1) or (M_b+1,).
        a (Tensor): IIR denominator coefficients with shape (B, M_a) or (M_a,).
        x (Tensor): Input signal with shape (B, N) or (N,).
        padmode (str, optional): Padding mode for the input signal. Default is 'replicate'.
        padlen (int, optional): Padding length. If None, set to 3 times the number of taps.
        method (str, optional): Filtering method, one of {'pad', 'gust'}. Default is 'pad'.
        irlen (int, optional): Impulse response length (unused). This parameter is copied from SciPy and will be implemented in the future.
        form (str, optional): Filter form, one of {'df2', 'tdf2', 'df1', 'tdf1'}. Default is 'tdf2'.
        **kwargs: Additional keyword arguments for `lfilter`.

    Returns:
        Tensor: Zero-phase filtered output with the same shape as x.
    """
    assert method in ("pad", "gust"), "Method must be either 'pad' or 'gust'."

    if method == "gust":
        raise NotImplementedError("Gustafsson's method is not implemented yet.")

    if padmode is None:
        padlen = 0

    ntaps = max(b.size(-1), a.size(-1) + 1)
    if padlen is None:
        edge = 3 * ntaps
    else:
        edge = padlen

    assert (
        x.size(-1) > edge
    ), f"Input signal length {x.size(-1)} must be greater than pad length {edge}."

    if edge > 0 and padmode is not None:
        ext = F.pad(
            x.view(-1, x.size(-1)),
            (edge, edge),
            mode=padmode,
        )
        if x.dim() == 1:
            ext = ext.squeeze(0)
    else:
        ext = x

    zi = lfilter_zi(a, b, transpose=(form == "tdf2"))
    x0 = ext[..., :1]

    y, _ = lfilter(b, a, ext, zi=zi * x0, form=form, **kwargs)
    y0 = y[..., -1:]

    y, _ = lfilter(b, a, y.flip(-1), zi=zi * y0, form=form, **kwargs)

    if edge > 0:
        y = y[..., edge:-edge]

    return y.flip(-1)
