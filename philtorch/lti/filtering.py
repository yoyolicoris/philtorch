import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from functools import partial

from .ssm import state_space, state_space_recursion, diag_state_space
from ..mat import companion
from ..utils import chain_functions
from ..poly import polydiv


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
    """Apply a batch of time-invariant FIR filters to input signal
    Args:
        b (Tensor): Coefficients of the FIR filters, shape (B, M+1).
        x (Tensor): Input signal, shape (B, N).
        zi (Tensor, optional): Initial conditions for the filter, shape (B, M).
    Returns:
        Filtered output signal, shape (B, N), and optionally the final state of the filter.
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

    match backend:
        case "ssm":
            y = _ssm_lfilter(b, a, x, zi, form=form, **kwargs)
        case "diag_ssm":
            y = _diag_ssm_lfilter(b, a, x, zi, form=form, **kwargs)
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
    """Apply a batch of time-invariant linear filters to input signal in both forward and backward directions.
    Args:
        b (Tensor): Coefficients of the FIR filters, shape (B, M+1) or (M+1).
        a (Tensor): Coefficients of the all-pole filters, shape (B, M) or (M).
        x (Tensor): Input signal, shape (B, N) or (N).
        padmode (str, optional): Padding mode to use. Default is 'replicate'.
        padlen (int, optional): Length of padding to apply to the input signal. If None, it will be set to 3 times the number of taps.
        method (str, optional): Method to use for filtering. Options are 'pad' or 'gust'. Default is 'pad'.
        irlen (int, optional): Length of the impulse response. If provided, it will be used to determine the padding length.
        form (str): The filter form to use. Options are 'df2', 'tdf2', 'df1', 'tdf1'.
        **kwargs: Additional keyword arguments to pass to the filtering function.
    Returns:
        Filtered output signal with the same shape as x and optionally the final state of the filter.
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
        x.size(1) > edge
    ), f"Input signal length {x.size(1)} must be greater than pad length {edge}."

    if edge > 0 and padmode is not None:
        ext = F.pad(
            x,
            (edge, edge),
            mode=padmode,
        )
    else:
        ext = x

    zi = lfilter_zi(a, b, transpose=(form == "tdf2"))
    x0 = ext[:, :1]

    y, _ = lfilter(b, a, ext, zi=zi * x0, form=form, **kwargs)
    y0 = y[:, -1:]

    y, _ = lfilter(b, a, y.flip(1), zi=zi * y0, form=form, **kwargs)

    if edge > 0:
        y = y[:, edge:-edge]

    return y.flip(1)
