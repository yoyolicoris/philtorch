import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union

from .recur import linear_recurrence, LTIRecurrence
from .. import EXTENSION_LOADED


def _first_order_filt(
    x: Tensor, a: Tensor, zi: Tensor, b: Optional[Tensor] = None, **kwargs
) -> Tensor:
    xb = x if b is None else x * b
    if EXTENSION_LOADED:
        y = LTIRecurrence.apply(
            a.broadcast_to(x.shape[0]), zi.broadcast_to(x.shape[0]), xb
        )
    else:
        y = linear_recurrence(a.broadcast_to(x.shape[0]), zi, xb, **kwargs)
    return y


def _cubic_coeff(
    x: Tensor, parallel_form: bool, scipy_padding: bool, **kwargs
) -> Tensor:
    r = torch.tensor(3**0.5 - 2, device=x.device, dtype=x.dtype)
    # k_0 = min(14, x.shape[-1] - 1)

    if scipy_padding:
        # in scipy, a sequence [1, 2, 3] is mirrored to [3, 2, 1, 1, 2, 3, 3, 2, 1]
        powers = r ** torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
        causal_zi = x @ powers
    else:
        # while in torch reflect padding, [1, 2, 3] is padded to [2, 1, 2, 3, 2]
        powers = r ** torch.arange(x.shape[-1] - 1, device=x.device, dtype=x.dtype)
        causal_zi = x[..., 1:] @ powers

    if parallel_form:
        mirrored_x = torch.cat(
            [x, x.flip(-1) if scipy_padding else x[..., :-1].flip(-1)], dim=-1
        )

        h = _first_order_filt(mirrored_x, r, causal_zi, **kwargs)
        causal_h, anticausal_h = h[..., : x.shape[-1]], h[..., -x.shape[-1] :].flip(-1)
        c = -6 * r / (1 - r * r) * (causal_h + anticausal_h - x)
    else:
        # causal inverse filtering
        h = _first_order_filt(x, r, causal_zi, **kwargs)
        if scipy_padding:
            zi = r / (r - 1) * h[..., -1]
        else:
            zi = -r / (1 - r * r) * (2 * h[..., -1] - x[..., -1])

        # anticausal inverse filtering
        c_flip = _first_order_filt(h[..., :-1].flip(-1), r, zi, -r, **kwargs).flip(-1)
        c = torch.cat([c_flip, zi.unsqueeze(-1)], dim=-1) * 6
    return c


def _cubic_spline_kernel(x: Tensor) -> Tensor:
    abs_x = x.abs()
    mask1 = abs_x <= 1
    mask2 = ~mask1 & (abs_x < 2)
    return torch.where(
        mask1,
        (4 - 6 * abs_x**2 + 3 * abs_x**3) / 6,
        torch.where(
            mask2,
            (2 - abs_x) ** 3 / 6,
            0.0,
        ),
    )


def cspline(
    x: Tensor,
    parallel_form: bool = True,
    scipy_padding: bool = False,
    lamb: float = 0.0,
    **kwargs,
) -> Tensor:
    r"""
    Compute the coefficients for cubic spline interpolation of the input tensor `x` using the method described in "B-Spline Signal Processing: Part II: Efficient Design and Applications" by M. Unser.

    Args:
        x (Tensor): Input tensor of shape (B, L).
        parallel_form (bool): If True, use the partial fraction expansion form for cubic spline interpolation. If False, use cascaded form. Default is True.
        scipy_padding (bool): If True, use the same padding convention as `scipy.signal.cspline1d` (mirrored padding). If False, use PyTorch's reflect padding convention. Default is False.
        lamb (float): Smoothing coefficient. Current implementation only supports `lamb=0.0` (no smoothing). Default is 0.0.
        **kwargs: Additional keyword arguments passed to the underlying `linear_recurrence` function for inverse filtering.
    Returns:
        Tensor: Coefficients for cubic spline interpolation of shape (B, L).
    """
    if lamb != 0.0:
        raise NotImplementedError(
            "Regularization for cubic spline interpolation is not implemented."
        )
    return _cubic_coeff(x, parallel_form, scipy_padding, **kwargs)


def cubic_spline(x: Tensor, m: int, scipy_padding: bool = False, **kwargs) -> Tensor:
    r"""
    Upsample the input tensor `x` by a factor of `m` using cubic spline interpolation
    based on the method described in "B-Spline Signal Processing: Part II: Efficient Design
    and Applications" by M. Unser.

    Args:
        x (Tensor): Input tensor of shape (B, L).
        m (int): Interpolation factor (must be an integer >= 1).
        scipy_padding (bool): If True, use the same padding convention as `scipy.signal.cspline1d` (mirrored padding). If False, use PyTorch's reflect padding convention. Default is False.
        **kwargs: Additional keyword arguments passed to the underlying `cspline` function for coefficient computation.

    Returns:
        Tensor: Upsampled tensor of shape (B, (L - 1) * m + 1).
    """
    assert m >= 1 and isinstance(
        m, int
    ), "Interpolation factor m must be an integer >= 1."
    if m == 1:
        return x

    c = cspline(x, scipy_padding=scipy_padding, **kwargs)

    kernel_idx = torch.arange(-2, 2, 1 / m, device=x.device, dtype=x.dtype).reshape(
        4, m
    )
    kernel = _cubic_spline_kernel(kernel_idx).flip(0).T

    interped = F.conv1d(
        (
            torch.cat([c[:, :1], c, c[:, -2:].flip(-1)], dim=1).unsqueeze(1)
            if scipy_padding
            else F.pad(c.unsqueeze(1), (1, 2), mode="reflect")
        ),
        kernel.unsqueeze(1),
    ).mT.flatten(1, 2)
    return interped[..., : -(m - 1)]
