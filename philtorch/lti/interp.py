import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union, Tuple

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


def cubic_spline(x: Tensor, m: int, parallel_form: bool = True, **kwargs) -> Tensor:
    r"""
    Upsample the input tensor `x` by a factor of `m` using cubic spline interpolation
    based on the method described in "B-Spline Signal Processing: Part II: Efficient Design
    and Applications" by M. Unser.

    Args:
        x (Tensor): Input tensor of shape (B, L).
        parallel_form (bool): If True, use the partial fraction expansion form for
            cubic spline interpolation. If False, use cascaded form. Default is True.
        m (int): Interpolation factor (must be an integer >= 1).
        **kwargs: Additional keyword arguments passed to the underlying ```linear_recurrence```
            function for inverse filtering.

    Returns:
        Tensor: Upsampled tensor of shape (B, (L - 1) * m + 1).
    """
    assert m >= 1 and isinstance(
        m, int
    ), "Interpolation factor m must be an integer >= 1."
    if m == 1:
        return x
    r = torch.tensor(3**0.5 - 2, device=x.device, dtype=x.dtype)
    k_0 = min(14, x.shape[-1] - 1)

    powers = r ** torch.arange(k_0, device=x.device, dtype=x.dtype)
    causal_zi = x[..., 1 : k_0 + 1] @ powers
    if parallel_form:
        mirrored_x = torch.cat([x, x.flip(-1)[..., 1:]], dim=-1)

        h = _first_order_filt(mirrored_x, r, causal_zi, **kwargs)
        causal_h, anticausal_h = h[..., : x.shape[-1]], h[..., -x.shape[-1] :].flip(-1)
        c = -6 * r / (1 - r * r) * (causal_h + anticausal_h - x)
    else:
        # causal inverse filtering
        h = _first_order_filt(x, r, causal_zi, **kwargs)
        zi = -r / (1 - r * r) * (2 * h[..., -1] - x[..., -1])

        # anticausal inverse filtering
        c_flip = _first_order_filt(h[..., :-1].flip(-1), r, zi, -r, **kwargs).flip(-1)
        c = torch.cat([c_flip, zi.unsqueeze(-1)], dim=-1) * 6

    kernel_idx = torch.arange(-2, 2, 1 / m, device=x.device, dtype=x.dtype).reshape(
        4, m
    )
    kernel = _cubic_spline_kernel(kernel_idx).flip(0).T

    interped = F.conv1d(
        F.pad(c.unsqueeze(1), (1, 2), mode="reflect"), kernel.unsqueeze(1)
    ).mT.flatten(1, 2)
    return interped[..., : -(m - 1)]
