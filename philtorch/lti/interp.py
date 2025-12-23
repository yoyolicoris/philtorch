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
    Cubic spline interpolation kernel.

    Args:
        x: Input tensor.
        m: Interpolation factor.

    Returns:
        Interpolated tensor.
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

        # causal_h = _first_order_filt(x, r.broadcast_to(x.shape[0]), causal_zi, **kwargs)
        # anticausal_zi = causal_h[..., -1]
        # anticausal_h = torch.cat(
        #     [
        #         _first_order_filt(
        #             x.flip(-1)[..., 1:],
        #             r.broadcast_to(x.shape[0]),
        #             anticausal_zi,
        #             **kwargs,
        #         ).flip(-1),
        #         anticausal_zi.unsqueeze(-1),
        #     ],
        #     dim=-1,
        # )
        # anticausal_zi = -r * x[..., -k_0 - 1 : -1] @ powers.flip(0)

        # causal_h, anticausal_h = _first_order_filt(
        #     torch.cat([x, x.flip(-1)], dim=0),
        #     r.broadcast_to(x.shape[0] * 2),
        #     torch.cat([causal_zi, anticausal_zi], dim=0),
        #     **kwargs,
        # ).chunk(2, dim=0)
        c = -6 * r / (1 - r * r) * (causal_h + anticausal_h - x)
    else:
        # causal inverse filtering
        h = _first_order_filt(x, r, causal_zi, **kwargs)
        zi = -r / (1 - r * r) * (2 * h[..., -1] - x[..., -1])
        c_flip = _first_order_filt(h[..., :-1].flip(-1), r, zi, -r, **kwargs).flip(-1)
        c = torch.cat([c_flip, zi.unsqueeze(-1)], dim=-1) * 6

        # zi = x[..., 1 : k_0 + 1] @ r ** torch.arange(
        #     k_0, device=x.device, dtype=x.dtype
        # )
        # h = _first_order_filt(x, r.broadcast_to(x.shape[0]), zi, **kwargs)

        # # anticausal inverse filtering
        # r2 = r * r
        # zi = (2 - r2) / (1 - r2) * h[..., -1] + r / (1 - r2) * h[..., -2]
        # c = (
        #     _first_order_filt(
        #         h.flip(-1), r.broadcast_to(x.shape[0]), zi, -r, **kwargs
        #     ).flip(-1)
        #     * 6
        # )

    kernel_idx = torch.arange(-2, 2, 1 / m, device=x.device, dtype=x.dtype).reshape(
        4, m
    )
    kernel = _cubic_spline_kernel(kernel_idx).flip(0).T

    interped = F.conv1d(
        F.pad(c.unsqueeze(1), (1, 2), mode="reflect"), kernel.unsqueeze(1)
    ).mT.flatten(1, 2)
    return interped[..., : -(m - 1)]
