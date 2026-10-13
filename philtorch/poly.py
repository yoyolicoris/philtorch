"""Polynomial helpers for batched coefficient tensors."""

import torch
import torch.nn.functional as F
from torch import Tensor


def polydiv(u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    r"""Divide polynomials, returning the quotient and the remainder.

    This performs long division of :math:`u(x)` by :math:`v(x)` along the last
    dimension, finding the quotient :math:`q(x)` and the remainder :math:`r(x)`
    with

    .. math::
        u(x) = q(x) v(x) + r(x), \quad \deg r < \deg v.

    Coefficients run from the highest degree down, as in :func:`numpy.polydiv`.

    Args:
        u (Tensor): dividend coefficients, of shape :math:`(*, M + 1)`, where
            :math:`*` is zero or more batch dimensions.
        v (Tensor): divisor coefficients, of shape :math:`(*, N + 1)`, where
            :math:`N \le M`. Its batch dimensions must broadcast to those of
            :attr:`u`, and its leading coefficient ``v[..., 0]`` must be
            nonzero.

    Returns:
        tuple of Tensor: the quotient, of shape :math:`(*, M - N + 1)`, and the
        remainder, of shape :math:`(*, N)`.

    Raises:
        ValueError: if :attr:`u` or :attr:`v` is 0-D.
        RuntimeError: if :math:`N > M`, or if :attr:`u` has an integer dtype.

    Note:
        Unlike :func:`numpy.polydiv`, the remainder always has :math:`N`
        coefficients: leading zeros are kept, not trimmed.

    Example::

        >>> from philtorch.poly import polydiv
        >>> # x^3 - 3x^2 + 4 = (x - 1)(x^2 - 2x - 2) + 2
        >>> u = torch.tensor([1.0, -3.0, 0.0, 4.0])
        >>> v = torch.tensor([1.0, -1.0])
        >>> polydiv(u, v)
        (tensor([ 1., -2., -2.]), tensor([2.]))
    """
    if u.ndim < 1 or v.ndim < 1:
        raise ValueError(f"u and v must be at least 1D, got {tuple(u.shape)} and {tuple(v.shape)}")

    # w has the common type
    m = u.size(-1) - 1
    n = v.size(-1) - 1
    scale = v[..., 0].reciprocal()
    r = u
    q = []
    for k in range(0, m - n + 1):
        d = scale * r[..., k]
        q.append(d)
        # Out of place: the product above saved r[..., k] for its gradient.
        r = r - F.pad(d.unsqueeze(-1) * v, (k, m - n - k))

    r = r[..., m - n + 1 :]
    return torch.stack(q, dim=-1), r
