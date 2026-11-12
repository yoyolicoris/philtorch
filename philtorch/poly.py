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


def polymul(a: Tensor, b: Tensor) -> Tensor:
    r"""Multiply polynomials, returning the coefficients of the product.

    This computes the product :math:`c(x) = a(x) b(x)` along the last
    dimension, as the convolution of the coefficients, one grouped
    :func:`torch.nn.functional.conv1d`. The coefficients run from the highest
    degree down, as in :func:`numpy.polymul`; a convolution doesn't depend on
    the order, so ascending coefficients multiply the same way.

    Args:
        a (Tensor): coefficients of the first factor, of shape
            :math:`(*, M + 1)`, where :math:`*` is zero or more batch
            dimensions.
        b (Tensor): coefficients of the second factor, of shape
            :math:`(*, N + 1)`. Its batch dimensions must broadcast with those
            of :attr:`a`, and its dtype is that of :attr:`a`.

    Returns:
        Tensor: the coefficients of the product, of shape
        :math:`(*, M + N + 1)`.

    Raises:
        ValueError: if :attr:`a` or :attr:`b` is 0-D.
        RuntimeError: if :attr:`a` and :attr:`b` differ in dtype.

    Example::

        >>> from philtorch.poly import polymul
        >>> # (x + 1)(x - 2) = x^2 - x - 2
        >>> polymul(torch.tensor([1.0, 1.0]), torch.tensor([1.0, -2.0]))
        tensor([ 1., -1., -2.])
    """
    if a.ndim < 1 or b.ndim < 1:
        raise ValueError(f"a and b must be at least 1D, got {tuple(a.shape)} and {tuple(b.shape)}")

    # conv1d correlates, so the weights are flipped; the batch is one group each
    batch = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    n, m = a.size(-1), b.size(-1)
    x = a.expand(*batch, n).reshape(1, -1, n)
    w = b.expand(*batch, m).flip(-1).reshape(-1, 1, m)
    return F.conv1d(x, w, padding=m - 1, groups=w.size(0)).view(*batch, n + m - 1)
