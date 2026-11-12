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
    dimension, as the convolution of the coefficients. They run from the
    highest degree down, as in :func:`numpy.polymul`; a convolution doesn't
    depend on the order, so ascending coefficients multiply the same way.

    Args:
        a (Tensor): coefficients of the first factor, of shape
            :math:`(*, M + 1)`, where :math:`*` is zero or more batch
            dimensions.
        b (Tensor): coefficients of the second factor, of shape
            :math:`(*, N + 1)`. Its batch dimensions must broadcast with those
            of :attr:`a`.

    Returns:
        Tensor: the coefficients of the product, of shape
        :math:`(*, M + N + 1)`.

    Raises:
        ValueError: if :attr:`a` or :attr:`b` is 0-D.

    Example::

        >>> from philtorch.poly import polymul
        >>> # (x + 1)(x - 2) = x^2 - x - 2
        >>> polymul(torch.tensor([1.0, 1.0]), torch.tensor([1.0, -2.0]))
        tensor([ 1., -1., -2.])
    """
    if a.ndim < 1 or b.ndim < 1:
        raise ValueError(f"a and b must be at least 1D, got {tuple(a.shape)} and {tuple(b.shape)}")

    m, n = a.size(-1), b.size(-1)
    outer = a.unsqueeze(-1) * b.unsqueeze(-2)
    # Row i of the outer product is shifted right by i: padding every row to
    # m + n and reading the flattened rows at width m + n - 1 moves row i by i.
    shifted = F.pad(outer, (0, m)).flatten(-2)[..., : m * (m + n - 1)]
    return shifted.unflatten(-1, (m, m + n - 1)).sum(-2)


_RTYPES = ("max", "maximum", "min", "minimum", "avg", "mean")


def _reduce_groups(p: Tensor, member: Tensor, rtype: str) -> Tensor:
    """Reduce the roots of each group to one value.

    ``p`` is of shape (B, M), ``member`` of shape (B, G, M) marks the roots of
    each of G groups, and the result is of shape (B, G): the value of each group
    by ``rtype``. A group without a member gets an arbitrary value.
    """
    if rtype in ("avg", "mean"):
        total = torch.where(member, p.unsqueeze(1), 0).sum(-1)
        return total / member.sum(-1).clamp(min=1)
    # Compare by the real part, then by the imaginary part; a maximum is the
    # minimum of the negated parts.
    sign = 1.0 if rtype in ("min", "minimum") else -1.0
    real = torch.where(member, sign * p.real.unsqueeze(1), torch.inf)
    candidate = member & (real == real.amin(-1, keepdim=True))
    if p.is_complex():
        imag = torch.where(candidate, sign * p.imag.unsqueeze(1), torch.inf)
        index = imag.argmin(-1)
    else:
        index = candidate.to(torch.int8).argmax(-1)
    return p.gather(-1, index)


def unique_roots(p: Tensor, tol: float = 1e-3, rtype: str = "min") -> tuple[Tensor, ...]:
    r"""Find the unique roots and their multiplicities.

    The roots are grouped greedily, in the order they are given: each root not
    yet in a group starts one with every later root within a distance
    :attr:`tol` of it that is not in a group yet. If :math:`a` is close to
    :math:`b` and :math:`b` to :math:`c`, :math:`a` and :math:`c` can still be
    farther apart than :attr:`tol`, so the grouping depends on the order. Each
    group gives one unique root, chosen by :attr:`rtype`.

    Args:
        p (Tensor): the roots, real or complex, of shape :math:`(M)` or
            :math:`(B, M)`.
        tol (float, optional): the distance within which two roots are
            considered equal. Default: ``1e-3``.
        rtype (str, optional): how to choose the root of a group: ``"max"`` or
            ``"maximum"``, ``"min"`` or ``"minimum"``, or ``"avg"`` or
            ``"mean"``. Complex roots are compared by their real parts, then
            by their imaginary parts. Default: ``"min"``.

    Returns:
        tuple of Tensor: the unique roots, of :attr:`p`'s dtype, and their
        multiplicities, of dtype ``torch.int64``, both of shape :math:`(U)`,
        where :math:`U` is the number of groups, and for a 2-D :attr:`p` also
        the batch index of each, of shape :math:`(U)` and dtype
        ``torch.int64``. The groups are in the order of their first roots, one
        row of :attr:`p` after the other, so the output is flat, not padded:
        use the batch indices in scatter operations such as
        :meth:`torch.Tensor.index_add`.

    Raises:
        ValueError: if :attr:`p` is not 1-D or 2-D, or :attr:`rtype` is
            unknown.

    Note:
        The number of groups depends on the values of :attr:`p`, so the output
        shapes do too, and the call can't be compiled without a graph break.

    Example::

        >>> from philtorch.poly import unique_roots
        >>> p = torch.tensor([0.0, 1.3, 1.31, 2.8, 1.25, 2.2, 10.3])
        >>> unique, multiplicity = unique_roots(p, tol=2e-2, rtype="avg")
        >>> unique[multiplicity > 1]
        tensor([1.3050])
    """
    if rtype not in _RTYPES:
        raise ValueError(f"rtype must be one of {_RTYPES}, got {rtype!r}")
    if p.ndim not in (1, 2):
        raise ValueError(f"p must be 1D or 2D, got {p.ndim}D")

    batched = p.ndim == 2
    roots = p if batched else p.unsqueeze(0)
    B, M = roots.shape
    close = (roots.unsqueeze(-1) - roots.unsqueeze(-2)).abs() <= tol
    used = torch.zeros(B, M, dtype=torch.bool, device=p.device)
    # group[b, i]: the roots of the group that root i starts, none if it is used
    group = torch.zeros_like(close)
    for i in range(M):
        group[:, i] = close[:, i] & ~used & ~used[:, i : i + 1]
        used = used | group[:, i]

    multiplicity = group.sum(-1)
    first = multiplicity > 0
    unique = _reduce_groups(roots, group, rtype)[first]
    multiplicity = multiplicity[first]
    if not batched:
        return unique, multiplicity
    batch_index = torch.arange(B, device=p.device).unsqueeze(-1).expand(B, M)[first]
    return unique, multiplicity, batch_index
