"""Conversions between the forms of time-invariant filters.

The coefficients follow :mod:`scipy.signal`'s functions of the same names, with
the differences noted in each docstring.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ..mat import companion
from ..poly import _RTYPES, _reduce_groups, polydiv, polymul, unique_roots
from .filtering import _check_sos


def _tree_prod(polys: Tensor) -> Tensor:
    """Multiply the polynomials along dimension -2, in log2(P) batched products.

    ``polys`` is of shape (..., P, d + 1), P >= 1, in ascending powers (an odd
    P is padded with the polynomial 1), and the result of shape (..., P * d + 1).
    """
    P, d = polys.size(-2), polys.size(-1) - 1
    while polys.size(-2) > 1:
        if polys.size(-2) % 2:
            polys = torch.cat(
                [polys, F.pad(torch.ones_like(polys[..., :1, :1]), (0, polys.size(-1) - 1))], -2
            )
        polys = polymul(polys[..., 0::2, :], polys[..., 1::2, :])
    return polys[..., 0, : P * d + 1]


def _as_batch(**named: Tensor) -> tuple[list[Tensor], bool]:
    """Make 1-D and 2-D arguments 2-D with one shared batch size.

    Returns the arguments and whether any of them was batched.
    """
    for name, t in named.items():
        if t.dim() not in (1, 2):
            raise ValueError(f"{name} must be 1D or 2D.")
    sizes = {t.size(0) for t in named.values() if t.dim() == 2}
    if len(sizes) > 1:
        raise ValueError(
            f"Batched {', '.join(named)} must share a batch size, got {sorted(sizes)}."
        )
    B = sizes.pop() if sizes else 1
    tensors = [(t if t.dim() == 2 else t.unsqueeze(0)).expand(B, -1) for t in named.values()]
    return tensors, any(t.dim() == 2 for t in named.values())


def _expansion_matrix(p: Tensor, group: Tensor, power: Tensor) -> Tensor:
    r"""Return the matrix F of the partial fractions, of shape (B, M, M).

    ``p`` holds the M poles of each filter, repeated once per power, ``group``
    the index of each pole's group and ``power`` the power of its entry. Column
    i holds the coefficients of

    .. math::
        \frac{\prod_k (1 - p_k w)}{(1 - p_i w)^{j_i}}

    in ascending powers of w, which are the first M of the product with the
    entries of i's group of power up to j_i left out.
    """
    leave_out = (group.unsqueeze(-1) == group.unsqueeze(-2)) & (
        power.unsqueeze(-2) <= power.unsqueeze(-1)
    )
    linear = torch.where(leave_out, 0, -p.unsqueeze(-2))
    columns = _tree_prod(torch.stack([torch.ones_like(linear), linear], dim=-1))
    return columns[..., : p.size(-1)].transpose(-1, -2)


def sos2tf(sos: Tensor) -> tuple[Tensor, Tensor]:
    r"""Convert second-order sections to transfer function coefficients.

    This multiplies the sections' numerators and denominators,

    .. math::
        H(z) = \prod_{k=1}^{K}
            \frac{b_{k0} + b_{k1} z^{-1} + b_{k2} z^{-2}}
                 {1 + a_{k1} z^{-1} + a_{k2} z^{-2}}
            = \frac{b_0 + b_1 z^{-1} + \cdots + b_{2K} z^{-2K}}
                   {1 + a_1 z^{-1} + \cdots + a_{2K} z^{-2K}},

    by a tree of :func:`~philtorch.poly.polymul`: pairs of sections are
    multiplied, then pairs of their products, and so on, so that there are
    :math:`\lceil \log_2 K \rceil` levels, each one batched product. The
    products are exact, with no evaluation at points.

    Args:
        sos (Tensor): the sections, of shape :math:`(K, 5)` or
            :math:`(B, K, 5)`, :math:`K \ge 1`. Each row is
            :math:`b_0, b_1, b_2, a_1, a_2`, as in
            :func:`~philtorch.lti.sosfilt`.

    Returns:
        tuple of Tensor: the numerator :math:`b`, of shape :math:`(2K + 1)` or
        :math:`(B, 2K + 1)`, and the denominator :math:`a` without its leading
        1, of shape :math:`(2K)` or :math:`(B, 2K)`, as
        :func:`~philtorch.lti.lfilter` takes them, both in ascending powers of
        :math:`z^{-1}`.

    Raises:
        ValueError: if :attr:`sos` is not 2-D or 3-D, its last dimension is
            not 5, or it has no section.

    Note:
        Unlike :func:`scipy.signal.sos2tf`, the rows of :attr:`sos` omit the
        column :math:`a_0`, which is taken to be 1, and the returned
        denominator omits its leading 1.

    Example::

        >>> from philtorch.lti import sos2tf
        >>> # (1 + z^-1) / (1 - 0.5 z^-1), then (1 - z^-1) / (1 + 0.5 z^-1)
        >>> sos = torch.tensor([[1.0, 1.0, 0.0, -0.5, 0.0],
        ...                     [1.0, -1.0, 0.0, 0.5, 0.0]])
        >>> b, a = sos2tf(sos)
        >>> b
        tensor([ 1.,  0., -1.,  0.,  0.])
        >>> a
        tensor([ 0.0000, -0.2500,  0.0000,  0.0000])
    """
    _check_sos(sos)
    denominators = torch.cat([torch.ones_like(sos[..., :1]), sos[..., 3:]], dim=-1)
    both = _tree_prod(torch.stack([sos[..., :3], denominators], dim=-3))
    return both[..., 0, :], both[..., 1, 1:]


def residuez(
    b: Tensor, a: Tensor, tol: float = 1e-3, rtype: str = "avg"
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Expand a transfer function into partial fractions.

    For

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + \cdots + b_{L-1} z^{-(L-1)}}
                    {1 + a_1 z^{-1} + \cdots + a_M z^{-M}},

    this returns the residues :math:`r_i`, the poles :math:`p_i` and the direct
    term :math:`k` of

    .. math::
        H(z) = \sum_{i=1}^{M} \frac{r_i}{(1 - p_i z^{-1})^{j_i}}
            + \sum_n k_n z^{-n},

    where a pole of multiplicity :math:`m` is listed :math:`m` times in a row,
    with the powers :math:`j_i = 1, \dots, m`. The poles are the roots of the
    denominator, grouped by :func:`~philtorch.poly.unique_roots` and sorted by
    magnitude, ascending. The direct term :math:`k` is the quotient of the
    numerator by the denominator, in :math:`w = z^{-1}`, with the remainder
    :math:`\rho(w)` of degree below :math:`M` left. All the residues come from
    one linear system, :math:`F r = \rho`, where column :math:`i` of :math:`F`
    holds the coefficients of

    .. math::
        \frac{\prod_k (1 - p_k w)}{(1 - p_i w)^{j_i}},

    so that simple and repeated poles take the same path.

    Args:
        b (Tensor): numerator coefficients, of shape :math:`(L)` or
            :math:`(B, L)`.
        a (Tensor): denominator coefficients without the leading 1, of shape
            :math:`(M)` or :math:`(B, M)`, with :math:`a_M \ne 0`. Batched
            :attr:`b` and :attr:`a` have the same batch size.
        tol (float, optional): the distance within which two poles are
            considered equal, as in :func:`~philtorch.poly.unique_roots`.
            Default: ``1e-3``.
        rtype (str, optional): how to choose the pole of a group of equal poles,
            as in :func:`~philtorch.poly.unique_roots`. Default: ``"avg"``.

    Returns:
        tuple of Tensor: the residues :math:`r` and the poles :math:`p`, both
        complex and of shape :math:`(M)` or :math:`(B, M)`, and the direct term
        :math:`k`, of the dtype of :attr:`b` and :attr:`a`, and of shape
        :math:`(\max(L - M, 0))` or :math:`(B, \max(L - M, 0))`, empty when
        the numerator's degree is below the denominator's. The outputs are
        batched when :attr:`b` or :attr:`a` is.

    Raises:
        ValueError: if :attr:`b` or :attr:`a` is not 1-D or 2-D, their batch
            sizes differ, or :attr:`rtype` is unknown.

    Note:
        Unlike :func:`scipy.signal.residuez`, the denominator omits its leading
        1, and no zero coefficients are trimmed, so the shapes of the outputs
        follow from those of the inputs. A denominator with :math:`a_M = 0`
        has a pole at 0 that the division by it can't handle: the result is
        not finite.

    Note:
        The poles are the roots of :attr:`a`, which are ill-conditioned at high
        orders: as in SciPy, a Butterworth filter of order 12 already fails. For
        a filter given as sections, expanding it from the sections' own poles
        will be better than converting to :attr:`a` first.

    Example::

        >>> from philtorch.lti import residuez
        >>> # (1 + z^-1) / (1 - 0.25 z^-2)
        >>> #     = 1.5 / (1 - 0.5 z^-1) - 0.5 / (1 + 0.5 z^-1)
        >>> b = torch.tensor([1.0, 1.0])
        >>> a = torch.tensor([0.0, -0.25])
        >>> r, p, k = residuez(b, a)
        >>> r.real
        tensor([ 1.5000, -0.5000])
        >>> p.real
        tensor([ 0.5000, -0.5000])
    """
    if rtype not in _RTYPES:
        raise ValueError(f"rtype must be one of {_RTYPES}, got {rtype!r}")
    (b, a), batched = _as_batch(b=b, a=a)
    dtype = torch.promote_types(b.dtype, a.dtype)
    b, a = b.to(dtype), a.to(dtype)
    cdtype = torch.promote_types(dtype, torch.complex64)
    B, M = a.shape
    L = b.size(-1)

    if M == 0:
        r = p = b.new_empty(B, 0, dtype=cdtype)
        return (r, p, b) if batched else (r[0], p[0], b[0])

    poles = torch.linalg.eigvals(companion(a.to(cdtype)))
    unique, multiplicity, batch_index = unique_roots(poles, tol, rtype)
    # each row by ascending magnitude: sort by magnitude, then stably by row
    by_magnitude = unique.abs().argsort(stable=True)
    order = by_magnitude[batch_index[by_magnitude].argsort(stable=True)]
    unique, multiplicity = unique[order], multiplicity[order]
    # every row has M entries, one per power of each unique pole
    n = B * M
    first = multiplicity.cumsum(0) - multiplicity
    group = torch.arange(unique.numel(), device=a.device).repeat_interleave(
        multiplicity, output_size=n
    )
    power = (
        torch.arange(n, device=a.device) - first.repeat_interleave(multiplicity, output_size=n) + 1
    )
    entries = unique.repeat_interleave(multiplicity, output_size=n)

    # b(w) = k(w) a(w) + remainder(w), in descending powers of w, as SciPy does
    b_rev = b.flip(-1)
    if L > M:
        k_rev, remainder_rev = polydiv(
            b_rev, torch.cat([a.flip(-1), torch.ones_like(a[:, :1])], -1)
        )
        k = k_rev.flip(-1)
    else:
        remainder_rev = F.pad(b_rev, (M - L, 0))
        k = b[:, :0]
    remainder = remainder_rev.flip(-1).to(cdtype)

    columns = _expansion_matrix(entries.view(B, M), group.view(B, M), power.view(B, M))
    r = torch.linalg.solve(columns, remainder.unsqueeze(-1)).squeeze(-1)
    p = entries.view(B, M)
    return (r, p, k) if batched else (r[0], p[0], k[0])


def invresz(
    r: Tensor, p: Tensor, k: Tensor, tol: float = 1e-3, rtype: str = "avg"
) -> tuple[Tensor, Tensor]:
    r"""Convert a partial fraction expansion to transfer function coefficients.

    This is the inverse of :func:`residuez`: it multiplies out

    .. math::
        H(z) = \sum_{i=1}^{M} \frac{r_i}{(1 - p_i z^{-1})^{j_i}}
            + \sum_n k_n z^{-n},

    where equal poles are adjacent and have the powers :math:`j_i = 1, 2, \dots`
    in a row. The denominator is the product of :math:`(1 - p_i z^{-1})` over
    the poles, by a tree of :func:`~philtorch.poly.polymul`. The numerator is
    :math:`k` times the denominator plus the matrix :math:`F` of
    :func:`residuez` times :math:`r`.

    Args:
        r (Tensor): the residues, of shape :math:`(M)` or :math:`(B, M)`. A
            repeated pole's residues go by ascending power.
        p (Tensor): the poles, of the shape of :attr:`r`. Equal poles must be
            adjacent.
        k (Tensor): the direct term, of shape :math:`(M_k)` or
            :math:`(B, M_k)`, possibly empty. Batched :attr:`r`, :attr:`p` and
            :attr:`k` have the same batch size.
        tol (float, optional): the distance within which two adjacent poles
            are considered equal. Default: ``1e-3``.
        rtype (str, optional): how to choose the pole of a group of equal
            poles: ``"max"`` or ``"maximum"``, ``"min"`` or ``"minimum"``, or
            ``"avg"`` or ``"mean"``, as in :func:`~philtorch.poly.unique_roots`.
            Default: ``"avg"``.

    Returns:
        tuple of Tensor: the numerator :math:`b`, of shape :math:`(M + M_k)`
        or :math:`(B, M + M_k)`, and the denominator :math:`a` without its
        leading 1, of shape :math:`(M)` or :math:`(B, M)`, both complex and in
        ascending powers of :math:`z^{-1}`. The outputs are batched when any
        input is.

    Raises:
        ValueError: if :attr:`r`, :attr:`p` or :attr:`k` is not 1-D or 2-D,
            their batch sizes differ, :attr:`r` and :attr:`p` differ in
            shape, or :attr:`rtype` is unknown.

    Note:
        Unlike :func:`scipy.signal.invresz`, the denominator omits its leading
        1, and no zero coefficients are trimmed. A group of poles has the value
        that :attr:`rtype` chooses among its members, whereas SciPy reduces
        copies of the group's first pole, so that its ``rtype`` has no effect.

    Example::

        >>> from philtorch.lti import invresz
        >>> r = torch.tensor([1.5, -0.5])
        >>> p = torch.tensor([0.5, -0.5])
        >>> b, a = invresz(r, p, torch.zeros(0))
        >>> b.real
        tensor([1., 1.])
        >>> a.real
        tensor([ 0.0000, -0.2500])
    """
    if rtype not in _RTYPES:
        raise ValueError(f"rtype must be one of {_RTYPES}, got {rtype!r}")
    (r, p, k), batched = _as_batch(r=r, p=p, k=k)
    if r.size(-1) != p.size(-1):
        raise ValueError(
            f"r and p must have the same shape, got {tuple(r.shape)} and {tuple(p.shape)}."
        )
    cdtype = torch.promote_types(torch.promote_types(r.dtype, p.dtype), k.dtype)
    cdtype = torch.promote_types(cdtype, torch.complex64)
    r, p, k = r.to(cdtype), p.to(cdtype), k.to(cdtype)
    B, M = p.shape

    if M == 0:
        a = p
        return (k, a) if batched else (k[0], a[0])

    # Runs of adjacent poles within tol of the run's first pole
    index = torch.arange(M, device=p.device)
    starts = [torch.ones(B, dtype=torch.bool, device=p.device)]
    reference = p[:, 0]
    for i in range(1, M):
        new = ~((p[:, i] - reference).abs() <= tol)
        starts.append(new)
        reference = torch.where(new, p[:, i], reference)
    starts = torch.stack(starts, dim=-1)
    group = starts.cumsum(-1)
    power = index - torch.where(starts, index, 0).cummax(-1).values + 1
    poles = _reduce_groups(p, group.unsqueeze(-1) == group.unsqueeze(-2), rtype)

    denominator = _tree_prod(torch.stack([torch.ones_like(poles), -poles], dim=-1))
    numerator = F.pad(
        (_expansion_matrix(poles, group, power) @ r.unsqueeze(-1)).squeeze(-1), (0, k.size(-1))
    )
    if k.size(-1) > 0:
        numerator = numerator + polymul(k, denominator)
    a = denominator[:, 1:]
    return (numerator, a) if batched else (numerator[0], a[0])
