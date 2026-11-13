"""Conversions between the forms of time-invariant filters.

The coefficients follow :mod:`scipy.signal`'s functions of the same names, with
the differences noted in each docstring.
"""

from functools import reduce

import torch
import torch.nn.functional as F
from torch import Tensor

from ..mat import companion
from ..poly import polymul
from .filtering import _check_sos

_RTYPES = ("max", "maximum", "min", "minimum", "avg", "mean")


def _tree_prod(polys: Tensor) -> Tensor:
    """Multiply the polynomials along dimension -2 in batched products.

    ``polys`` is of shape (..., P, d + 1), P >= 1, in ascending powers, and the
    result of shape (..., P * d + 1). There are ceil(log2 P) levels of products
    of pairs, an odd P padded with the polynomial 1.
    """

    def halve(polys: Tensor, _) -> Tensor:
        if polys.size(-2) % 2:
            one = F.pad(torch.ones_like(polys[..., :1, :1]), (0, polys.size(-1) - 1))
            polys = torch.cat([polys, one], -2)
        return polymul(polys[..., 0::2, :], polys[..., 1::2, :])

    P, d = polys.size(-2), polys.size(-1) - 1
    return reduce(halve, range((P - 1).bit_length()), polys)[..., 0, : P * d + 1]


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
    batched = bool(sizes)
    B = sizes.pop() if sizes else 1
    tensors = [t.unsqueeze(0).expand(B, -1) if t.dim() == 1 else t for t in named.values()]
    return tensors, batched


def _reduce_groups(p: Tensor, member: Tensor, rtype: str) -> Tensor:
    """Reduce each group of roots to one value.

    ``member`` of shape (..., M) marks the roots of each group among the M of
    ``p``, which broadcasts with it, and each group has a member. The result is
    of shape (...): the value of each group by ``rtype``.
    """
    if rtype in ("avg", "mean"):
        return (member * p).sum(-1) / member.sum(-1)
    # Compare by the real part, then by the imaginary part; a maximum is the
    # minimum of the negated parts.
    key = (lambda t: t) if rtype in ("min", "minimum") else torch.neg
    if p.is_complex():
        real = torch.where(member, key(p.real), torch.inf)
        candidate = member & (real == real.amin(-1, keepdim=True))
        index = torch.where(candidate, key(p.imag), torch.inf).argmin(-1)
    else:
        index = torch.where(member, key(p), torch.inf).argmin(-1)
    return torch.take_along_dim(p, index.unsqueeze(-1), -1).squeeze(-1)


def _vandermonde(p: Tensor, extra: Tensor, rows: int) -> Tensor:
    r"""Return the confluent Vandermonde matrix of shape (B, rows, M).

    ``p`` holds the M poles of each filter, repeated once per power, and
    ``extra`` the power minus 1 of each entry, e. Entry [n, i] is
    :math:`\binom{n + e_i}{e_i} p_i^n`, the impulse response of
    :math:`1 / (1 - p_i z^{-1})^{e_i + 1}`. A cumulative product builds it from
    the ratios :math:`p_i (n + e_i) / n` of consecutive rows, which keeps the
    binomials exact up to rounding, with no gamma functions.
    """
    n = torch.arange(1, rows, dtype=p.dtype.to_real(), device=p.device).unsqueeze(-1)
    ratios = p.unsqueeze(-2) * ((n + extra.unsqueeze(-2)) / n)
    return torch.cumprod(F.pad(ratios, (0, 0, 1, 0), value=1), dim=-2)


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
    both = _tree_prod(torch.stack((sos[..., :3], F.pad(sos[..., 3:], (1, 0), value=1)), -3))
    return both[..., 0, :], both[..., 1, 1:]


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

        >>> from philtorch.lti import unique_roots
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
    if M == 0:
        none = torch.empty(0, dtype=torch.int64, device=p.device)
        return (roots.new_empty(0), none, none) if batched else (roots.new_empty(0), none)
    close = (roots.unsqueeze(-1) - roots.unsqueeze(-2)).abs() <= tol

    # A root starts a group when no earlier group took it; the group takes the
    # roots that are close and not yet taken. For booleans, c > u is c & ~u.
    def take(carry: tuple[Tensor, tuple[Tensor, ...]], i_row: tuple[int, Tensor]):
        used, used_before = carry
        i, row = i_row
        was_used = used[:, i : i + 1]
        return used | (row > was_used), (*used_before, was_used)

    unused = torch.zeros(B, M, dtype=torch.bool, device=p.device)
    _, used_before = reduce(take, enumerate(close.unbind(1)), (unused, ()))
    leader = ~torch.cat(used_before, 1)
    # a root belongs to the first leader that is close to it
    owner = (close & leader.unsqueeze(-1)).to(torch.int8).argmax(1)
    batch_index, first = leader.nonzero(as_tuple=True)
    group = owner[batch_index] == first.unsqueeze(-1)
    unique = _reduce_groups(roots[batch_index], group, rtype)
    multiplicity = group.sum(-1)
    return (unique, multiplicity, batch_index) if batched else (unique, multiplicity)


def residuez(
    b: Tensor, a: Tensor, tol: float = 1e-3, rtype: str = "avg", delayed_form: bool = False
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Expand a transfer function into partial fractions.

    For

    .. math::
        H(z) = \frac{b_0 + b_1 z^{-1} + \cdots + b_{L-1} z^{-(L-1)}}
                    {1 + a_1 z^{-1} + \cdots + a_M z^{-M}},

    this returns the residues :math:`r_i`, the poles :math:`p_i` and the direct
    term :math:`k` of

    .. math::
        H(z) = \sum_{n=0}^{D-1} k_n z^{-n}
            + z^{-d} \sum_{i=1}^{M} \frac{r_i}{(1 - p_i z^{-1})^{j_i}},
        \quad D = \max(L - M, 0),

    with the fractions starting with the direct term, :math:`d = 0`, as in
    SciPy, or, with ``delayed_form=True``, delayed until it has ended,
    :math:`d = D`. The two agree when :math:`L \le M`. A pole of multiplicity
    :math:`m` is listed :math:`m` times in a row, with the powers
    :math:`j_i = 1, \dots, m`. The poles are the roots of the denominator,
    sorted by magnitude, ascending, then by angle in :math:`(-\pi, \pi]`, and
    grouped in that order by :func:`unique_roots`.

    Both parts come from the impulse response :math:`h` of :math:`H`, found as
    the first :math:`D + M` terms of :math:`b / a` by one triangular solve. The
    impulse response of each fraction is
    :math:`\binom{n + j_i - 1}{j_i - 1} p_i^n`, so that after the direct term,
    :math:`h[D + n] = \sum_i V_{D - d + n, i} r_i` with the confluent
    Vandermonde matrix :math:`V_{ni} = \binom{n + j_i - 1}{j_i - 1} p_i^n`, and
    one linear solve gives :math:`r`. The direct term is what the fractions
    leave of the response before,
    :math:`k_n = h[n] - \sum_i V_{n - d, i} r_i`, which is :math:`h[n]` itself
    in the delayed form. Simple and repeated poles take the same path.

    Args:
        b (Tensor): numerator coefficients, of shape :math:`(L)` or
            :math:`(B, L)`.
        a (Tensor): denominator coefficients without the leading 1, of shape
            :math:`(M)` or :math:`(B, M)`. Batched :attr:`b` and :attr:`a`
            have the same batch size.
        tol (float, optional): the distance within which two poles are
            considered equal, as in :func:`unique_roots`. Default: ``1e-3``.
        rtype (str, optional): how to choose the pole of a group of equal poles,
            as in :func:`unique_roots`. Default: ``"avg"``.
        delayed_form (bool, optional): if ``True``, delay the fractions until
            the direct term has ended, as :func:`lfilter`'s ``delayed_form``
            does. Default: ``False``.

    Returns:
        tuple of Tensor: the residues :math:`r` and the poles :math:`p`, both
        complex and of shape :math:`(M)` or :math:`(B, M)`, and the direct term
        :math:`k`, complex when :attr:`b` or :attr:`a` is, and of shape
        :math:`(\max(L - M, 0))` or :math:`(B, \max(L - M, 0))`, empty when
        the numerator's degree is below the denominator's. The outputs are
        batched when :attr:`b` or :attr:`a` is.

    Raises:
        ValueError: if :attr:`b` or :attr:`a` is not 1-D or 2-D, their batch
            sizes or precisions differ (real and complex inputs of one
            precision mix), :attr:`rtype` is unknown, or, without
            ``delayed_form``, :math:`L > M` and :math:`a_M = 0` for any filter.
        torch.linalg.LinAlgError: if the linear system is singular: a pole at 0
            is repeated, or, without ``delayed_form``, :math:`p_i^D`
            underflows.

    Note:
        Unlike :func:`scipy.signal.residuez`, the denominator omits its leading
        1, and no zero coefficients are trimmed, so the output shapes follow
        from the input shapes. The poles are grouped after they are sorted,
        not before, which differs only for a chain of poles each close to the
        next while the ends are not. Poles of equal magnitude, such as a
        conjugate pair, are ordered by angle, where SciPy keeps the order of
        the root finder, which can vary between platforms; magnitudes that
        differ only by rounding still can.

        A pole at 0 (:math:`a_M = 0`) has the constant fraction :math:`r_i`.
        When :math:`L > M`, that overlaps the direct term, so the expansion
        isn't unique: SciPy trims the pole from :math:`a`, changing the
        shapes, where this raises an error. In the delayed form, the pole is a
        tap at delay :math:`D`, which is unique.

    Note:
        Without ``delayed_form``, the residues come from the response after the
        direct term, from :math:`h[D]` on, where a pole contributes
        :math:`p_i^D`. If that underflows, as :math:`0.5^{1100}` does in
        float64, the linear system is singular, and the solve raises an error.
        The delayed form has no such limit.

    Note:
        The poles are the roots of :attr:`a`, which are ill-conditioned at high
        orders: for a Butterworth filter of order 12, the expansion's impulse
        response is off by about :math:`10^{-3}` in float64 (SciPy's by about
        100%). For a filter given as sections, expanding it from the sections'
        own poles will be better than converting to :attr:`a` first.

    Example::

        >>> from philtorch.lti import residuez
        >>> # (1 + z^-1) / (1 - 0.75 z^-1 + 0.125 z^-2)
        >>> #     = -5 / (1 - 0.25 z^-1) + 6 / (1 - 0.5 z^-1)
        >>> b = torch.tensor([1.0, 1.0])
        >>> a = torch.tensor([-0.75, 0.125])
        >>> r, p, k = residuez(b, a)
        >>> r.real
        tensor([-5.,  6.])
        >>> p.real
        tensor([0.2500, 0.5000])
    """
    if rtype not in _RTYPES:
        raise ValueError(f"rtype must be one of {_RTYPES}, got {rtype!r}")
    (b, a), batched = _as_batch(b=b, a=a)
    if b.dtype.to_real() != a.dtype.to_real():
        raise ValueError(f"b and a must share a precision, got {b.dtype} and {a.dtype}")
    # a real one is made complex, at its own precision, with a complex other
    dtype = b.dtype.to_complex() if b.is_complex() or a.is_complex() else b.dtype
    b, a = b.to(dtype), a.to(dtype)
    B, M = a.shape
    L = b.size(-1)

    if M == 0:
        r = p = b.new_empty(B, 0, dtype=b.dtype.to_complex())
        return (r, p, b) if batched else (r[0], p[0], b[0])
    if not delayed_form and L > M and (a[:, -1] == 0).any():
        raise ValueError(
            "A pole at 0 (a_M = 0) with a numerator longer than the denominator "
            "has no unique expansion; pass delayed_form=True."
        )

    poles = torch.linalg.eigvals(companion(a))
    # by ascending magnitude, then angle, in each row, so that the groups come
    # out in order and tied poles don't keep the solver's order
    poles = poles.gather(-1, poles.angle().argsort(dim=-1))
    poles = poles.gather(-1, poles.abs().argsort(dim=-1, stable=True))
    unique, multiplicity, _ = unique_roots(poles, tol, rtype)
    # every row has M entries, one per power of each unique pole
    n = B * M
    first = multiplicity.cumsum(0) - multiplicity
    extra = torch.arange(n, device=a.device) - first.repeat_interleave(multiplicity, output_size=n)
    p = unique.repeat_interleave(multiplicity, output_size=n).view(B, M)

    # The impulse response h = b / a: T h = b, T the lower Toeplitz matrix of
    # [1, a], whose unit diagonal the solver doesn't read.
    N = max(L, M)
    T = F.pad(a, (N, N - 1 - M)).unfold(-1, N, 1).flip(-1)
    h = torch.linalg.solve_triangular(
        T, F.pad(b, (0, N - L)).unsqueeze(-1), upper=False, unitriangular=True
    )
    h = h.squeeze(-1)

    # after the direct term, h[D:] = V r, V's rows starting at D - d
    D = N - M
    V = _vandermonde(p, extra.view(B, M), M if delayed_form else N)
    r = torch.linalg.solve(V[:, -M:], h[:, D:].to(p.dtype))
    k = h[:, :D]
    if not delayed_form and D > 0:
        k = torch.baddbmm(k.to(p.dtype).unsqueeze(-1), V[:, :D], r.unsqueeze(-1), alpha=-1)
        k = k.squeeze(-1) if b.is_complex() else k.squeeze(-1).real
    return (r, p, k) if batched else (r[0], p[0], k[0])


def invresz(
    r: Tensor,
    p: Tensor,
    k: Tensor,
    tol: float = 1e-3,
    rtype: str = "avg",
    delayed_form: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""Convert a partial fraction expansion to transfer function coefficients.

    This is the inverse of :func:`residuez`: it multiplies out

    .. math::
        H(z) = \sum_{n=0}^{M_k-1} k_n z^{-n}
            + z^{-d} \sum_{i=1}^{M} \frac{r_i}{(1 - p_i z^{-1})^{j_i}},

    with :math:`d = 0`, or :math:`d = M_k` with ``delayed_form=True``,
    where equal poles are adjacent and have the powers :math:`j_i = 1, 2, \dots`
    in a row. The denominator is the product of :math:`(1 - p_i z^{-1})` over
    the poles, by a tree of :func:`~philtorch.poly.polymul`. The numerator is
    :math:`k` times the denominator plus, delayed by :math:`d`, the fractions'
    numerator: the first :math:`M` coefficients of the denominator times their
    impulse response :math:`V r`, with the confluent Vandermonde matrix
    :math:`V` of :func:`residuez`.

    Args:
        r (Tensor): the residues, of shape :math:`(M)` or :math:`(B, M)`. A
            repeated pole's residues go by ascending power.
        p (Tensor): the poles, of shape :math:`(M)` or :math:`(B, M)`. Equal
            poles must be adjacent.
        k (Tensor): the direct term, of shape :math:`(M_k)` or
            :math:`(B, M_k)`, possibly empty. Batched :attr:`r`, :attr:`p` and
            :attr:`k` have the same batch size, and an unbatched one is shared
            by the batch.
        tol (float, optional): the distance within which two adjacent poles
            are considered equal. Default: ``1e-3``.
        rtype (str, optional): how to choose the pole of a group of equal
            poles: ``"max"`` or ``"maximum"``, ``"min"`` or ``"minimum"``, or
            ``"avg"`` or ``"mean"``, as in :func:`unique_roots`.
            Default: ``"avg"``.
        delayed_form (bool, optional): if ``True``, the fractions are delayed
            until the direct term has ended, as in :func:`residuez`.
            Default: ``False``.

    Returns:
        tuple of Tensor: the numerator :math:`b`, of shape :math:`(M + M_k)`
        or :math:`(B, M + M_k)`, and the denominator :math:`a` without its
        leading 1, of shape :math:`(M)` or :math:`(B, M)`, both complex and in
        ascending powers of :math:`z^{-1}`. The outputs are batched when any
        input is.

    Raises:
        ValueError: if :attr:`r`, :attr:`p` or :attr:`k` is not 1-D or 2-D,
            their batch sizes or precisions differ (real and complex inputs of
            one precision mix), :attr:`r` and :attr:`p` differ in length, or
            :attr:`rtype` is unknown.

    Note:
        Unlike :func:`scipy.signal.invresz`, the denominator omits its leading
        1, and no zero coefficients are trimmed. A group of poles has the value
        that :attr:`rtype` chooses among its members, whereas SciPy reduces
        copies of the group's first pole, so that its ``rtype`` has no effect.
        A pole continues a group when it is within :attr:`tol` of the previous
        pole, whereas SciPy compares it with the group's first pole. That is
        the same for repeated poles, but a chain in which each pole is close to
        the next while the ends are not is one group here.

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
        raise ValueError(f"r and p must have the same length, got {r.size(-1)} and {p.size(-1)}.")
    if len({t.dtype.to_real() for t in (r, p, k)}) > 1:
        raise ValueError(
            f"r, p and k must share a precision, got {r.dtype}, {p.dtype} and {k.dtype}"
        )
    # real inputs are made complex at their own precision
    cdtype = r.dtype.to_complex()
    r, p, k = r.to(cdtype), p.to(cdtype), k.to(cdtype)
    B, M = p.shape
    L = M + k.size(-1)

    if M == 0:
        return (k, p) if batched else (k[0], p[0])

    # a run of poles, each within tol of the one before; its powers 1, 2, ...
    starts = F.pad(torch.diff(p).abs() > tol, (1, 0), value=True)
    run = starts.cumsum(-1)
    index = torch.arange(M, device=p.device)
    extra = index - (starts * index).cummax(-1).values
    poles = _reduce_groups(p.unsqueeze(1), run.unsqueeze(-1) == run.unsqueeze(-2), rtype)

    denominator = _tree_prod(F.pad(-poles.unsqueeze(-1), (1, 0), value=1))
    # k a, then the fractions' numerator, from their first M samples, delayed by
    # d; a zero pads k, so that an empty k multiplies too
    fractions = (_vandermonde(poles, extra, M) @ r.unsqueeze(-1)).squeeze(-1)
    numerator = polymul(denominator, fractions)[..., :M]
    d = k.size(-1) if delayed_form else 0
    b = polymul(F.pad(k, (0, 1)), denominator)[..., :L] + F.pad(numerator, (d, L - M - d))
    a = denominator[:, 1:]
    return (b, a) if batched else (b[0], a[0])
