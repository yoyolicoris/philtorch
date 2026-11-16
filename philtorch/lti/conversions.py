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


def _tree_prod(polys: Tensor, length: int | None = None) -> Tensor:
    """Multiply the polynomials along dimension -2 in batched products.

    ``polys`` is of shape (..., P, d + 1), P >= 1, in ascending powers, and the
    result of shape (..., P * d + 1), or its first ``length`` coefficients,
    truncated after each level, if given. There are ceil(log2 P) levels of
    products of pairs, an odd P padded with the polynomial 1.
    """

    def halve(polys: Tensor, _) -> Tensor:
        if polys.size(-2) % 2:
            one = F.pad(torch.ones_like(polys[..., :1, :1]), (0, polys.size(-1) - 1))
            polys = torch.cat([polys, one], -2)
        return polymul(polys[..., 0::2, :], polys[..., 1::2, :])[..., :length]

    P, d = polys.size(-2), polys.size(-1) - 1
    end = P * d + 1 if length is None else length
    return reduce(halve, range((P - 1).bit_length()), polys)[..., 0, :end]


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


def _reduce_groups(p: Tensor, label: Tensor, rtype: str) -> Tensor:
    """Give each root the value of its group.

    ``label`` holds the group of each root of ``p``, both of shape (B, M), as
    a position from 0 to M - 1. The result has ``p``'s shape: each root's group
    value by ``rtype``.
    """

    # The groups' values land at their labels, which they all read back.
    # Workaround of an upstream bug: with include_self=False, the backward of
    # "amin" and "amax" still splits the gradient with the base wherever the
    # base equals the result, so the base is NaN, which equals nothing.
    def by_group(t: Tensor, reduce: str) -> Tensor:
        base = torch.full_like(t, torch.nan)
        return base.scatter_reduce(-1, label, t, reduce, include_self=False).gather(-1, label)

    if rtype in ("avg", "mean"):
        # CUDA's scatter_reduce takes no complex tensors
        if p.is_complex():
            return torch.complex(by_group(p.real, "mean"), by_group(p.imag, "mean"))
        return by_group(p, "mean")
    reduce = "amin" if rtype in ("min", "minimum") else "amax"
    if not p.is_complex():
        return by_group(p, reduce)
    # Compare by the real part, then by the imaginary parts of the roots of
    # that real part.
    real = by_group(p.real, reduce)
    tied = torch.where(p.real == real, p.imag, torch.inf if reduce == "amin" else -torch.inf)
    return torch.complex(real, by_group(tied, reduce))


def _group_roots(roots: Tensor, tol: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Group the roots within ``tol`` of each other, directly or through a
    chain of such roots.

    ``roots`` is of shape (B, M), M >= 1. Returns the group of each root, the
    position of its first root, of shape (B, M), and the batch index, the
    position of the first root and the number of roots of each group, of
    shape (G), the groups in row order.
    """
    real = roots.dtype.to_real()
    near = ((roots.unsqueeze(-1) - roots.unsqueeze(-2)).abs() <= tol).to(real)
    # each squaring doubles the length of the chains linked
    steps = range((roots.size(-1) - 1).bit_length())
    same = reduce(lambda c, _: (c @ c > 0).to(real), steps, near) > 0
    label = same.to(torch.int8).argmax(-1)
    M = roots.size(-1)
    rows = torch.arange(roots.size(0), device=roots.device).unsqueeze(-1)
    group, multiplicity = (rows * M + label).unique(return_counts=True)
    return label, group // M, group % M, multiplicity


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
        H(z) = \prod_{l=1}^{K}
            \frac{b_{l0} + b_{l1} z^{-1} + b_{l2} z^{-2}}
                 {1 + a_{l1} z^{-1} + a_{l2} z^{-2}}
            = \frac{b_0 + b_1 z^{-1} + \cdots + b_{2K} z^{-2K}}
                   {1 + a_1 z^{-1} + \cdots + a_{2K} z^{-2K}}.

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

    Roots within a distance :attr:`tol` of each other, directly or through a
    chain of such roots, are one group, whatever their order. Each group gives
    one unique root, chosen by :attr:`rtype`.

    Args:
        p (Tensor): the roots, floating-point or complex, of shape
            :math:`(M)` or :math:`(B, M)`.
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
        ValueError: if :attr:`p` is not 1-D or 2-D, or neither floating-point
            nor complex, or :attr:`rtype` is unknown.

    Note:
        Unlike :func:`scipy.signal.unique_roots`, which groups the roots
        greedily in their order, a chain of roots, each within :attr:`tol` of
        the next, is one group even when its ends are farther apart.

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
    if not (p.is_floating_point() or p.is_complex()):
        raise ValueError(f"p must be floating-point or complex, got {p.dtype}")

    batched = p.ndim == 2
    roots = p if batched else p.unsqueeze(0)
    if roots.size(-1) == 0:
        none = torch.empty(0, dtype=torch.int64, device=p.device)
        return (roots.new_empty(0), none, none) if batched else (roots.new_empty(0), none)
    label, batch_index, first, multiplicity = _group_roots(roots, tol)
    unique = _reduce_groups(roots, label, rtype)[batch_index, first]
    return (unique, multiplicity, batch_index) if batched else (unique, multiplicity)


def residuez(
    b: Tensor, a: Tensor, tol: float = 1e-3, rtype: str = "avg", delayed_form: bool = False
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Expand a transfer function into partial fractions.

    For

    .. math::
        H(z) = \frac{B(z)}{A(z)}
             = \frac{b_0 + b_1 z^{-1} + \cdots + b_{L-1} z^{-(L-1)}}
                    {1 + a_1 z^{-1} + \cdots + a_M z^{-M}},

    this returns the residues :math:`r_i`, the poles :math:`p_i` and the direct
    term :math:`k` of

    .. math::
        H(z) = F(z) + \sum_{i=1}^{M} \frac{r_i}{(1 - p_i z^{-1})^{j_i}},

    as SciPy does, or, with ``delayed_form=True``, of

    .. math::
        H(z) = F(z) + z^{-D} \sum_{i=1}^{M} \frac{r_i}{(1 - p_i z^{-1})^{j_i}},

    where :math:`F(z) = k_0 + k_1 z^{-1} + \cdots + k_{D-1} z^{-(D-1)}` is the
    FIR part, :math:`D = \max(L - M, 0)`. The two agree when :math:`L \le M`.
    A pole of multiplicity
    :math:`m` is listed :math:`m` times in a row, with the powers
    :math:`j_i = 1, \dots, m`. The poles are the roots of the denominator,
    sorted by magnitude, ascending, then by angle in :math:`(-\pi, \pi]`, and
    grouped by :func:`unique_roots`.

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
        torch.linalg.LinAlgError: if a pole at 0 is repeated, or, without
            ``delayed_form``, :math:`p_i^D` underflows for a pole, which
            ``delayed_form=True`` avoids.

    Note:
        Unlike :func:`scipy.signal.residuez`, no zero coefficients are
        trimmed, so the output shapes follow from the input shapes: where
        SciPy trims a pole at 0 from :attr:`a` when :math:`L > M`, this raises
        an error, unless ``delayed_form=True``.

        The poles are the roots of :attr:`a`, which are ill-conditioned at high
        orders, and so is the expansion. For a filter given as sections, use
        :func:`sos2pfe`, whose poles are the sections' own.

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
        H(z) = F(z) + \sum_{i=1}^{M} \frac{r_i}{(1 - p_i z^{-1})^{j_i}},

    or, with ``delayed_form=True``,

    .. math::
        H(z) = F(z) + z^{-D} \sum_{i=1}^{M} \frac{r_i}{(1 - p_i z^{-1})^{j_i}},

    where :math:`F(z) = k_0 + k_1 z^{-1} + \cdots + k_{D-1} z^{-(D-1)}` is the
    FIR part, and equal poles are adjacent, with the powers
    :math:`j_i = 1, 2, \dots` in a row.

    Args:
        r (Tensor): the residues, of shape :math:`(M)` or :math:`(B, M)`. A
            repeated pole's residues go by ascending power.
        p (Tensor): the poles, of shape :math:`(M)` or :math:`(B, M)`. Equal
            poles must be adjacent.
        k (Tensor): the direct term, of shape :math:`(D)` or
            :math:`(B, D)`, possibly empty. Batched :attr:`r`, :attr:`p` and
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
        tuple of Tensor: the numerator :math:`b`, of shape :math:`(M + D)`
        or :math:`(B, M + D)`, and the denominator :math:`a` without its
        leading 1, of shape :math:`(M)` or :math:`(B, M)`, both complex and in
        ascending powers of :math:`z^{-1}`. The outputs are batched when any
        input is.

    Raises:
        ValueError: if :attr:`r`, :attr:`p` or :attr:`k` is not 1-D or 2-D,
            their batch sizes or precisions differ (real and complex inputs of
            one precision mix), :attr:`r` and :attr:`p` differ in length, or
            :attr:`rtype` is unknown.

    Note:
        Unlike :func:`scipy.signal.invresz`, no zero coefficients are
        trimmed.

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
    index = torch.arange(M, device=p.device)
    first = (starts * index).cummax(-1).values
    extra = index - first
    poles = _reduce_groups(p, first, rtype)

    denominator = _tree_prod(F.pad(-poles.unsqueeze(-1), (1, 0), value=1))
    # k a, then the fractions' numerator, from their first M samples, delayed by
    # d; a zero pads k, so that an empty k multiplies too
    fractions = (_vandermonde(poles, extra, M) @ r.unsqueeze(-1)).squeeze(-1)
    numerator = polymul(denominator, fractions)[..., :M]
    d = k.size(-1) if delayed_form else 0
    b = polymul(F.pad(k, (0, 1)), denominator)[..., :L] + F.pad(numerator, (d, L - M - d))
    a = denominator[:, 1:]
    return (b, a) if batched else (b[0], a[0])


def sos2pfe(
    sos: Tensor, tol: float | None = None, rtype: str = "avg"
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
    r"""Expand second-order sections into partial fractions.

    For the cascade

    .. math::
        H(z) = \prod_{l=1}^{K}
            \frac{b_{l0} + b_{l1} z^{-1} + b_{l2} z^{-2}}
                 {1 + a_{l1} z^{-1} + a_{l2} z^{-2}},

    this returns the residues :math:`r_i`, the poles :math:`p_i`, the direct
    gain :math:`g = \prod_l b_{l0}` and the powers :math:`j_i` of the delayed
    form

    .. math::
        H(z) = g + \sum_{i=1}^{M}
            \frac{r_i z^{-j_i}}{(1 - p_i z^{-1})^{j_i}},

    as :func:`~philtorch.lti.pfefilt` takes it. A pole of multiplicity
    :math:`m` is listed :math:`m` times in a row, with the powers
    :math:`j_i = 1, \dots, m`. Each term is delayed by its power, so that the
    powers of a pole at 0 are the taps :math:`r_i z^{-j_i}` of an FIR filter.
    For simple poles, this is the delayed form of the partial fraction
    expansion in `Introduction to Digital Filters with Audio Applications`_
    (Smith, 2007), with the FIR part :math:`F(z) = g`. A real :attr:`sos`
    lists one pole of each conjugate pair, that of positive imaginary part,
    which stands for both in :func:`~philtorch.lti.pfefilt`, and its real
    poles have real residues. A complex :attr:`sos` lists every pole.

    Poles within :attr:`tol` of each other, directly or through a chain of
    such poles, are one group.

    Args:
        sos (Tensor): the sections, of shape :math:`(K, 5)` or
            :math:`(B, K, 5)`, :math:`K \ge 1`. Each row is
            :math:`b_0, b_1, b_2, a_1, a_2`, as in
            :func:`~philtorch.lti.sosfilt`.
        tol (float, optional): the distance within which two poles are
            considered equal. ``None`` means :math:`10 \sqrt{\epsilon}`,
            :math:`\epsilon` being the machine epsilon of :attr:`sos`'s
            dtype, enough to merge a double root that rounding splits.
            Default: ``None``.
        rtype (str, optional): how to choose the pole of a group of equal poles,
            as in :func:`unique_roots`, up to the conjugate symmetry of a real
            :attr:`sos`. Default: ``"avg"``.

    Returns:
        tuple: the residues :math:`r` and the poles :math:`p`, both complex
        and of shape :math:`(M)`, :math:`M` being the number of terms of all
        the filters: at most :math:`2K` per filter, fewer for a real
        :attr:`sos`; the direct gain :math:`g`, of :attr:`sos`'s
        dtype and of shape :math:`()` or :math:`(B)`; the batch index, the row
        of the filter each term belongs to, of shape :math:`(M)`, or ``None``
        for a 2-D :attr:`sos`; and the powers :math:`j_i`, of shape
        :math:`(M)`. The terms are in the order of the rows, and in a row,
        grouped in the order of their first pole, the sections' poles one
        section after the other; each entry of a group holds the group's pole.

    Raises:
        ValueError: if :attr:`sos` is not 2-D or 3-D, its last dimension is
            not 5, it has no section, or :attr:`rtype` is unknown.

    Note:
        For simple poles, the expansion is that of :func:`residuez` with
        ``delayed_form=True`` on :func:`sos2tf`'s output, up to the order of
        the poles and, for a real :attr:`sos`, the conjugates it leaves out;
        :func:`residuez` can't hold a repeated pole at 0, as an FIR
        section has.

    Example::

        >>> from philtorch.lti import sos2pfe
        >>> # (1 + z^-1) / (1 - 0.5 z^-1) = 1 + 1.5 z^-1 / (1 - 0.5 z^-1),
        >>> # and a pole at 0 with residue 0
        >>> sos = torch.tensor([[1.0, 1.0, 0.0, -0.5, 0.0]])
        >>> r, p, g, batch_index, power = sos2pfe(sos)
        >>> r.real
        tensor([1.5000, 0.0000])
        >>> p.real
        tensor([0.5000, 0.0000])
        >>> g
        tensor(1.)
        >>> power
        tensor([1, 1])
        >>> # a filter of order 24, 12 sections of unit gain at DC: the
        >>> # expansion filters as the cascade does
        >>> from philtorch.lti import pfefilt, sosfilt
        >>> a1 = -1.9 * torch.linspace(0.05, 0.3, 12, dtype=torch.float64).cos()
        >>> a2 = torch.full_like(a1, 0.9025)
        >>> b = (1 + a1 + a2) / 4
        >>> sos = torch.stack([b, 2 * b, b, a1, a2], -1)
        >>> x = torch.zeros(200, dtype=torch.float64)
        >>> x[0] = 1.0
        >>> r, p, g, batch_index, power = sos2pfe(sos)
        >>> y = pfefilt(r, p, g, x, batch_index=batch_index, power=power)
        >>> torch.allclose(y, sosfilt(sos, x))
        True

    .. _Introduction to Digital Filters with Audio Applications:
        https://ccrma.stanford.edu/~jos/filters/
    """
    if rtype not in _RTYPES:
        raise ValueError(f"rtype must be one of {_RTYPES}, got {rtype!r}")
    _check_sos(sos)
    if tol is None:
        tol = 10 * torch.finfo(sos.dtype).eps ** 0.5
    batched = sos.dim() == 3
    sos = sos if batched else sos.unsqueeze(0)
    B, K, _ = sos.shape
    if B == 0:
        none = torch.empty(0, dtype=torch.long, device=sos.device)
        empty = sos.new_empty(0, dtype=sos.dtype.to_complex())
        return empty, empty, sos.new_empty(0), none, none

    # the roots of z^2 + a1 z + a2: q is the one without cancellation, and the
    # other is a2 / q (q is 0 only where a1 = a2 = 0), or, for a real section,
    # q's conjugate where the roots are conjugates
    a1, a2 = sos[..., 3:].to(sos.dtype.to_complex()).unbind(-1)
    disc = a1 * a1 - 4 * a2
    d = torch.sqrt(disc)
    d = torch.where((a1.conj() * d).real < 0, -d, d)
    q = -(a1 + d) / 2
    # by the discriminant's sign: sqrt's real part isn't exactly 0 on CUDA
    conjugates = (disc.real < 0) & (not sos.is_complex())
    poles = torch.stack((q, torch.where(conjugates, q.conj(), a2 / (q + (q == 0)))), -1)
    poles = poles.view(B, 2 * K)

    # the closure maps the group of p to that of conj(p)
    label, batch_index, first, multiplicity = _group_roots(poles, tol)
    # each pole's group value; a real sos's are made exact conjugates
    value = _reduce_groups(poles, label, rtype)
    if not sos.is_complex():
        # a section's conjugate poles are adjacent; a real pole is its own
        partner = torch.arange(2 * K, device=sos.device) ^ conjugates.repeat_interleave(2, -1)
        value = (value + value.gather(-1, partner).conj()) / 2
        # a real sos lists one group of each conjugate pair, that of Im >= 0
        half = value[batch_index, first].imag >= 0
        batch_index, first, multiplicity = batch_index[half], first[half], multiplicity[half]
    unique = value[batch_index, first]
    group = label[batch_index] == first.unsqueeze(-1)
    m = int(multiplicity.max())
    # each pole's group value minus this group's
    away = (value[batch_index] - unique.unsqueeze(-1)).unflatten(-1, (K, 2))

    # the first m terms of the series of 1 / (s - away) are -cumprod(1 / away);
    # a member of the group has 1
    member = group.unflatten(-1, (K, 2))
    series = (
        F.pad(member.unsqueeze(-1), (0, m - 1))
        + torch.cumprod((~member / (away + member)).unsqueeze(-1).expand(-1, -1, -1, m), -1).neg()
    )
    # the numerators at p + s, in ascending powers of s
    b0, b1, b2 = sos[batch_index, :, :3].unbind(-1)
    u = unique.unsqueeze(-1)
    t = b0 * u + b1
    numerator = torch.stack((t * u + b2, t + b0 * u, b0), -1)
    terms = torch.cat((F.pad(numerator, (0, m - 3)).unsqueeze(-2), series), -2)
    f = _tree_prod(terms.flatten(1, 2), m)

    # r_j = f[m - j]: each group's first m terms, reversed
    power = torch.arange(1 - m, 1, device=sos.device) + multiplicity.unsqueeze(-1)
    keep = power >= 1
    p, r, power = unique.unsqueeze(-1).expand(-1, m)[keep], f.flip(-1)[keep], power[keep]
    if not sos.is_complex():
        # a real pole's residue is real; r.real is made complex as a workaround
        # of an upstream bug: torch.where's backward fails for a real and a
        # complex input
        r = torch.where(p.imag == 0, r.real.to(r.dtype), r)
    g = sos[..., 0].prod(-1)
    if not batched:
        return r, p, g[0], None, power
    return r, p, g, batch_index.repeat_interleave(multiplicity), power
