"""Conversions between the forms of time-invariant filters.

The coefficients follow :mod:`scipy.signal`'s functions of the same names, with
the differences noted in each docstring.
"""

from collections.abc import Sequence
from functools import reduce

import torch
import torch.nn.functional as F
from torch import Tensor

from .. import _ssm
from ..mat import _confluent_vandermonde, companion
from ..poly import polymul
from ..utils import _check_shared
from .filtering import _check_sos, _sections_by_row

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


def _group_roots(
    roots: Tensor, tol: float, live: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Group the roots within ``tol`` of each other, directly or through a
    chain of such roots.

    ``roots`` is of shape (B, M), M >= 1. A boolean ``live`` of that shape
    keeps the roots where it is ``False`` apart from the others: they group
    only among themselves. Returns the group of each root, the
    position of its first root, of shape (B, M), and the batch index, the
    position of the first root and the number of roots of each group, of
    shape (G), the groups in row order.
    """
    real = roots.dtype.to_real()
    near = (roots.unsqueeze(-1) - roots.unsqueeze(-2)).abs() <= tol
    if live is not None:
        near &= live.unsqueeze(-1) == live.unsqueeze(-2)
    near = near.to(real)
    # each squaring doubles the length of the chains linked
    steps = range((roots.size(-1) - 1).bit_length())
    same = reduce(lambda c, _: (c @ c > 0).to(real), steps, near) > 0
    label = same.to(torch.int8).argmax(-1)
    M = roots.size(-1)
    rows = torch.arange(roots.size(0), device=roots.device).unsqueeze(-1)
    group, multiplicity = (rows * M + label).unique(return_counts=True)
    return label, group // M, group % M, multiplicity


def sos2tf(sos: Tensor, batch_index: Tensor | None = None) -> tuple[Tensor, Tensor]:
    r"""Convert second-order sections to transfer function coefficients.

    This multiplies the sections' numerators and denominators,

    .. math::
        H(z) = \prod_{l=1}^{K}
            \frac{b_{l0} + b_{l1} z^{-1} + b_{l2} z^{-2}}
                 {1 + a_{l1} z^{-1} + a_{l2} z^{-2}}
            = \frac{b_0 + b_1 z^{-1} + \cdots + b_{2K} z^{-2K}}
                   {1 + a_1 z^{-1} + \cdots + a_{2K} z^{-2K}}.

    With :attr:`batch_index`, each filter is the product of its own sections,
    in the order they are listed, and :math:`K` in the shapes below is the
    most sections of a filter: the coefficients of a filter of fewer
    sections end in zeros, and a filter without a section is 1.

    Args:
        sos (Tensor): the sections, of shape :math:`(K, 5)` for one cascade,
            :math:`K \ge 1`, or :math:`(S, 5)` with :attr:`batch_index`,
            :math:`S \ge 1`. Each row is one section,
            :math:`b_{l0}, b_{l1}, b_{l2}, a_{l1}, a_{l2}`, :math:`l` being its
            place in its cascade, as in :func:`~philtorch.lti.sosfilt`.
        batch_index (Tensor, optional): the filter each section belongs to,
            of shape :math:`(S)`, int32 or int64 on :attr:`sos`'s device and
            not negative. Sections with the same value are multiplied in the
            order they are listed; they need not be consecutive. The filters
            are numbered 0 to :math:`B - 1`, :math:`B` being the largest value
            plus 1; the filters past it, being 1, are left out. Default:
            ``None``, one cascade of all the sections.

    Returns:
        tuple of Tensor: the numerator :math:`b`, of shape :math:`(2K + 1)`,
        and the denominator :math:`a` without its leading 1, of shape
        :math:`(2K)`, both of :attr:`sos`'s dtype and in ascending powers of
        :math:`z^{-1}`, as :func:`~philtorch.lti.lfilter` takes them. With
        :attr:`batch_index`, of shapes :math:`(B, 2K + 1)` and
        :math:`(B, 2K)`.

    Raises:
        ValueError: if :attr:`sos` is not 2-D, its last dimension is not 5, or
            it has no section, or :attr:`batch_index` is not of shape
            :math:`(S)`, not int32 or int64, not on :attr:`sos`'s device, or
            negative.

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
        >>> # the same sections as two filters of one section each
        >>> b, a = sos2tf(sos, torch.tensor([0, 1]))
        >>> b
        tensor([[ 1.,  1.,  0.],
                [ 1., -1.,  0.]])
        >>> a
        tensor([[-0.5000,  0.0000],
                [ 0.5000,  0.0000]])
    """
    _check_sos(sos)
    if batch_index is not None:
        sos = _sections_by_row(sos, batch_index)[0]
    both = _tree_prod(torch.stack((sos[..., :3], F.pad(sos[..., 3:], (1, 0), value=1)), -3))
    return both[..., 0, :], both[..., 1, 1:]


def sos2ss(sos: Tensor, batch_index: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Convert second-order sections to one state-space model.

    Each section is the state-space model :func:`~philtorch.lti.sosfilt`
    runs, in transposed direct form II, and the model returned is their
    cascade, as :func:`cascade_ss` combines them: its state is the
    sections' states in order, so the :math:`(K, 2)` states of
    :func:`~philtorch.lti.sosfilt` flattened are its :math:`2K` states, and
    back.

    With :attr:`batch_index`, each filter is the cascade of its own sections,
    in the order they are listed, and :math:`K` in the shapes below is the
    most sections of a filter: a filter of fewer sections ends in sections
    of gain 1, whose states stay zero from a zero state, and a filter
    without a section is a gain of 1.

    Args:
        sos (Tensor): the sections, of shape :math:`(K, 5)` for one cascade,
            :math:`K \ge 1`, or :math:`(S, 5)` with :attr:`batch_index`,
            :math:`S \ge 1`, as in :func:`sos2tf`.
        batch_index (Tensor, optional): the filter each section belongs to,
            of shape :math:`(S)`, as in :func:`sos2tf`. Default: ``None``,
            one cascade of all the sections.

    Returns:
        tuple of Tensor: the matrices :math:`A`, :math:`B`, :math:`C` and
        :math:`D` of :func:`~philtorch.lti.state_space`, of :attr:`sos`'s
        dtype, for a scalar input and output: of shapes :math:`(2K, 2K)`,
        :math:`(2K)`, :math:`(2K)` and :math:`()`, or with :attr:`batch_index`
        :math:`(B, 2K, 2K)`, :math:`(B, 2K)`, :math:`(B, 2K)` and
        :math:`(B)`, :math:`B` being the largest value of
        :attr:`batch_index` plus 1.

    Raises:
        ValueError: if :attr:`sos` or :attr:`batch_index` is invalid, as in
            :func:`sos2tf`.

    Example::

        >>> from philtorch.lti import sos2ss, sosfilt, state_space
        >>> sos = torch.tensor([[1.0, 1.0, 0.0, -0.5, 0.0],
        ...                     [1.0, -1.0, 0.0, 0.5, 0.0]])
        >>> A, B, C, D = sos2ss(sos)
        >>> A.shape
        torch.Size([4, 4])
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        >>> state_space(A, x, B, C, D)
        tensor([[ 1.0000,  0.0000, -0.7500,  0.0000]])
        >>> sosfilt(sos, x)
        tensor([[ 1.0000,  0.0000, -0.7500,  0.0000]])
    """
    _check_sos(sos)
    if batch_index is not None:
        sos = _sections_by_row(sos, batch_index)[0]
    _, a, B, D = _ssm.direct_form(sos[..., :3], sos[..., 3:])
    C = F.pad(torch.ones_like(B[..., :1]), (0, 1))
    return cascade_ss(companion(a).mT.unbind(-3), B.unbind(-2), C.unbind(-2), D.unbind(-1))


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
        >>> # 1.3 and 1.31 are within tol: one root, of multiplicity 2
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
    A pole of multiplicity :math:`m` is listed :math:`m` times in a row, with
    the powers :math:`j_i = 1, \dots, m`. The poles are the roots of the
    denominator, sorted by magnitude, ascending, then by angle in
    :math:`(-\pi, \pi]`, and grouped by :func:`unique_roots`.

    The roots of :attr:`a` are ill-conditioned at high orders, and so is the
    expansion. For a filter given as sections, use :func:`sos2pfe`, whose poles
    are the sections' own.

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
            sizes, precisions or devices differ (real and complex inputs of
            one precision mix), :attr:`rtype` is unknown, or, without
            ``delayed_form``, :math:`L > M` and :math:`a_M = 0` for any filter.
        torch.linalg.LinAlgError: if a pole at 0 is repeated, or, without
            ``delayed_form``, :math:`p_i^D` underflows for a pole, which
            ``delayed_form=True`` avoids.

    Note:
        Unlike :func:`scipy.signal.residuez`, no zero coefficients are
        trimmed, so the output shapes follow from the input shapes: where
        SciPy trims a pole at 0 from :attr:`a` when :math:`L > M`, this raises
        an error, unless ``delayed_form=True``.

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
    _check_shared(precision=True, b=b, a=a)
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
    V = _confluent_vandermonde(p, extra.view(B, M), M if delayed_form else N)
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
        rtype (str, optional): how to choose the pole of a group of equal poles,
            as in :func:`unique_roots`. Default: ``"avg"``.
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
            their batch sizes, precisions or devices differ (real and complex
            inputs of one precision mix), :attr:`r` and :attr:`p` differ in
            length, or :attr:`rtype` is unknown.

    Note:
        Unlike :func:`scipy.signal.invresz`, no zero coefficients are
        trimmed.

    Example::

        >>> from philtorch.lti import invresz
        >>> # 1.5 / (1 - 0.5 z^-1) - 0.5 / (1 + 0.5 z^-1)
        >>> #     = (1 + z^-1) / (1 - 0.25 z^-2), with no direct term
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
    _check_shared(precision=True, r=r, p=p, k=k)
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
    fractions = (_confluent_vandermonde(poles, extra, M) @ r.unsqueeze(-1)).squeeze(-1)
    numerator = polymul(denominator, fractions)[..., :M]
    d = k.size(-1) if delayed_form else 0
    b = polymul(F.pad(k, (0, 1)), denominator)[..., :L] + F.pad(numerator, (d, L - M - d))
    a = denominator[:, 1:]
    return (b, a) if batched else (b[0], a[0])


def sos2pfe(
    sos: Tensor,
    tol: float | None = None,
    rtype: str = "avg",
    batch_index: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
    r"""Expand second-order sections into partial fractions.

    For the cascade

    .. math::
        H(z) = \prod_{l=1}^{K}
            \frac{b_{l0} + b_{l1} z^{-1} + b_{l2} z^{-2}}
                 {1 + a_{l1} z^{-1} + a_{l2} z^{-2}},

    this returns the residues :math:`r_{ij}`, the poles :math:`p_i` with their
    multiplicities :math:`m_i`, and the direct gain :math:`g = \prod_l b_{l0}`
    of the expansion

    .. math::
        H(z) = g + \sum_{i=1}^{U} \sum_{j=1}^{m_i}
            \frac{r_{ij} z^{-j}}{(1 - p_i z^{-1})^{j}},

    as :func:`~philtorch.lti.pfefilt` takes it, with :math:`U` poles and
    :math:`M = m_1 + \dots + m_U` terms. Each term is delayed by its power
    :math:`j`, so that the powers of a pole at 0 are the taps
    :math:`r_{ij} z^{-j}` of an FIR filter.
    For simple poles, this is the delayed form of the partial fraction
    expansion in `Introduction to Digital Filters with Audio Applications`_
    (Smith, 2007), with the FIR part :math:`F(z) = g`. A real :attr:`sos`
    lists one pole of each conjugate pair, that of positive imaginary part,
    which stands for both in :func:`~philtorch.lti.pfefilt`, and its real
    poles have real residues. A complex :attr:`sos` lists every pole.

    Poles within :attr:`tol` of each other, directly or through a chain of
    such poles, are one group, listed as one pole whose multiplicity is the
    group's size.

    With :attr:`batch_index`, each filter's cascade, its sections in the order
    they are listed, is expanded on its own: a filter's expansion is the sum
    above over the poles whose returned batch index is its number, with its
    own :math:`g`. Poles of different filters are never one group, and a
    filter without a section has no term and :math:`g = 1`.

    The outputs aren't differentiable where a section's two poles are exactly
    equal, :math:`a_{l1}^2 = 4 a_{l2}`, as in every FIR section
    (:math:`a_{l1} = a_{l2} = 0`): the gradients of that section's
    :math:`a_{l1}` and :math:`a_{l2}` are then NaN. Poles merged within
    :attr:`tol`, or repeated across sections, keep finite gradients.

    Args:
        sos (Tensor): the sections, of shape :math:`(K, 5)` for one cascade,
            :math:`K \ge 1`, or :math:`(S, 5)` with :attr:`batch_index`,
            :math:`S \ge 1`. Each row is one section,
            :math:`b_{l0}, b_{l1}, b_{l2}, a_{l1}, a_{l2}`, :math:`l` being its
            place in its cascade, as in :func:`~philtorch.lti.sosfilt`.
        tol (float, optional): the distance within which two poles are
            considered equal. ``None`` means :math:`10 \sqrt{\epsilon}`,
            :math:`\epsilon` being the machine epsilon of :attr:`sos`'s
            dtype, enough to merge a double root that rounding splits.
            Default: ``None``.
        rtype (str, optional): how to choose the pole of a group of equal poles,
            as in :func:`unique_roots`, up to the conjugate symmetry of a real
            :attr:`sos`. Default: ``"avg"``.
        batch_index (Tensor, optional): the filter each section belongs to,
            of shape :math:`(S)`, int32 or int64 on :attr:`sos`'s device and
            not negative, as in :func:`sos2tf`: sections with the same value
            form one cascade, in the order they are listed. The filters are
            numbered 0 to :math:`B - 1`, :math:`B` being the largest value
            plus 1. Default: ``None``, one cascade of all the sections.

    Returns:
        tuple of Tensor: the residues :math:`r`, complex at :attr:`sos`'s
        precision (complex64 for float32) and of shape :math:`(M)`, :math:`M`
        being the number of terms of all the filters: at most two per section, a
        real :attr:`sos` listing one term for a conjugate pair; the poles
        :math:`p`, of the same dtype and of shape :math:`(U)`; the direct
        gain :math:`g`, of :attr:`sos`'s dtype and of shape :math:`()`, or
        :math:`(B)` with :attr:`batch_index`; the batch index, the filter
        each pole belongs to, int64 and of shape :math:`(U)`, or ``None``
        without :attr:`batch_index`; and the multiplicities :math:`m_i`, int64
        and of shape :math:`(U)`. The poles are in the order of the filters,
        and in a filter, in the order of their first appearance, the
        sections' poles one section after the other, a section's pole of
        larger magnitude first; :math:`r` lists each pole's residues in a
        row, by ascending power.

    Raises:
        ValueError: if :attr:`sos` is not 2-D, its last dimension is not 5, it
            has no section, it is float16 or bfloat16, :attr:`rtype` is
            unknown, or :attr:`batch_index` is not of shape :math:`(S)`, not
            int32 or int64, not on :attr:`sos`'s device, or negative.

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
        >>> r, p, g, batch_index, multiplicity = sos2pfe(sos)
        >>> r.real
        tensor([1.5000, 0.0000])
        >>> p.real
        tensor([0.5000, 0.0000])
        >>> g
        tensor(1.)
        >>> multiplicity
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
        >>> r, p, g, batch_index, multiplicity = sos2pfe(sos)
        >>> y = pfefilt(r, p, g, x, batch_index=batch_index,
        ...             multiplicity=multiplicity)
        >>> torch.allclose(y, sosfilt(sos, x))
        True
        >>> # two filters of one section each, the first's as above
        >>> sos = torch.tensor([[1.0, 1.0, 0.0, -0.5, 0.0],
        ...                     [1.0, -1.0, 0.0, 0.5, 0.0]])
        >>> index = torch.tensor([0, 1])
        >>> r, p, g, batch_index, multiplicity = sos2pfe(sos, batch_index=index)
        >>> r.real
        tensor([ 1.5000,  0.0000, -1.5000,  0.0000])
        >>> batch_index
        tensor([0, 0, 1, 1])

    .. _Introduction to Digital Filters with Audio Applications:
        https://ccrma.stanford.edu/~jos/filters/
    """
    if rtype not in _RTYPES:
        raise ValueError(f"rtype must be one of {_RTYPES}, got {rtype!r}")
    _check_sos(sos)
    if sos.dtype in (torch.float16, torch.bfloat16):
        raise ValueError(f"sos can't be float16 or bfloat16, got {sos.dtype}.")
    if tol is None:
        tol = 10 * torch.finfo(sos.dtype).eps ** 0.5
    if batch_index is None:
        sos, live = sos.unsqueeze(0), sos.new_ones(1, sos.size(0), dtype=torch.bool)
    else:
        # the rows' sections, padded with identity sections, which are not live
        sos, counts, _ = _sections_by_row(sos, batch_index)
        live = torch.arange(sos.size(1), device=sos.device) < counts.unsqueeze(-1)
    B, K, _ = sos.shape
    # a padded section's double pole at 0 gives no term: it groups apart from
    # the live poles, and its factors of the products are 1
    live_pole = live.repeat_interleave(2, -1)

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
    label, row, first, multiplicity = _group_roots(poles, tol, live_pole)
    keep = live_pole[row, first]
    # each pole's group value; a real sos's are made exact conjugates
    value = _reduce_groups(poles, label, rtype)
    if not sos.is_complex():
        # a section's conjugate poles are adjacent; a real pole is its own
        partner = torch.arange(2 * K, device=sos.device) ^ conjugates.repeat_interleave(2, -1)
        value = (value + value.gather(-1, partner).conj()) / 2
        # a real sos lists one group of each conjugate pair, that of Im >= 0
        keep &= value[row, first].imag >= 0
    row, first, multiplicity = row[keep], first[keep], multiplicity[keep]
    unique = value[row, first]
    # a padded pole counts as a member, which has the factor 1
    member = (label[row] == first.unsqueeze(-1)) | ~live_pole[row]
    m = int(multiplicity.max())
    # each pole's group value minus this group's
    away = (value[row] - unique.unsqueeze(-1)).masked_fill(member, 0).unflatten(-1, (K, 2))

    # the first m terms of the series of 1 / (s - away) are -cumprod(1 / away);
    # a member of the group has 1
    member = member.unflatten(-1, (K, 2))
    series = (
        F.pad(member.unsqueeze(-1), (0, m - 1))
        + torch.cumprod((~member / (away + member)).unsqueeze(-1).expand(-1, -1, -1, m), -1).neg()
    )
    # the numerators at p + s, in ascending powers of s; a padded section's is 1
    b = torch.where(live[row].unsqueeze(-1), sos[row, :, :3], sos.new_tensor([0.0, 0.0, 1.0]))
    b0, b1, b2 = b.unbind(-1)
    u = unique.unsqueeze(-1)
    t = b0 * u + b1
    numerator = torch.stack((t * u + b2, t + b0 * u, b0), -1)
    terms = torch.cat((F.pad(numerator, (0, m - 3)).unsqueeze(-2), series), -2)
    f = _tree_prod(terms.flatten(1, 2), m)

    # r_j = f[m - j]: each group's first m terms, reversed
    keep = torch.arange(1 - m, 1, device=sos.device) + multiplicity.unsqueeze(-1) >= 1
    r = f.flip(-1)[keep]
    if not sos.is_complex():
        # a real pole's residue is real; r.real is made complex as a workaround
        # of an upstream bug: torch.where's backward fails for a real and a
        # complex input
        real_pole = unique.repeat_interleave(multiplicity, output_size=r.size(0)).imag == 0
        r = torch.where(real_pole, r.real.to(r.dtype), r)
    g = sos[..., 0].prod(-1)
    if batch_index is None:
        return r, unique, g[0], None, multiplicity
    return r, unique, g, row, multiplicity


def cascade_ss(
    A: Sequence[Tensor], B: Sequence[Tensor], C: Sequence[Tensor], D: Sequence[Tensor]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Combine state-space models in cascade into one.

    In the cascade, the output of each model is the input of the next. Two
    models :math:`(A_1, B_1, C_1, D_1)` and :math:`(A_2, B_2, C_2, D_2)`, in
    the form of :func:`~philtorch.lti.state_space`, give

    .. math::
        A = \begin{bmatrix} A_1 & 0 \\ B_2 C_1 & A_2 \end{bmatrix},
        \quad
        B = \begin{bmatrix} B_1 \\ B_2 D_1 \end{bmatrix},
        \quad
        C = \begin{bmatrix} D_2 C_1 & C_2 \end{bmatrix},
        \quad
        D = D_2 D_1,

    whose state is that of the first model followed by that of the second. A
    longer cascade of :math:`K` models combines them in the same way, and its
    state lists the states of :math:`l = 1, \dots, K` in order.

    Args:
        A (tuple or list of Tensor): the state matrices :math:`A_l`, one for
            each model in cascade order, of shape :math:`(M_l, M_l)`, or
            :math:`(B, M_l, M_l)` if batched. The first one decides whether
            the cascade is batched, which every other tensor must follow.
        B (tuple or list of Tensor): the input matrices :math:`B_l`, of shape
            :math:`(M_l)` for a scalar input or :math:`(M_l, F_l)`, with
            a leading :math:`B` if batched.
        C (tuple or list of Tensor): the output matrices :math:`C_l`, of shape
            :math:`(M_l)` for a scalar output or :math:`(P_l, M_l)`, with a
            leading :math:`B` if batched.
        D (tuple or list of Tensor): the feedthrough matrices :math:`D_l`, of
            shape :math:`()` for a scalar input and output, :math:`(P_l)` for
            a scalar input only, :math:`(F_l)` for a scalar output only, or
            :math:`(P_l, F_l)`, with a leading :math:`B` if batched.

    Returns:
        tuple of Tensor: the matrices :math:`A`, :math:`B`, :math:`C` and
        :math:`D` of the cascade, of the dtype and device of the arguments.
        Its input is scalar if :math:`B_1` is a vector, and its output is
        scalar if :math:`C_K` is a vector, so that the shapes follow the
        arguments' conventions and :func:`~philtorch.lti.state_space` takes
        them.

    Raises:
        ValueError: if the arguments aren't tuples or lists of the same
            length, at least 1, if the tensors don't share a dtype and a
            device, if a shape is not one of those above or is not batched
            as the first tensor of :attr:`A` is, or if a model's number of
            outputs differs from the next one's number of inputs, a scalar
            counting as 1.
        TypeError: if an element of the arguments is not a tensor.

    Example::

        >>> from philtorch.lti import cascade_ss, state_space
        >>> # Two integrators in cascade: y[n + 1] = y[n] + x[n] for each.
        >>> one, zero = torch.tensor([[1.0]]), torch.tensor(0.0)
        >>> ss = cascade_ss([one] * 2, [one[0]] * 2, [one[0]] * 2, [zero] * 2)
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        >>> state_space(ss[0], x, *ss[1:])
        tensor([[0., 0., 1., 2., 3.]])
    """
    stages = dict(A=A, B=B, C=C, D=D)
    for name, seq in stages.items():
        if not isinstance(seq, (tuple, list)):
            raise ValueError(f"{name} must be a tuple or list of tensors, got {type(seq).__name__}")
    K = len(A)
    if K < 1 or any(len(seq) != K for seq in stages.values()):
        raise ValueError(
            "A, B, C and D must have the same length, at least 1, got "
            + ", ".join(f"{name}: {len(seq)}" for name, seq in stages.items())
        )
    named = {f"{name}[{k}]": t for name, seq in stages.items() for k, t in enumerate(seq)}
    for name, t in named.items():
        if not isinstance(t, Tensor):
            raise TypeError(f"{name} must be a Tensor, got {type(t).__name__}")
    _check_shared(**named)
    if A[0].dim() not in (2, 3):
        raise ValueError(f"A[0] must be 2D or 3D, got {tuple(A[0].shape)}")

    # Lift every model to matrices, A (M, M), B (M, F), C (P, M) and D (P, F).
    lead = A[0].shape[:-2]
    n = len(lead)
    lifted = []
    for k, (a, b, c, d) in enumerate(zip(A, B, C, D)):
        M = a.size(-1)
        D_shape = (*lead, *c.shape[n:-1], *b.shape[n + 1 :])
        for name, t, ok, expected in (
            ("A", a, a.shape == (*lead, M, M), (*lead, M, M)),
            (
                "B",
                b,
                b.dim() - n in (1, 2) and b.shape[: n + 1] == (*lead, M),
                f"{(*lead, M)} or {(*lead, M, 'F')}",
            ),
            (
                "C",
                c,
                c.dim() - n in (1, 2) and c.shape[:n] == lead and c.size(-1) == M,
                f"{(*lead, M)} or {(*lead, 'P', M)}",
            ),
            ("D", d, d.shape == D_shape, D_shape),
        ):
            if not ok:
                raise ValueError(f"{name}[{k}] must be of shape {expected}, got {tuple(t.shape)}")
        b = b if b.dim() > n + 1 else b.unsqueeze(-1)
        c = c if c.dim() > n + 1 else c.unsqueeze(-2)
        if lifted and lifted[-1][2].size(-2) != b.size(-1):
            raise ValueError(
                f"The number of outputs of model {k - 1} ({lifted[-1][2].size(-2)}) must equal "
                f"the number of inputs of model {k} ({b.size(-1)})"
            )
        lifted.append((a, b, c, d.reshape(*lead, c.size(-2), b.size(-1))))

    # Stack the models along dimension -3, zero-padded to the most states and to one width W
    # of every signal between them: padded states, inputs and outputs stay zero throughout.
    Ms = [a.size(-1) for a, *_ in lifted]
    Mm = max(Ms)
    W = max(max(b.size(-1), c.size(-2)) for _, b, c, _ in lifted)
    stacked = (
        torch.stack([F.pad(m[i], (0, w - m[i].size(-1), 0, h - m[i].size(-2))) for m in lifted], -3)
        for i, (h, w) in enumerate(((Mm, Mm), (Mm, W), (W, Mm), (W, W)))
    )
    real = (torch.arange(Mm) < torch.tensor(Ms)[:, None]).to(A[0].device)
    a, b, c, d = (t[..., 0, :, :] for t in _cascade_stacked(*stacked, real, Ms))

    # The cascade's inputs are the first model's, its outputs the last one's.
    F_in, P_out = lifted[0][1].size(-1), lifted[-1][2].size(-2)
    b, c, d = b[..., :F_in], c[..., :P_out, :], d[..., :P_out, :F_in]
    if C[-1].dim() == n + 1:
        c, d = c[..., 0, :], d[..., 0, :]
    if B[0].dim() == n + 1:
        b, d = b[..., 0], d[..., 0]
    return a, b, c, d


def _cascade_stacked(
    A: Tensor, B: Tensor, C: Tensor, D: Tensor, real: Tensor, Ms: list[int]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine the S models stacked along dimension -3 in cascade, merging neighbours in pairs.

    A, B, C and D are (..., S, M, M), (..., S, M, W), (..., S, W, M) and (..., S, W, W),
    zero-padded; real (S, M) marks each model's states and Ms counts them. Returns the
    cascade, stacked as one model.
    """
    if len(Ms) == 1:
        return A, B, C, D
    if len(Ms) % 2:
        # An identity model, with no states, completes the last pair.
        A, B, C = (F.pad(t, (0, 0, 0, 0, 0, 1)) for t in (A, B, C))
        D = torch.cat(
            [D, torch.eye(D.size(-1), dtype=D.dtype, device=D.device).expand_as(D[..., :1, :, :])],
            -3,
        )
        real, Ms = F.pad(real, (0, 0, 0, 1)), [*Ms, 0]
    A1, B1, C1, D1 = (t[..., 0::2, :, :] for t in (A, B, C, D))
    A2, B2, C2, D2 = (t[..., 1::2, :, :] for t in (A, B, C, D))
    A = torch.cat([torch.cat([A1, torch.zeros_like(A1)], -1), torch.cat([B2 @ C1, A2], -1)], -2)
    B, C, D = torch.cat([B1, B2 @ D1], -2), torch.cat([D2 @ C1, C2], -1), D2 @ D1
    real, Ms = torch.cat([real[0::2], real[1::2]], -1), [*map(sum, zip(Ms[0::2], Ms[1::2]))]
    # Move each model's states ahead of its padding, and drop the padding beyond the most states.
    order = (~real).byte().argsort(dim=-1, stable=True)[:, : max(Ms)]
    real = real.take_along_dim(order, -1)
    order = order.unsqueeze(-1).expand(*A.shape[:-3], -1, -1, -1)
    A = A.take_along_dim(order, -2).take_along_dim(order.mT, -1)
    B, C = B.take_along_dim(order, -2), C.take_along_dim(order.mT, -1)
    return _cascade_stacked(A, B, C, D, real, Ms)
