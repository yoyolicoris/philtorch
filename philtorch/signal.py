import torch
from torch import Tensor
from typing import Optional, Tuple
from functools import reduce


from .poly import roots, polydiv


def unique_roots(
    p: Tensor, tol: float = 1e-3, rtype: str = "min"
) -> Tuple[Tensor, Tensor]:
    """Determine unique roots and their multiplicities from a list of roots."""
    match rtype:
        case "max" | "maximum":
            reduction = torch.max
        case "min" | "minimum":
            reduction = torch.min
        case "avg" | "mean":
            reduction = torch.mean
        case _:
            raise ValueError(
                "`rtype` must be one of "
                "{'max', 'maximum', 'min', 'minimum', 'avg', 'mean'}"
            )

    assert p.is_complex(), "Input must be complex array."
    assert p.ndim == 1, "Input must be rank-1 array."

    points = torch.view_as_real_copy(p)
    dist = torch.cdist(points, points)
    group_mask = dist < tol

    p_unique = []
    p_multiplicity = []
    used = torch.zeros(len(p), dtype=torch.bool)
    # for i in range(len(p)):
    #     if used[i]:
    #         continue

    #     group = group_mask[i] & ~used
    #     used[group] = True
    #     p_unique.append(reduction(p[group]))
    #     p_multiplicity.append(torch.count_nonzero(group))

    p_unique, p_multiplicity, _ = reduce(
        lambda acc, mask: (
            acc[0] + [reduction(p[mask & ~acc[2]])],
            acc[1] + [torch.count_nonzero(mask & ~acc[2])],
            acc[2] | mask,
        ),
        group_mask.unbind(0),
        ([], [], used),
    )

    return torch.stack(p_unique), torch.stack(p_multiplicity)


def _compute_residues(poles: Tensor, multiplicity: Tensor, numerator: Tensor) -> Tensor:
    denominator_factors, _ = _compute_factors(poles, multiplicity)
    numerator = numerator.astype(poles.dtype)

    residues = []
    for pole, mult, factor in zip(poles, multiplicity, denominator_factors):
        if mult == 1:
            residues.append(np.polyval(numerator, pole) / np.polyval(factor, pole))
        else:
            numer = numerator.copy()
            monomial = np.array([1, -pole])
            factor, d = np.polydiv(factor, monomial)

            block = []
            for _ in range(mult):
                numer, n = np.polydiv(numer, monomial)
                r = n[0] / d[0]
                numer = np.polysub(numer, r * factor)
                block.append(r)

            residues.extend(reversed(block))

    return np.asarray(residues)


def residue(
    b: Tensor, a: Tensor, tol: float = 1e-3, rtype: str = "avg"
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute partial-fraction expansion of b(s) / a(s).

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

              b(s)     b[0] s**(M) + b[1] s**(M-1) + ... + b[M]
      H(s) = ------ = ------------------------------------------
              a(s)     a[0] s**(N) + a[1] s**(N-1) + ... + a[N]

    then the partial-fraction expansion H(s) is defined as::

               r[0]       r[1]             r[-1]
           = -------- + -------- + ... + --------- + k(s)
             (s-p[0])   (s-p[1])         (s-p[-1])

    If there are any repeated roots (closer together than `tol`), then H(s)
    has terms like::

          r[i]      r[i+1]              r[i+n-1]
        -------- + ----------- + ... + -----------
        (s-p[i])  (s-p[i])**2          (s-p[i])**n

    This function is used for polynomials in positive powers of s or z,
    such as analog filters or digital filters in controls engineering.  For
    negative powers of z (typical for digital filters in DSP), use `residuez`.

    See Notes for details about the algorithm.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    r : ndarray
        Residues corresponding to the poles. For repeated poles, the residues
        are ordered to correspond to ascending by power fractions.
    p : ndarray
        Poles ordered by magnitude in ascending order.
    k : ndarray
        Coefficients of the direct polynomial term.

    See Also
    --------
    invres, residuez, numpy.poly, unique_roots

    Notes
    -----
    The "deflation through subtraction" algorithm is used for
    computations --- method 6 in [1]_.

    The form of partial fraction expansion depends on poles multiplicity in
    the exact mathematical sense. However there is no way to exactly
    determine multiplicity of roots of a polynomial in numerical computing.
    Thus you should think of the result of `residue` with given `tol` as
    partial fraction expansion computed for the denominator composed of the
    computed poles with empirically determined multiplicity. The choice of
    `tol` can drastically change the result if there are close poles.

    References
    ----------
    .. [1] J. F. Mahoney, B. D. Sivazlian, "Partial fractions expansion: a
           review of computational methodology and efficiency", Journal of
           Computational and Applied Mathematics, Vol. 9, 1983.
    """
    assert (
        b.is_floating_point() and a.is_floating_point()
    ), "Residue function only supports floating point types."
    assert b.ndim == 1 and a.ndim == 1, "Input must be rank-1 arrays."
    assert b.numel() > 0 and a.numel() > 0, "Input must be non-empty arrays."

    # poles = np.roots(a)
    poles = roots(a)

    if b.numel() < a.numel():
        k = torch.empty(0)
    else:
        k, b = polydiv(b, a)

    unique_poles, multiplicity = unique_roots(poles, tol=tol, rtype=rtype)
    order = torch.argsort(torch.abs(unique_poles))
    unique_poles = unique_poles[order]
    multiplicity = multiplicity[order]

    residues = _compute_residues(unique_poles, multiplicity, b)
    poles = torch.cat(
        sum([[pole] * mult for pole, mult in zip(unique_poles, multiplicity)], [])
    )
    return residues / a[0], poles, k
