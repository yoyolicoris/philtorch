import torch
from torch import Tensor
from typing import Optional, Tuple, List

from .poly import roots, polydiv, polymul, polyval, polysub, polysmul


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
    used = p.new_zeros(len(p), dtype=torch.bool)
    for i in range(len(p)):
        if used[i]:
            continue

        group = group_mask[i] & ~used
        used[group] = True
        p_unique.append(reduction(p[group]))
        p_multiplicity.append(torch.count_nonzero(group))

    return torch.stack(p_unique), torch.stack(p_multiplicity)


def _compute_factors(
    roots: Tensor, multiplicity: Tensor, include_powers: bool = False
) -> Tuple[List[Tensor], Tensor]:
    """Compute the total polynomial divided by factors for each root."""
    current = roots.new_ones(1)
    suffixes = [current]
    monomial = torch.vstack([torch.ones_like(roots), -roots]).T
    for mole, mult in zip(monomial.flip(0)[:-1], multiplicity.flip(0)[:-1]):
        current = polysmul(current, *[mole] * mult)
        suffixes.append(current)
    suffixes = suffixes[::-1]

    factors = []
    current = roots.new_ones(1)
    for mole, mult, suffix in zip(monomial, multiplicity, suffixes):
        block = []
        for i in range(mult):
            if i == 0 or include_powers:
                block.append(polymul(current, suffix))
            current = polymul(current, mole)
        factors.extend(reversed(block))

    return factors, current


def _compute_residues(poles: Tensor, multiplicity: Tensor, numerator: Tensor) -> Tensor:
    denominator_factors, _ = _compute_factors(poles, multiplicity)
    numerator = numerator.to(poles.dtype)

    residues = []
    for pole, mult, factor in zip(poles, multiplicity, denominator_factors):
        if mult == 1:
            residues.append(polyval(numerator, pole) / polyval(factor, pole))
        else:
            numer = numerator.clone()
            monomial = torch.stack([torch.ones_like(pole), -pole])
            factor, d = polydiv(factor, monomial)

            block = []
            for _ in range(mult):
                numer, n = polydiv(numer, monomial)
                r = n[0] / d[0]
                numer = polysub(numer, r * factor)
                block.append(r)

            residues.extend(reversed(block))

    return torch.stack(residues)


def residue(
    b: Tensor, a: Tensor, tol: float = 1e-3, rtype: str = "avg"
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute partial-fraction expansion of b(s) / a(s)."""
    assert (
        b.is_floating_point() and a.is_floating_point()
    ), "Residue function only supports floating point types."
    assert b.ndim == 1 and a.ndim == 1, "Input must be rank-1 arrays."
    assert b.numel() > 0 and a.numel() > 0, "Input must be non-empty arrays."

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
    poles = torch.stack(
        sum([[pole] * mult for pole, mult in zip(unique_poles, multiplicity)], [])
    )
    return residues / a[0], poles, k


def residuez(
    b: Tensor, a: Tensor, tol: float = 1e-3, rtype: str = "avg"
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute partial-fraction expansion of b(z) / a(z)."""
    assert (
        b.is_floating_point() and a.is_floating_point()
    ), "Residue function only supports floating point types."
    assert b.ndim == 1 and a.ndim == 1, "Input must be rank-1 arrays."
    assert b.numel() > 0 and a.numel() > 0, "Input must be non-empty arrays."

    poles = roots(a)

    b_rev = b.flip(0)
    a_rev = a.flip(0)

    if len(b_rev) < len(a_rev):
        k_rev = torch.empty(0)
    else:
        k_rev, b_rev = polydiv(b_rev, a_rev)

    unique_poles, multiplicity = unique_roots(poles, tol=tol, rtype=rtype)
    order = torch.argsort(torch.abs(unique_poles))
    unique_poles = unique_poles[order]
    multiplicity = multiplicity[order]

    residues = _compute_residues(unique_poles.reciprocal(), multiplicity, b_rev)

    poles = torch.stack(
        sum([[pole] * mult for pole, mult in zip(unique_poles, multiplicity)], [])
    )
    powers = torch.concat([torch.arange(0, mult) + 1 for mult in multiplicity])

    residues = residues * (-poles) ** powers / a_rev[0]

    return residues, poles, k_rev.flip(0)


def sos2pfe(sos: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert second-order sections to partial-fraction expansion."""
    assert sos.is_floating_point(), "SOS must be floating point type."
    assert sos.ndim == 3, "SOS must be 3D array."
    assert sos.shape[2] == 6, "SOS must have 6 columns."

    batch, sections, _ = sos.shape
    b, a = sos.chunk(2, dim=-1)
    b = b / a[:, :, :1]
    a = a[..., 1:] / a[:, :, :1]
    k0 = b[:, :, 0]
    b = b[..., 1:] / b[:, :, :1]
    G0 = k0.log().sum(1).exp()

    sqrt_term = a[..., 0] ** 2 - 4 * a[..., 1]
    is_complex_root = sqrt_term < 0
    is_double_root = sqrt_term == 0
    is_real_root = ~is_complex_root & ~is_double_root

    # if torch.any(is_real_root)
