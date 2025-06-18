import torch
from torch import Tensor
from typing import Optional, Tuple, List
from functools import reduce
from itertools import chain, starmap

from .poly import roots, polydiv, polymul, polyval, polysub, polysmul, polyder, polyadd


def unique_roots(
    p: Tensor,
    tol: float = 1e-3,
    rtype: str = "min",
    is_complex: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
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

    if is_complex is None:
        is_complex = p.new_ones(len(p), dtype=torch.bool)
    p_unique = []
    p_multiplicity = []
    p_is_complex = []
    used = p.new_zeros(len(p), dtype=torch.bool)
    for i in range(len(p)):
        if used[i]:
            continue

        group = group_mask[i] & ~used & (is_complex if is_complex[i] else ~is_complex)
        used[group] = True
        p_unique.append(reduction(p[group]))
        p_multiplicity.append(torch.count_nonzero(group))
        p_is_complex.append(is_complex[i])

    return torch.stack(p_unique), torch.stack(p_multiplicity), torch.stack(p_is_complex)


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

    unique_poles, multiplicity, _ = unique_roots(poles, tol=tol, rtype=rtype)
    order = torch.argsort(torch.abs(unique_poles), descending=True)
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

    unique_poles, multiplicity, _ = unique_roots(poles, tol=tol, rtype=rtype)
    order = torch.argsort(torch.abs(unique_poles), descending=True)
    unique_poles = unique_poles[order]
    multiplicity = multiplicity[order]

    residues = _compute_residues(unique_poles.reciprocal(), multiplicity, b_rev)

    poles = torch.stack(
        sum([[pole] * mult for pole, mult in zip(unique_poles, multiplicity)], [])
    )
    powers = torch.concat([torch.arange(0, mult) + 1 for mult in multiplicity])

    residues = residues * (-poles) ** powers / a_rev[0]

    return residues, poles, k_rev.flip(0)


def sos2pfe(
    sos: Tensor, tol: float = 0.001, rtype: str = "avg"
) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor]:
    """Convert second-order sections to partial-fraction expansion."""
    assert sos.is_floating_point(), "SOS must be floating point type."
    assert sos.ndim == 2, "SOS must be 2D array."
    assert sos.shape[1] == 6, "SOS must have 6 columns."

    sections, _ = sos.shape
    b, a = sos.chunk(2, dim=-1)
    k0 = b[:, 0] / a[:, 0]
    b = b[:, 1:] / b[:, :1]
    a = a[:, 1:] / a[:, :1]

    expsumlog = lambda x: x.log().sum(dim=-1).exp()
    G0 = expsumlog(k0)

    a1, a2 = a[:, 0], a[:, 1]
    b1, b2 = b[:, 0], b[:, 1]

    sqrt_term = a1 * a1 - 4 * a2
    is_complex_root = sqrt_term < 0

    all_roots = (
        torch.stack(
            [
                -a1 + (sqrt_term + 0j).sqrt(),
                -a1 - (sqrt_term + 0j).sqrt(),
            ],
            dim=1,
        ).flatten()
        * 0.5
    )
    is_complex_root_ext = is_complex_root.repeat(2)
    unique_poles, multiplicity, is_complex = unique_roots(
        all_roots, tol=tol, rtype=rtype, is_complex=is_complex_root_ext
    )

    order = torch.argsort(torch.abs(unique_poles), descending=True)
    unique_poles = unique_poles[order]
    multiplicity = multiplicity[order]
    is_complex = is_complex[order]

    repeated_mask = multiplicity > 1

    if torch.any(~repeated_mask):
        non_repeated_poles = unique_poles[~repeated_mask]
        numerator_val = expsumlog(
            non_repeated_poles[:, None] * non_repeated_poles[:, None]
            + non_repeated_poles[:, None] * b1
            + b2
        )

        mask = repeated_mask.new_zeros(len(non_repeated_poles), len(unique_poles))
        mask[range(len(non_repeated_poles)), torch.where(~repeated_mask)[0]] = True

        denominator_val = expsumlog(
            (non_repeated_poles[:, None] - unique_poles)
            ** torch.where(mask, 0, multiplicity)
        )
        non_repeated_residues = numerator_val / denominator_val

        sub_complex_mask = is_complex[~repeated_mask]
        real_non_repeated = (
            non_repeated_poles[~sub_complex_mask].real,
            non_repeated_residues[~sub_complex_mask].real,
        )
        cplx_non_repeated = (
            non_repeated_poles[sub_complex_mask],
            non_repeated_residues[sub_complex_mask],
        )
    else:
        real_non_repeated = (G0.new_empty(0),) * 2
        cplx_non_repeated = (unique_poles.new_empty(0),) * 2

    if torch.any(repeated_mask):
        repeated_poles = unique_poles[repeated_mask]

        numerator = polysmul(*torch.hstack([b.new_ones(sections, 1), b]).unbind(0))
        monomial = torch.vstack([torch.ones_like(unique_poles), -unique_poles]).T
        non_repeated_monomial = monomial[~repeated_mask]
        repeated_monomial = monomial[repeated_mask]
        if torch.all(repeated_mask):
            non_repeated_denominator = repeated_monomial.new_ones(1)
        else:
            non_repeated_denominator = polysmul(*non_repeated_monomial.unbind(0))

        repeated_residues = []
        for i in range(len(repeated_poles)):
            pole = repeated_poles[i]
            repeats = multiplicity[repeated_mask][i]
            divided_denominator = (
                lambda x: polysmul(*x) if len(x) > 0 else pole.new_ones(1)
            )(
                sum(
                    map(
                        lambda x: [x[0]] * x[1],
                        map(
                            lambda x: x[1] if i != x[0] else (x[1][0], 0),
                            enumerate(
                                zip(
                                    repeated_monomial.unbind(0),
                                    multiplicity[repeated_mask].unbind(0),
                                )
                            ),
                        ),
                    ),
                    [],
                )
            )

            d = polymul(non_repeated_denominator, divided_denominator)
            n_val = polyval(numerator, pole)
            d_val = polyval(d, pole)
            block = [n_val / d_val]

            n = numerator.clone()
            for j in range(2, repeats + 1):
                n_diff = polyder(n)
                d_diff = polyder(d)
                n = polysub(polymul(n_diff + 0j, d), polymul(n + 0j, d_diff))
                d = polymul(d, d) * (j - 1)
                block.append(polyval(n, pole) / polyval(d, pole))

            repeated_residues.append(block)

        sub_complex_mask = is_complex[repeated_mask]

        real_repeated = (
            repeated_poles[~sub_complex_mask].real.repeat_interleave(
                multiplicity[repeated_mask][~sub_complex_mask]
            ),
            torch.stack(
                sum(
                    [repeated_residues[i] for i in torch.where(~sub_complex_mask)[0]],
                    [],
                )
            ).real,
            torch.cat(
                [
                    torch.arange(mult.item(), 0, -1, device=repeated_poles.device)
                    for mult in multiplicity[repeated_mask][~sub_complex_mask]
                ],
            ),
        )
        cplx_repeated = (
            repeated_poles[sub_complex_mask].repeat_interleave(
                multiplicity[repeated_mask][sub_complex_mask]
            ),
            (
                torch.stack(
                    sum(
                        [
                            repeated_residues[i]
                            for i in torch.where(sub_complex_mask)[0]
                        ],
                        [],
                    )
                )
                if torch.any(sub_complex_mask)
                else repeated_poles.new_empty(0)
            ),
            (
                torch.cat(
                    [
                        torch.arange(mult.item(), 0, -1, device=repeated_poles.device)
                        for mult in multiplicity[repeated_mask][sub_complex_mask]
                    ],
                )
                if torch.any(sub_complex_mask)
                else multiplicity.new_empty(0)
            ),
        )
    else:
        real_repeated = (G0.new_empty(0),) * 2 + (G0.new_empty(0, dtype=torch.long),)
        cplx_repeated = (unique_poles.new_empty(0),) * 2 + (
            unique_poles.new_empty(0, dtype=torch.long),
        )

    real_poles = torch.cat([real_non_repeated[0], real_repeated[0]], dim=0)
    real_residues = torch.cat([real_non_repeated[1], real_repeated[1]], dim=0) * G0
    real_powers = torch.cat(
        [real_repeated[2].new_ones(len(real_non_repeated[0])), real_repeated[2]], dim=0
    )

    cplx_poles = torch.cat([cplx_non_repeated[0], cplx_repeated[0]], dim=0)
    cplx_residues = torch.cat([cplx_non_repeated[1], cplx_repeated[1]], dim=0) * G0
    cplx_powers = torch.cat(
        [cplx_repeated[2].new_ones(len(cplx_non_repeated[0])), cplx_repeated[2]], dim=0
    )

    assert len(cplx_poles) % 2 == 0, "Complex poles must be even."
    cplx_poles = cplx_poles[::2]
    cplx_residues = cplx_residues[::2]
    cplx_powers = cplx_powers[::2]

    return (
        (real_residues, cplx_residues),
        (real_poles, cplx_poles),
        (real_powers, cplx_powers),
        G0,
    )
