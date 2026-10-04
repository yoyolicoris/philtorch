"""Polynomial helpers for batched coefficient tensors."""

import torch
from torch import Tensor


def polydiv(u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Divide polynomials, returning the quotient and the remainder.

    This performs long division of ``u`` by ``v`` along the last dimension.
    Coefficients run from the highest degree down, as in :func:`numpy.polydiv`.

    Args:
        u (Tensor): Dividend coefficients with shape ``(..., M + 1)``.
        v (Tensor): Divisor coefficients with shape ``(..., N + 1)``, where
            ``N <= M``. Its batch dimensions must broadcast to those of ``u``, and
            its leading coefficient ``v[..., 0]`` must be nonzero.

    Returns:
        tuple[Tensor, Tensor]: The quotient, with shape ``(..., M - N + 1)``, and
        the remainder, with shape ``(..., N)``.

    Raises:
        RuntimeError: If ``N > M``, or if ``u`` has an integer dtype.

    Note:
        Unlike :func:`numpy.polydiv`, the remainder always has ``N`` coefficients:
        leading zeros are kept, not trimmed.

    Example:
        >>> import torch
        >>> from philtorch.poly import polydiv
        >>> # x^3 - 3x^2 + 4 = (x - 1)(x^2 - 2x - 2) + 2
        >>> polydiv(torch.tensor([1.0, -3.0, 0.0, 4.0]), torch.tensor([1.0, -1.0]))
        (tensor([ 1., -2., -2.]), tensor([2.]))
    """
    assert u.ndim >= 1 and v.ndim >= 1

    # w has the common type
    m = u.size(-1) - 1
    n = v.size(-1) - 1
    scale = v[..., 0].reciprocal()
    r = u.clone()
    q = []
    for k in range(0, m - n + 1):
        d = scale * r[..., k]
        q.append(d)
        r[..., k : k + n + 1] -= d.unsqueeze(-1) * v

    r = r[..., m - n + 1 :]
    return torch.stack(q, dim=-1), r
