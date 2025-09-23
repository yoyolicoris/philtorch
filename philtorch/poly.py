import torch
from torch import Tensor
from typing import Tuple


def polydiv(u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    """Divide polynomials and return quotient and remainder.

    Performs polynomial long division of `u` by `v` along the last dimension.

    Args:
        u (Tensor): Dividend coefficients (..., M+1) where highest-degree
            coefficient is at index 0.
        v (Tensor): Divisor coefficients (..., N+1).

    Returns:
        A 2-tuple where the first element is the quotient coefficients
        (..., M-N+1) and the second element is the remainder coefficients
        (..., N) if M >= N, otherwise the quotient is an empty tensor and the
        remainder is `u`.
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
