import torch
from torch import Tensor
from typing import Tuple


def polydiv(u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Returns the quotient and remainder of polynomial division.

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
