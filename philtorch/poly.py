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
    # scale = 1.0 / v[0]
    scale = v[..., 0].reciprocal()  # Use reciprocal for better numerical stability
    r = u.clone()
    q = []
    for k in range(0, m - n + 1):
        d = scale * r[..., k]
        # q[k] = d
        q.append(d)
        r[..., k : k + n + 1] -= d.unsqueeze(-1) * v

    all_zeros = torch.all(torch.abs(r) < 1e-8, dim=tuple(range(r.dim() - 1)))
    # while torch.abs(r[0]) < 1e-8 and (r.shape[-1] > 1):
    for i in range(all_zeros.shape[0]):
        if all_zeros[i]:
            r = r[..., 1:]
        else:
            break
    return torch.stack(q, dim=-1), r
