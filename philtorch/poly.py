import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from functools import reduce


def trim_zeros(p: Tensor) -> Tensor:
    """
    Trim leading zeros from a polynomial.

    """
    if p.numel() == 1:
        return p

    # find non-zero array entries
    non_zero = torch.nonzero(p).squeeze()
    if non_zero.numel() == 0:
        return p.new_zeros(1)
    non_zero = non_zero.view(-1)
    return p[non_zero[0] :]


def roots(p: Tensor) -> Tensor:
    """
    Return the roots of a polynomial with coefficients given in p.

    """
    # If input is scalar, this makes it an array
    assert p.is_floating_point(), "Roots function only supports floating point types."
    if p.ndim != 1:
        raise ValueError("Input must be a rank-1 array.")

    # find non-zero array entries
    non_zero = torch.nonzero(p).squeeze()

    # Return an empty array if polynomial is all zeros
    if non_zero.numel() == 0:
        return torch.tensor([])

    # find the number of trailing zeros -- this is the number of roots at 0.
    trailing_zeros = p.numel() - non_zero[-1] - 1

    # strip leading and trailing zeros
    p = p[int(non_zero[0]) : int(non_zero[-1]) + 1]

    N = p.numel()
    if N > 1:
        # build companion matrix and find its eigenvalues (the roots)
        # A = diag(NX.ones((N - 2,), p.dtype), -1)
        A = torch.diag(p.new_ones(N - 2), -1)
        A[0, :] = -p[1:] / p[0]
        roots = torch.linalg.eigvals(A)
    else:
        roots = torch.tensor([])

    # tack any zeros onto the back of the array
    if trailing_zeros > 0:
        roots = torch.cat((roots, p.new_zeros(trailing_zeros)))
    return roots


def polydiv(u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Returns the quotient and remainder of polynomial division.

    """
    # assert (
    #     u.is_floating_point() and v.is_floating_point()
    # ), "polydiv function only supports floating point types."
    assert u.ndim == 1 and v.ndim == 1, "polydiv function only supports 1D arrays."

    u = trim_zeros(u)
    v = trim_zeros(v)

    # w has the common type
    m = len(u) - 1
    n = len(v) - 1
    scale = 1.0 / v[0]
    q = u.new_zeros(max(m - n + 1, 1))
    r = u.clone()
    for k in range(0, m - n + 1):
        d = scale * r[k]
        q[k] = d
        r[k : k + n + 1] -= d * v
    while torch.abs(r[0]) < 1e-8 and (r.shape[-1] > 1):
        r = r[1:]
    return q, r


def polysmul(*polynomials: Tensor) -> Tensor:
    n = len(polynomials)
    if n == 1:
        return polynomials[0]

    a1 = polysmul(*polynomials[: n // 2])
    a2 = polysmul(*polynomials[n // 2 :])
    return polymul(a1, a2)


def polymul(a1: Tensor, a2: Tensor) -> Tensor:
    a1 = trim_zeros(a1)
    a2 = trim_zeros(a2)
    if a1.shape[0] > a2.shape[0]:
        a1, a2 = a2, a1
    weight = a1.flip(0).unsqueeze(0).unsqueeze(0)
    prod = (
        F.conv1d(
            a2.unsqueeze(0).unsqueeze(0),
            weight,
            padding=weight.shape[2] - 1,
            # groups=c2.shape[0],
        )
        .squeeze(0)
        .squeeze(0)
    )
    return prod


def polyval(p: Tensor, x: Tensor) -> Tensor:
    """
    Evaluate a polynomial at specific values.
    """
    p = trim_zeros(p)
    if p.numel() == 1:
        return p
    return reduce(lambda y, pv: y * x + pv, p.unbind(0))


def polysub(a1: Tensor, a2: Tensor) -> Tensor:
    """
    Difference (subtraction) of two polynomials.

    """
    diff = len(a2) - len(a1)
    if diff == 0:
        val = a1 - a2
    elif diff > 0:
        zr = a1.new_zeros(diff)
        val = torch.cat((zr, a1)) - a2
    else:
        zr = a2.new_zeros(-diff)
        val = a1 - torch.cat((zr, a2))
    return val


def polyder(p: Tensor, m: int = 1) -> Tensor:
    """
    Return the derivative of the specified order of a polynomial.
    """
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")

    if m == 0:
        return p

    p = trim_zeros(p)

    if p.numel() == 1:
        return p.new_zeros(1)

    n = len(p) - 1
    y = p[:-1] * torch.arange(n, 0, -1, device=p.device).float()
    return polyder(y, m - 1)


def polyadd(a1: Tensor, a2: Tensor) -> Tensor:
    """
    Sum (addition) of two polynomials.

    """
    diff = len(a2) - len(a1)
    if diff == 0:
        val = a1 + a2
    elif diff > 0:
        zr = a1.new_zeros(diff)
        val = torch.cat((zr, a1)) + a2
    else:
        zr = a2.new_zeros(-diff)
        val = a1 + torch.cat((zr, a2))
    return val
