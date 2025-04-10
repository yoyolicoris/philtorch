import torch
from torch import Tensor
from typing import Tuple


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
        # roots = eigvals(A)
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

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    The input arrays are the coefficients (including any coefficients
    equal to zero) of the "numerator" (dividend) and "denominator"
    (divisor) polynomials, respectively.

    Parameters
    ----------
    u : array_like or poly1d
        Dividend polynomial's coefficients.

    v : array_like or poly1d
        Divisor polynomial's coefficients.

    Returns
    -------
    q : ndarray
        Coefficients, including those equal to zero, of the quotient.
    r : ndarray
        Coefficients, including those equal to zero, of the remainder.

    See Also
    --------
    poly, polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub
    polyval

    Notes
    -----
    Both `u` and `v` must be 0-d or 1-d (ndim = 0 or 1), but `u.ndim` need
    not equal `v.ndim`. In other words, all four possible combinations -
    ``u.ndim = v.ndim = 0``, ``u.ndim = v.ndim = 1``,
    ``u.ndim = 1, v.ndim = 0``, and ``u.ndim = 0, v.ndim = 1`` - work.

    Examples
    --------
    .. math:: \\frac{3x^2 + 5x + 2}{2x + 1} = 1.5x + 1.75, remainder 0.25

    >>> x = np.array([3.0, 5.0, 2.0])
    >>> y = np.array([2.0, 1.0])
    >>> np.polydiv(x, y)
    (array([1.5 , 1.75]), array([0.25]))

    """
    assert (
        u.is_floating_point() and v.is_floating_point()
    ), "polydiv function only supports floating point types."
    assert u.ndim == 1 and v.ndim == 1, "polydiv function only supports 1D arrays."

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
