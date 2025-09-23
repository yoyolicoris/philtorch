import torch
from torch import Tensor
from itertools import accumulate
from sympy.ntheory import factorint


def find_eigenvectors(A: Tensor, eigenvalues: Tensor) -> Tensor:
    """Construct normalised eigenvectors for a square matrix from its eigenvalues.

    This solves (A - lambda I) v = 0 for each eigenvalue and returns normalised right-eigenvectors. The output is shaped so that columns correspond to eigenvectors.

    Args:
        A (Tensor): Square matrix of shape (..., N, N).
        eigenvalues (Tensor): Eigenvalues of shape (..., N).

    Returns:
        Tensor: Eigenvectors with shape (..., N, N) where columns are eigenvectors.

    Raises:
        AssertionError: If inputs are not compatible.
    """
    assert A.dim() >= 2, "Matrix A must be at least 2D."
    assert eigenvalues.dim() >= 1, "Eigenvalues must be at least 1D."
    assert A.size(-2) == A.size(-1), "Matrix A must be square."
    assert A.size(-1) == eigenvalues.size(-1), "Eigenvalues must match the size of A."

    n = A.size(-1)
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    W = A.unsqueeze(-3) - eigenvalues[..., None, None] * I
    B, W = torch.split(W, [1, n - 1], dim=-1)

    WtW = W.mT.conj() @ W
    WtB = W.mT.conj() @ -B

    X = torch.linalg.solve(WtW, WtB)
    X = X.reshape(A.shape[:-2] + (n, n - 1))
    X = torch.cat([X.new_ones(X.shape[:-1] + (1,)), X], dim=-1)
    X = X / torch.linalg.vector_norm(X, dim=-1, keepdim=True)
    return X.mT


def companion(a: Tensor) -> Tensor:
    """Create the companion matrix from all-pole coefficients.

    The companion matrix is commonly used to convert polynomial coefficients
    into a state-space representation. Given coefficient vector `a` of length
    M, the returned matrix has shape (..., M, M).

    Args:
        a (Tensor): All-pole coefficients with shape (..., M).

    Returns:
        Tensor: Companion matrix of shape (..., M, M).
    """
    assert a.dim() >= 1, "All-pole coefficients must be at least 1D."
    M = a.size(-1)
    c = torch.cat([-a, a.new_zeros(a.shape[:-1] + (M * (M - 1),))], dim=-1).unflatten(
        -1, (M, M)
    )
    # c = A + torch.diag(a.new_ones(M - 1), diagonal=-1)
    c[..., list(range(1, M)), list(range(M - 1))] = 1
    return c


def vandermonde(poles: Tensor) -> Tensor:
    """Return a Vandermonde matrix constructed from input poles.

    For a poles vector p = [p0, p1, ..., p_{M-1}], the Vandermonde matrix
    returned has columns corresponding to successive powers of the poles.

    Args:
        poles (Tensor): Poles of shape (..., M).

    Returns:
        Tensor: Vandermonde matrix of shape (..., M, M).
    """
    if poles.size(-1) == 1:
        return torch.ones_like(poles).unsqueeze(-1)
    return torch.linalg.vander(poles).flip(-1).mT


def matrix_power_accumulate(A: Tensor, n: int) -> Tensor:
    """Compute and return accumulated matrix powers [A, A^2, ..., A^n].

    If n == 0 the identity is returned (with a singleton -3 dimension).
    Negative `n` values compute powers of the matrix inverse.

    Args:
        A (Tensor): Square matrix of shape (..., N, N).
        n (int): Exponent (may be negative).

    Returns:
        Tensor: Accumulated powers with shape (..., K, N, N) where K == max(n, 1).
    """
    assert A.dim() >= 2, "Input matrix A must have at least 2 dimensions."
    assert A.size(-2) == A.size(-1), "Input matrix A must be square."

    if n == 0:
        return (
            torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
            .broadcast_to(A.shape)
            .unsqueeze(-3)
        )
    elif n < 0:
        Ainv = torch.linalg.inv(A)
        return matrix_power_accumulate(Ainv, -n)
    elif n == 1:
        return A.unsqueeze(-3)

    factors = factorint(n, multiple=True)
    return _mat_pwr_accum_runner(A, factors)


def _mat_pwr_accum_runner(A: Tensor, factors: list[int]) -> Tensor:
    fac, *factors = factors
    accums = torch.stack(list(accumulate([A] * fac, torch.matmul)), dim=-3)
    if len(factors) == 0:
        return accums
    higher_powers = _mat_pwr_accum_runner(accums[..., -1, :, :], factors)
    tmp = accums[..., None, :-1, :, :] @ higher_powers[..., :-1, None, :, :]
    return torch.cat(
        [
            accums,
            torch.cat([tmp, higher_powers[..., 1:, None, :, :]], dim=-3).flatten(
                -4, -3
            ),
        ],
        dim=-3,
    )


def matrices_cumdot(A: Tensor) -> Tensor:
    """Compute the cumulative dot product of matrices along the last third dimension.
    Given a sequence of matrices [A_1, A_2, ..., A_M], this function returns
    [A_1, A_1 @ A_2, A_1 @ A_2 @ A_3, ..., A_1 @ A_2 @ ... @ A_M].
    Args:
        A (Tensor): Input tensor of shape (..., M, N, N) where ... can be any number of batch dimensions.
    Returns:
        Tensor: Cumulative dot product of matrices with shape (..., M, N, N).
    """
    assert A.dim() >= 3, "Input tensor A must have at least 3 dimensions."
    assert A.size(-2) == A.size(-1), "Input tensor A must have square matrices."

    M = A.size(-3)
    if M == 1:
        return A
    leading_dims = len(A.shape) - 3
    factors = factorint(M, multiple=True)[::-1]
    unfolded_A = A.unflatten(-3, factors)
    return _mat_cumdot_runner(unfolded_A, leading_dims).flatten(leading_dims, -3)


def _mat_cumdot_runner(A: Tensor, leading_dims: int) -> Tensor:
    accums = torch.stack(list(accumulate(A.unbind(-3), torch.matmul)), dim=-3)
    if A.dim() == leading_dims + 3:
        return accums

    higher_powers = _mat_cumdot_runner(accums[..., -1, :, :], leading_dims)
    tmp = accums[..., 1:, :-1, :, :] @ higher_powers[..., :-1, None, :, :]
    return torch.cat(
        [
            accums[..., :1, :, :, :],
            torch.cat([tmp, higher_powers[..., 1:, None, :, :]], dim=-3),
        ],
        dim=-4,
    )
