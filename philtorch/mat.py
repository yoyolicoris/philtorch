"""Batched matrix helpers for building and evaluating state-space filters."""

from itertools import accumulate

import torch
from sympy.ntheory import factorint
from torch import Tensor


def find_eigenvectors(A: Tensor, eigenvalues: Tensor) -> Tensor:
    """Compute unit-norm eigenvectors of square matrices from known eigenvalues.

    For each eigenvalue ``lam``, this solves ``(A - lam * I) v = 0`` by least
    squares with the first component of ``v`` fixed to 1, then scales ``v`` to
    unit 2-norm.

    Args:
        A (Tensor): Square matrices with shape ``(..., N, N)``.
        eigenvalues (Tensor): Eigenvalues of ``A`` with shape ``(..., N)``. Their
            batch dimensions must broadcast to those of ``A``.

    Returns:
        Tensor: Eigenvectors with shape ``(..., N, N)``. Column ``k`` is the
        eigenvector for ``eigenvalues[..., k]``, and its first component is real
        and positive. The dtype is the promotion of the inputs' dtypes, so a real
        ``A`` with complex eigenvalues gives complex eigenvectors.

    Raises:
        AssertionError: If ``A`` is not square or the number of eigenvalues does
            not match its size.
        torch.linalg.LinAlgError: If an eigenvector's first component is zero,
            or an eigenvalue has more than one independent eigenvector.

    Example:
        >>> import torch
        >>> from philtorch.mat import find_eigenvectors
        >>> A = torch.tensor([[2.0, 1.0], [0.0, 1.0]])
        >>> find_eigenvectors(A, torch.tensor([2.0, 1.0]))
        tensor([[ 1.0000,  0.7071],
                [ 0.0000, -0.7071]])
    """
    assert A.dim() >= 2, "Matrix A must be at least 2D."
    assert eigenvalues.dim() >= 1, "Eigenvalues must be at least 1D."
    assert A.size(-2) == A.size(-1), "Matrix A must be square."
    assert A.size(-1) == eigenvalues.size(-1), "Eigenvalues must match the size of A."

    n = A.size(-1)
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    W = A.unsqueeze(-3) - eigenvalues[..., None, None] * eye
    B, W = torch.split(W, [1, n - 1], dim=-1)

    WtW = W.mT.conj() @ W
    WtB = W.mT.conj() @ -B

    X = torch.linalg.solve(WtW, WtB)
    X = X.reshape(A.shape[:-2] + (n, n - 1))
    X = torch.cat([X.new_ones(X.shape[:-1] + (1,)), X], dim=-1)
    X = X / torch.linalg.vector_norm(X, dim=-1, keepdim=True)
    return X.mT


def companion(a: Tensor) -> Tensor:
    """Return the companion matrices of monic polynomials.

    For ``a = [a_1, ..., a_M]``, the first row is ``[-a_1, ..., -a_M]``, the
    subdiagonal is all ones, and every other entry is zero. Its eigenvalues are
    the roots of ``z^M + a_1 z^(M-1) + ... + a_M``, which are the poles of the
    all-pole filter ``1 / (1 + a_1 z^-1 + ... + a_M z^-M)``. That makes it the
    state matrix of the filter in state-space form.

    Args:
        a (Tensor): Polynomial coefficients without the leading 1, with shape
            ``(..., M)``. This is the ``a`` that :func:`philtorch.lti.lfilter`
            takes, which omits SciPy's leading ``a[0] = 1``.

    Returns:
        Tensor: Companion matrices with shape ``(..., M, M)``, with the dtype and
        device of ``a``.

    Example:
        >>> import torch
        >>> from philtorch.mat import companion
        >>> companion(torch.tensor([-0.25, -0.125]))
        tensor([[0.2500, 0.1250],
                [1.0000, 0.0000]])
    """
    assert a.dim() >= 1, "All-pole coefficients must be at least 1D."
    M = a.size(-1)
    c = torch.cat([-a, a.new_zeros(a.shape[:-1] + (M * (M - 1),))], dim=-1).unflatten(-1, (M, M))
    # c = A + torch.diag(a.new_ones(M - 1), diagonal=-1)
    c[..., list(range(1, M)), list(range(M - 1))] = 1
    return c


def vandermonde(poles: Tensor) -> Tensor:
    """Return the Vandermonde matrix whose columns are descending powers of poles.

    For poles ``p = [p_0, ..., p_(M-1)]``, column ``j`` is
    ``[p_j^(M-1), ..., p_j, 1]``. These columns are the eigenvectors of the
    :func:`companion` matrix with those poles as roots, so the matrix
    diagonalizes it. The result is the transpose of :func:`torch.vander`.

    Args:
        poles (Tensor): Poles with shape ``(M,)``. Batched poles are not
            supported.

    Returns:
        Tensor: Vandermonde matrix with shape ``(M, M)``, with the dtype and
        device of ``poles``.

    Example:
        >>> import torch
        >>> from philtorch.mat import companion, vandermonde
        >>> poles = torch.tensor([0.5, -0.25])
        >>> V = vandermonde(poles)
        >>> V
        tensor([[ 0.5000, -0.2500],
                [ 1.0000,  1.0000]])
        >>> # 1 - 0.25 z^-1 - 0.125 z^-2 has poles 0.5 and -0.25.
        >>> A = companion(torch.tensor([-0.25, -0.125]))
        >>> torch.allclose(A @ V, V @ torch.diag(poles))
        True
    """
    if poles.size(-1) == 1:
        return torch.ones_like(poles).unsqueeze(-1)
    return torch.vander(poles).mT


def matrix_power_accumulate(A: Tensor, n: int) -> Tensor:
    """Return the powers ``A, A^2, ..., A^n`` stacked along a new dimension.

    The powers are computed in as many sequential matrix-product steps as the sum
    of ``abs(n)``'s prime factors, rather than ``abs(n)`` steps.

    Args:
        A (Tensor): Square matrices with shape ``(..., N, N)``.
        n (int): Highest power. A negative ``n`` returns the powers of ``A``'s
            inverse, ``A^-1, ..., A^n``, and ``n = 0`` returns the identity.

    Returns:
        Tensor: Powers with shape ``(..., K, N, N)``, where ``K = abs(n)``, or 1 when
        ``n = 0``. Entry ``[..., k, :, :]`` is ``A^(k+1)``, or ``A^-(k+1)`` for a
        negative ``n``. The dtype and device are those of ``A``.

    Raises:
        AssertionError: If ``A`` is not a batch of square matrices.

    Example:
        >>> import torch
        >>> from philtorch.mat import matrix_power_accumulate
        >>> A = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        >>> matrix_power_accumulate(A, 3)[:, 0, 1]
        tensor([1., 2., 3.])
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
            torch.cat([tmp, higher_powers[..., 1:, None, :, :]], dim=-3).flatten(-4, -3),
        ],
        dim=-3,
    )


def matrices_cumdot(A: Tensor) -> Tensor:
    """Return the running products ``A_1, A_1 A_2, ..., A_1 A_2 ... A_M``.

    Each product multiplies the next matrix on the right. The products are
    computed in as many sequential matrix-product steps as the sum of ``M``'s
    prime factors, rather than ``M`` steps.

    Args:
        A (Tensor): Sequences of square matrices with shape ``(..., M, N, N)``,
            where dimension ``-3`` runs over the sequence.

    Returns:
        Tensor: Running products with shape ``(..., M, N, N)``. Entry
        ``[..., k, :, :]`` is ``A_1 ... A_(k+1)``. The dtype and device are those
        of ``A``.

    Raises:
        AssertionError: If ``A`` is not a batch of square-matrix sequences.

    Example:
        >>> import torch
        >>> from philtorch.mat import matrices_cumdot
        >>> A = torch.randn(6, 2, 2, dtype=torch.float64)
        >>> P = matrices_cumdot(A)
        >>> torch.allclose(P[3], A[0] @ A[1] @ A[2] @ A[3])
        True
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
    # Flatten the group dimensions so the group before group g is g - 1 in
    # sequence order, and prefix each group's partial products with the
    # product of all earlier groups, from the left as matrices don't commute.
    accums = accums.flatten(leading_dims, -4)
    higher_powers = higher_powers.flatten(leading_dims, -3)
    tmp = higher_powers[..., :-1, None, :, :] @ accums[..., 1:, :-1, :, :]
    return torch.cat(
        [
            accums[..., :1, :, :, :],
            torch.cat([tmp, higher_powers[..., 1:, None, :, :]], dim=-3),
        ],
        dim=-4,
    ).reshape(A.shape)
