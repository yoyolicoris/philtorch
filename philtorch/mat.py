"""Batched matrix helpers for building and evaluating state-space filters."""

from itertools import accumulate

import torch
from sympy.ntheory import factorint
from torch import Tensor


def find_eigenvectors(A: Tensor, eigenvalues: Tensor) -> Tensor:
    r"""Compute unit-norm eigenvectors of square matrices from eigenvalues.

    For each eigenvalue :math:`\lambda`, this solves
    :math:`(A - \lambda I) \mathbf{v} = \mathbf{0}` by least squares with the
    first component of :math:`\mathbf{v}` fixed to 1, then scales
    :math:`\mathbf{v}` to unit 2-norm.

    Args:
        A (Tensor): square matrices of shape :math:`(*, N, N)`, where :math:`*`
            is zero or more batch dimensions.
        eigenvalues (Tensor): eigenvalues of :attr:`A`, of shape
            :math:`(*, N)`. Their batch dimensions must broadcast to those of
            :attr:`A`.

    Returns:
        Tensor: the eigenvectors, of shape :math:`(*, N, N)`. Column :math:`k`
        is the eigenvector for ``eigenvalues[..., k]``, and its first
        component is real and positive. The dtype is the promotion of the
        inputs' dtypes, so a real :attr:`A` with complex eigenvalues gives
        complex eigenvectors.

    Raises:
        AssertionError: if :attr:`A` is not square or the number of
            eigenvalues does not match its size.
        torch.linalg.LinAlgError: if an eigenvector's first component is zero,
            or an eigenvalue has more than one independent eigenvector.

    Example::

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
    r"""Return the companion matrices of monic polynomials.

    For coefficients :math:`a_1, \dots, a_M`, this returns

    .. math::
        C = \begin{bmatrix}
            -a_1 & -a_2 & \cdots & -a_{M-1} & -a_M \\
            1 & 0 & \cdots & 0 & 0 \\
            0 & 1 & \cdots & 0 & 0 \\
            \vdots & \vdots & \ddots & \vdots & \vdots \\
            0 & 0 & \cdots & 1 & 0
        \end{bmatrix}.

    Its eigenvalues are the roots of :math:`z^M + a_1 z^{M-1} + \cdots + a_M`,
    which are the poles of the all-pole filter
    :math:`1 / (1 + a_1 z^{-1} + \cdots + a_M z^{-M})`, so :math:`C` is the
    state matrix of that filter in state-space form.

    Args:
        a (Tensor): polynomial coefficients without the leading 1, of shape
            :math:`(*, M)`. This is the ``a`` that
            :func:`philtorch.lti.lfilter` takes, which omits SciPy's leading
            ``a[0] = 1``.

    Returns:
        Tensor: the companion matrices, of shape :math:`(*, M, M)`, with the
        dtype and device of :attr:`a`.

    Example::

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
    r"""Return the Vandermonde matrix of descending powers of poles.

    For poles :math:`p_0, \dots, p_{M-1}`, this returns

    .. math::
        V = \begin{bmatrix}
            p_0^{M-1} & p_1^{M-1} & \cdots & p_{M-1}^{M-1} \\
            \vdots & \vdots & & \vdots \\
            p_0 & p_1 & \cdots & p_{M-1} \\
            1 & 1 & \cdots & 1
        \end{bmatrix},

    the transpose of :func:`torch.vander`. Its columns are the eigenvectors of
    the :func:`companion` matrix :math:`C` whose roots are the poles, so
    :math:`C V = V \operatorname{diag}(p_0, \dots, p_{M-1})`.

    Args:
        poles (Tensor): the poles, of shape :math:`(M)`. Batched poles are not
            supported.

    Returns:
        Tensor: the Vandermonde matrix, of shape :math:`(M, M)`, with the dtype
        and device of :attr:`poles`.

    Example::

        >>> from philtorch.mat import companion, vandermonde
        >>> poles = torch.tensor([0.5, -0.25])
        >>> V = vandermonde(poles)
        >>> V
        tensor([[ 0.5000, -0.2500],
                [ 1.0000,  1.0000]])
        >>> # 1 - 0.25 z^-1 - 0.125 z^-2 has poles 0.5 and -0.25.
        >>> C = companion(torch.tensor([-0.25, -0.125]))
        >>> torch.allclose(C @ V, V @ torch.diag(poles))
        True
    """
    if poles.size(-1) == 1:
        return torch.ones_like(poles).unsqueeze(-1)
    return torch.vander(poles).mT


def matrix_power_accumulate(A: Tensor, n: int) -> Tensor:
    r"""Return the powers of square matrices up to :math:`A^n`.

    For :math:`n > 0`, this returns :math:`A, A^2, \dots, A^n` stacked along a
    new dimension. Its longest chain of dependent matrix products is one
    shorter than the sum of :math:`|n|`'s prime factors, rather than the
    :math:`|n| - 1` products of computing one power after another.

    Args:
        A (Tensor): square matrices of shape :math:`(*, N, N)`, where :math:`*`
            is zero or more batch dimensions.
        n (int): the highest power. A negative :attr:`n` returns the powers of
            the inverse, :math:`A^{-1}, \dots, A^{n}`, and :math:`n = 0` returns
            the identity.

    Returns:
        Tensor: the powers, of shape :math:`(*, K, N, N)`, where
        :math:`K = |n|`, or 1 when :math:`n = 0`. Entry ``[..., k, :, :]`` is
        :math:`A^{k+1}`, or :math:`A^{-(k+1)}` for a negative :attr:`n`. The
        dtype and device are those of :attr:`A`.

    Raises:
        AssertionError: if :attr:`A` is not a batch of square matrices.

    Example::

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
    r"""Return the running products of a sequence of matrices.

    For matrices :math:`A_1, \dots, A_M` along dimension ``-3``, this computes

    .. math::
        P_k = A_1 A_2 \cdots A_k, \quad k = 1, \dots, M,

    multiplying each new matrix on the right. Its longest chain of dependent
    matrix products is one shorter than the sum of :math:`M`'s prime factors,
    rather than the :math:`M - 1` products of a running loop.

    Args:
        A (Tensor): sequences of square matrices of shape :math:`(*, M, N, N)`,
            where :math:`*` is zero or more batch dimensions.

    Returns:
        Tensor: the running products :math:`P_1, \dots, P_M`, of shape
        :math:`(*, M, N, N)`, with the dtype and device of :attr:`A`.

    Raises:
        AssertionError: if :attr:`A` is not a batch of square-matrix sequences.

    Example::

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
    return _mat_cumdot_runner(A, factorint(M, multiple=True))


def _mat_cumdot_runner(A: Tensor, factors: list[int]) -> Tensor:
    group, *factors = factors
    if not factors:
        return torch.stack(list(accumulate(A.unbind(-3), torch.matmul)), dim=-3)

    # Split the M matrices into consecutive groups of `group`, take products
    # within each group, then prefix the products of all earlier groups. The
    # earlier groups multiply from the left, as the matrices don't commute.
    A = A.unflatten(-3, (-1, group))
    accums = torch.stack(list(accumulate(A.unbind(-3), torch.matmul)), dim=-3)
    totals = _mat_cumdot_runner(accums[..., -1, :, :], factors)
    tmp = totals[..., :-1, None, :, :] @ accums[..., 1:, :-1, :, :]
    return torch.cat(
        [
            accums[..., :1, :, :, :],
            torch.cat([tmp, totals[..., 1:, None, :, :]], dim=-3),
        ],
        dim=-4,
    ).flatten(-4, -3)
