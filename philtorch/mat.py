"""Batched matrix helpers for building and evaluating state-space filters."""

from itertools import accumulate

import torch
from sympy.ntheory import factorint
from torch import Tensor

from .utils import _accumulation_dtype, _check_integer


def find_eigenvectors(A: Tensor, eigenvalues: Tensor) -> Tensor:
    r"""Compute unit-norm eigenvectors of square matrices from eigenvalues.

    For each eigenvalue :math:`\lambda`, this returns the solution
    :math:`\mathbf{v}` of :math:`(A - \lambda I) \mathbf{v} = \mathbf{0}`
    that has unit 2-norm and a real, positive first component.

    Args:
        A (Tensor): square matrices of shape :math:`(*, N, N)`, where :math:`*`
            is zero or more batch dimensions.
        eigenvalues (Tensor): eigenvalues of :attr:`A`, of shape
            :math:`(*, N)`. Their batch dimensions must broadcast to those of
            :attr:`A`.

    Returns:
        Tensor: the eigenvectors, of shape :math:`(*, N, N)`. Column :math:`k`
        is the eigenvector for ``eigenvalues[..., k]``. The dtype is the
        promotion of the inputs' dtypes, so a real :attr:`A` with complex
        eigenvalues gives complex eigenvectors.

    Raises:
        ValueError: if :attr:`A` is not square or the number of
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
    if A.dim() < 2:
        raise ValueError("Matrix A must be at least 2D.")
    if eigenvalues.dim() < 1:
        raise ValueError("Eigenvalues must be at least 1D.")
    if A.size(-2) != A.size(-1):
        raise ValueError("Matrix A must be square.")
    if A.size(-1) != eigenvalues.size(-1):
        raise ValueError("Eigenvalues must match the size of A.")

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

    Raises:
        ValueError: if :attr:`a` is 0-D.

    Example::

        >>> from philtorch.mat import companion
        >>> companion(torch.tensor([-0.25, -0.125]))
        tensor([[0.2500, 0.1250],
                [1.0000, 0.0000]])
    """
    if a.dim() < 1:
        raise ValueError("All-pole coefficients must be at least 1D.")
    M = a.size(-1)
    c = torch.cat([-a, a.new_zeros(a.shape[:-1] + (M * (M - 1),))], dim=-1).unflatten(-1, (M, M))
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


def confluent_vandermonde(poles: Tensor, e: Tensor) -> Tensor:
    r"""Return the confluent Vandermonde matrix of descending powers of poles.

    For :math:`M` poles :math:`p_k` with orders :math:`e_k`, row
    :math:`i = 0, \dots, M - 1` and column :math:`k = 0, \dots, M - 1` hold

    .. math::
        V_{ik} = \binom{n}{e_k} p_k^{\,n - e_k}, \quad n = M - 1 - i,

    and entries with :math:`n < e_k` are zero. Column :math:`k` is
    :math:`\frac{1}{e_k!} \frac{d^{e_k}}{dx^{e_k}} (x^{M-1}, \dots, x, 1)`
    at :math:`x = p_k`. A column of order 0 is the column of
    :func:`vandermonde`, so with every order 0, :math:`V` is
    :func:`vandermonde` of the poles.

    List a pole of multiplicity :math:`m` in :math:`m` adjacent columns, with
    the orders :math:`0, 1, \dots, m - 1`. Then :math:`C V = V J`, where
    :math:`C` is the :func:`companion` matrix of the :math:`a` with
    :math:`1 + \sum_{k=1}^{M} a_k z^{-k} = \prod_{k=0}^{M-1} (1 - p_k z^{-1})`,
    whose eigenvalues are the poles, and :math:`J` is the Jordan form in the
    columns' order: :math:`J_{kk} = p_k`, :math:`J_{k-1, k} = 1` where
    :math:`e_k \ge 1`, and zero elsewhere.

    Args:
        poles (Tensor): the poles :math:`p_k`, real or complex, of shape
            :math:`(*, M)`.
        e (Tensor): the orders :math:`e_k \ge 0`, an integer tensor of shape
            :math:`(M)` or the shape of :attr:`poles`.

    Returns:
        Tensor: the matrix, of shape :math:`(*, M, M)`, with the dtype and
        device of :attr:`poles`.

    Raises:
        ValueError: if :attr:`e` is not an integer tensor, is not on the
            device of :attr:`poles`, has neither shape :math:`(M)` nor that of
            :attr:`poles`, or has a negative entry.

    Example::

        >>> from philtorch.mat import companion, confluent_vandermonde
        >>> # a double pole at 0.5: a = [-1, 0.25], as
        >>> # 1 - z^-1 + 0.25 z^-2 = (1 - 0.5 z^-1)^2
        >>> poles = torch.tensor([0.5, 0.5])
        >>> V = confluent_vandermonde(poles, torch.tensor([0, 1]))
        >>> V
        tensor([[0.5000, 1.0000],
                [1.0000, 0.0000]])
        >>> C = companion(torch.tensor([-1.0, 0.25]))
        >>> J = torch.tensor([[0.5, 1.0], [0.0, 0.5]])
        >>> torch.allclose(C @ V, V @ J)
        True
    """
    _check_integer("e", e)
    if e.device != poles.device:
        raise ValueError(f"e must be on poles's device, got {e.device}.")
    if e.shape not in (poles.shape[-1:], poles.shape):
        raise ValueError(
            f"e must be of shape {tuple(poles.shape[-1:])} or {tuple(poles.shape)}, "
            f"got {tuple(e.shape)}."
        )
    if bool((e < 0).any()):
        raise ValueError("e must not be negative.")
    # float16 and bfloat16 accumulate in float32, complex32 in complex64
    acc = torch.complex64 if poles.dtype == torch.complex32 else _accumulation_dtype(poles.dtype)
    V = _confluent_vandermonde(poles.to(acc), e, poles.size(-1))
    return V.flip(-2).to(poles.dtype)


def _confluent_vandermonde(p: Tensor, e: Tensor, rows: int, start: Tensor | None = None) -> Tensor:
    r"""The confluent Vandermonde matrix of ascending powers, (*, rows, M).

    Entry [i, c] is :math:`\binom{n}{e_c} p_c^{n - e_c}`, zero where
    :math:`n < e_c`, with :math:`n = i + s_c` and ``start`` :math:`s_c` from 0
    (the default) to :math:`e_c`. With ``start = e``, column c is the impulse
    response of :math:`1 / (1 - p_c z^{-1})^{e_c + 1}`, which ``residuez`` and
    ``invresz`` use. A cumulative product builds each column from the ratios
    of consecutive rows, which keeps the binomials exact up to rounding.
    """
    # t = n - e runs from start - e <= 0; the ratio of row t to row t - 1 is
    # p (t + e) / t from t = 1 on, and 1 before, then t < 0 is masked out
    shift = -e if start is None else start - e
    n = torch.arange(rows, dtype=p.dtype.to_real(), device=p.device)
    t = n.unsqueeze(-1) + shift.unsqueeze(-2)
    ratio = p.unsqueeze(-2) * ((t + e.unsqueeze(-2)) / t.clamp(min=1))
    ratio = torch.where(t >= 1, ratio, 1)
    return torch.cumprod(ratio, dim=-2) * (t >= 0)


def matrix_power_accumulate(A: Tensor, n: int) -> Tensor:
    r"""Return the powers of square matrices up to :math:`A^n`.

    For :math:`n > 0`, this returns :math:`A, A^2, \dots, A^n` stacked along a
    new dimension.

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
        ValueError: if :attr:`A` is not a batch of square matrices.

    Example::

        >>> from philtorch.mat import matrix_power_accumulate
        >>> A = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
        >>> matrix_power_accumulate(A, 3)[:, 0, 1]
        tensor([1., 2., 3.])
    """
    if A.dim() < 2:
        raise ValueError("Input matrix A must have at least 2 dimensions.")
    if A.size(-2) != A.size(-1):
        raise ValueError("Input matrix A must be square.")

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
        P_k = A_1 A_2 \cdots A_k, \quad k = 1, \dots, M.

    Args:
        A (Tensor): sequences of square matrices of shape :math:`(*, M, N, N)`,
            where :math:`*` is zero or more batch dimensions.

    Returns:
        Tensor: the running products :math:`P_1, \dots, P_M`, of shape
        :math:`(*, M, N, N)`, with the dtype and device of :attr:`A`.

    Raises:
        ValueError: if :attr:`A` is not a batch of square-matrix sequences.

    Example::

        >>> from philtorch.mat import matrices_cumdot
        >>> A = torch.randn(6, 2, 2, dtype=torch.float64)
        >>> P = matrices_cumdot(A)
        >>> torch.allclose(P[3], A[0] @ A[1] @ A[2] @ A[3])
        True
    """
    if A.dim() < 3:
        raise ValueError("Input tensor A must have at least 3 dimensions.")
    if A.size(-2) != A.size(-1):
        raise ValueError("Input tensor A must have square matrices.")

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
