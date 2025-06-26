import torch


def find_eigenvectors(A: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
    assert A.dim() >= 2, "Matrix A must be at least 2D."
    assert eigenvalues.dim() >= 1, "Eigenvalues must be at least 1D."
    assert A.shape[-2] == A.shape[-1], "Matrix A must be square."
    assert A.shape[-1] == eigenvalues.shape[-1], "Eigenvalues must match the size of A."

    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    W = A.unsqueeze(-3) - eigenvalues[..., None, None] * I
    B, W = torch.split(W, [1, n - 1], dim=-1)

    WtW = W.mT.conj() @ W
    WtB = W.mT.conj() @ -B

    X = torch.linalg.solve(WtW, WtB)
    X = X.reshape(A.shape[:-2] + (n, n - 1))
    X = torch.cat([X.new_ones(X.shape[:-1] + (1,)), X], dim=-1).mT
    X = X / torch.linalg.matrix_norm(X, dim=(-2, -1), keepdim=True)
    return X


def a2companion(a: torch.Tensor) -> torch.Tensor:
    """
    Convert all-pole coefficients to companion matrix.

    Args:
        a (torch.Tensor): All-pole coefficients of shape (..., M).

    Returns:
        torch.Tensor: Companion matrix of shape (..., M, M).
    """
    assert a.dim() >= 1, "All-pole coefficients must be at least 1D."
    M = a.shape[-1]
    A = torch.cat([-a, a.new_zeros(a.shape[:-1] + (M * (M - 1),))], dim=-1).unflatten(
        -1, (M, M)
    )
    c = A + torch.diag(a.new_ones(M - 1), diagonal=-1)
    return c


def vandermonde(poles: torch.Tensor) -> torch.Tensor:
    """
    Create a Vandermonde matrix from poles.

    Args:
        poles (torch.Tensor): Poles of shape (..., M).

    Returns:
        torch.Tensor: Vandermonde matrix of shape (..., M, M).
    """
    assert poles.dim() >= 1, "Poles must be at least 1D."
    M = poles.shape[-1]
    vander = torch.stack([poles**i for i in range(M - 1, -1, -1)], dim=-2)
    return vander
