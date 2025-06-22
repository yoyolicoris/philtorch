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
