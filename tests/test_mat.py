from itertools import accumulate
from math import comb

import numpy as np
import pytest
import torch

from philtorch.mat import companion, confluent_vandermonde, find_eigenvectors, matrices_cumdot


# Primes, products of two primes, and products of three or more primes, which
# matrices_cumdot splits into nested groups.
@pytest.mark.parametrize("M", [1, 2, 3, 4, 6, 7, 8, 9, 12, 16, 18, 30, 64])
def test_matrices_cumdot(M: int):
    torch.manual_seed(0)
    A = torch.randn(2, 3, M, 2, 2, dtype=torch.float64) / 1.5

    expected = torch.stack(list(accumulate(A.unbind(-3), torch.matmul)), dim=-3)

    assert torch.allclose(matrices_cumdot(A), expected)


# bfloat16 and complex32 run in float32 and complex64, as CPU's cumprod takes
# no complex32
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.complex128, torch.bfloat16, torch.complex32]
)
def test_confluent_vandermonde(dtype):
    poles = torch.tensor([0.9, -0.3, 0.0], dtype=torch.float64)
    poles = poles * (1 + 0.5j) if dtype.is_complex else poles
    poles = poles.to(dtype).to(poles.dtype)  # the reference takes them as rounded
    multiplicity = torch.tensor([3, 1, 2])
    V = confluent_vandermonde(poles.to(dtype), multiplicity)
    p = poles.repeat_interleave(multiplicity)
    order = torch.cat([torch.arange(m) for m in multiplicity.tolist()])
    M = p.numel()
    expected = torch.zeros(M, M, dtype=p.dtype)
    for c, (pc, ec) in enumerate(zip(p.tolist(), order.tolist())):
        for i in range(M):
            n = M - 1 - i
            if n >= ec:
                expected[i, c] = comb(n, ec) * pc ** (n - ec)
    assert V.dtype == dtype
    low = dtype in (torch.bfloat16, torch.complex32)
    tol = dict(rtol=1e-2, atol=1e-2) if low else {}
    torch.testing.assert_close(V.to(p.dtype), expected, **tol)


def test_confluent_vandermonde_batched():
    poles = torch.tensor([[0.5, -0.3], [0.2, 0.9]], dtype=torch.float64)
    multiplicity = torch.tensor([2, 1])
    V = confluent_vandermonde(poles, multiplicity)
    expected = torch.stack([confluent_vandermonde(p, multiplicity) for p in poles])
    torch.testing.assert_close(V, expected)


def test_confluent_vandermonde_companion():
    poles = torch.tensor([0.5, -0.3], dtype=torch.float64)
    multiplicity = torch.tensor([2, 1])
    a = np.poly(poles.repeat_interleave(multiplicity).numpy())[1:]
    V = confluent_vandermonde(poles, multiplicity)
    J = torch.block_diag(
        *[
            p * torch.eye(m, dtype=torch.float64) + torch.diag(torch.ones(m - 1), 1)
            for p, m in zip(poles.tolist(), multiplicity.tolist())
        ]
    )
    C = companion(torch.from_numpy(a))
    torch.testing.assert_close(C @ V, V @ J)


def test_find_eigenvectors_raises_without_a_first_component():
    # the eigenvector of 0.3 has a zero first component
    eigenvalues = torch.tensor([0.5, 0.3], dtype=torch.float64)
    with pytest.raises(torch.linalg.LinAlgError, match="first component"):
        find_eigenvectors(torch.diag(eigenvalues), eigenvalues)
