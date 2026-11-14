from itertools import accumulate
from math import comb

import pytest
import torch

from philtorch.mat import confluent_vandermonde, matrices_cumdot


# Primes, products of two primes, and products of three or more primes, which
# matrices_cumdot splits into nested groups.
@pytest.mark.parametrize("M", [1, 2, 3, 4, 6, 7, 8, 9, 12, 16, 18, 30, 64])
def test_matrices_cumdot(M: int):
    torch.manual_seed(0)
    A = torch.randn(2, 3, M, 2, 2, dtype=torch.float64) / 1.5

    expected = torch.stack(list(accumulate(A.unbind(-3), torch.matmul)), dim=-3)

    assert torch.allclose(matrices_cumdot(A), expected)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("from_e", [False, True])
def test_confluent_vandermonde(dtype, from_e):
    p = torch.tensor([0.5, 0.5, 0.5, -0.3, 0.0, 0.0], dtype=torch.float64)
    p = p * (1 + 0.5j) if dtype.is_complex else p
    e = torch.tensor([0, 1, 2, 0, 0, 1])
    start = e if from_e else None
    V = confluent_vandermonde(p, e, 6, start=start)
    expected = torch.zeros(6, 6, dtype=dtype)
    for c, (pc, ec) in enumerate(zip(p.tolist(), e.tolist())):
        for i in range(6):
            n = i + (ec if from_e else 0)
            if n >= ec:
                expected[i, c] = comb(n, ec) * pc ** (n - ec)
    torch.testing.assert_close(V, expected)
