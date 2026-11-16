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


# bfloat16 and complex32 run in float32 and complex64, as CPU's cumprod takes
# no complex32
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.complex128, torch.bfloat16, torch.complex32]
)
def test_confluent_vandermonde(dtype):
    p = torch.tensor([0.9, 0.9, 0.9, -0.3, 0.0, 0.0], dtype=torch.float64)
    p = p * (1 + 0.5j) if dtype.is_complex else p
    p = p.to(dtype).to(p.dtype)  # the reference takes the poles as rounded
    e = torch.tensor([0, 1, 2, 0, 0, 1])
    V = confluent_vandermonde(p.to(dtype), e)
    M = p.numel()
    expected = torch.zeros(M, M, dtype=p.dtype)
    for c, (pc, ec) in enumerate(zip(p.tolist(), e.tolist())):
        for i in range(M):
            n = M - 1 - i
            if n >= ec:
                expected[i, c] = comb(n, ec) * pc ** (n - ec)
    assert V.dtype == dtype
    low = dtype in (torch.bfloat16, torch.complex32)
    tol = dict(rtol=1e-2, atol=1e-2) if low else {}
    torch.testing.assert_close(V.to(p.dtype), expected, **tol)


@pytest.mark.parametrize("batched_e", [False, True])
def test_confluent_vandermonde_batched(batched_e):
    poles = torch.tensor([[0.5, 0.5, -0.3], [0.2, 0.9, 0.9]], dtype=torch.float64)
    e = torch.tensor([[0, 1, 0], [0, 0, 1]]) if batched_e else torch.tensor([0, 1, 0])
    V = confluent_vandermonde(poles, e)
    expected = torch.stack(
        [confluent_vandermonde(p, ei) for p, ei in zip(poles, e.expand_as(poles))]
    )
    torch.testing.assert_close(V, expected)
