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
    power = torch.tensor([1, 2, 3, 1, 1, 2])
    V = confluent_vandermonde(p.to(dtype), power)
    M = p.numel()
    expected = torch.zeros(M, M, dtype=p.dtype)
    for c, (pc, jc) in enumerate(zip(p.tolist(), power.tolist())):
        for i in range(M):
            n = M - 1 - i
            if n >= jc - 1:
                expected[i, c] = comb(n, jc - 1) * pc ** (n - jc + 1)
    assert V.dtype == dtype
    low = dtype in (torch.bfloat16, torch.complex32)
    tol = dict(rtol=1e-2, atol=1e-2) if low else {}
    torch.testing.assert_close(V.to(p.dtype), expected, **tol)


@pytest.mark.parametrize("batched_power", [False, True])
def test_confluent_vandermonde_batched(batched_power):
    poles = torch.tensor([[0.5, 0.5, -0.3], [0.2, 0.9, 0.9]], dtype=torch.float64)
    power = torch.tensor([[1, 2, 1], [1, 1, 2]]) if batched_power else torch.tensor([1, 2, 1])
    V = confluent_vandermonde(poles, power)
    expected = torch.stack(
        [confluent_vandermonde(p, j) for p, j in zip(poles, power.expand_as(poles))]
    )
    torch.testing.assert_close(V, expected)


def test_confluent_vandermonde_unsigned_powers():
    # powers of an unsigned dtype once wrapped when negated
    poles = torch.tensor([0.5, 0.5, -0.3], dtype=torch.float64)
    power = torch.tensor([1, 2, 1])
    expected = confluent_vandermonde(poles, power)
    torch.testing.assert_close(confluent_vandermonde(poles, power.to(torch.uint8)), expected)
