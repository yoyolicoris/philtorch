from itertools import accumulate

import pytest
import torch

from philtorch.mat import matrices_cumdot


# Primes, products of two primes, and products of three or more primes, which
# matrices_cumdot splits into nested groups.
@pytest.mark.parametrize("M", [1, 2, 3, 4, 6, 7, 8, 9, 12, 16, 18, 30, 64])
def test_matrices_cumdot(M: int):
    torch.manual_seed(0)
    A = torch.randn(2, 3, M, 2, 2, dtype=torch.float64) / 1.5

    expected = torch.stack(list(accumulate(A.unbind(-3), torch.matmul)), dim=-3)

    assert torch.allclose(matrices_cumdot(A), expected)
