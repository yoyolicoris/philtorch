import pytest
import torch

from philtorch.lpv.utils import diag_shift


@pytest.mark.parametrize("M", [1, 3])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("discard_end", [False, True])
def test_diag_shift(M: int, offset: int, discard_end: bool):
    T = 5
    coef = torch.arange(1, 2 * T * M + 1, dtype=torch.float64).reshape(2, T, M)
    out = diag_shift(coef, offset=offset, discard_end=discard_end)

    length = T if discard_end else T + M + offset - 1
    assert out.shape == (2, length, M)
    for n in range(length):
        for k in range(M):
            src = n - k - offset
            expected = coef[:, src, k] if 0 <= src < T else torch.zeros(2, dtype=torch.float64)
            assert torch.equal(out[:, n, k], expected)
