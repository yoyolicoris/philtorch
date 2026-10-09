import pytest
import torch

from philtorch.prototype.dtw import dtw, dtw_fused, dtw_rowwise

# dtw runs Triton kernels, so it needs CUDA.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

STEP_PATTERNS = ["symmetric", "asymmetric", "orthogonal"]


def _cost(batch_size, N, M, *, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(batch_size, N, M, dtype=dtype, generator=gen).cuda()


def _softmin(values, gamma):
    values = torch.stack(values)
    if gamma == 0:
        return values.amin(0)
    return -gamma * torch.logsumexp(-values / gamma, dim=0)


def _sequential(cost, gamma, step_pattern):
    """The classical DTW recursion over the (N, M) grid; differentiable."""
    _, N, M = cost.shape
    # A large finite cost for unreachable cells: paths through them get zero
    # weight, while inf would give soft-min a NaN gradient.
    inf = torch.full_like(cost[:, 0, 0], 1e10)
    D = [[None] * M for _ in range(N)]
    for n in range(N):
        for m in range(M):
            if n == 0 and m == 0:
                D[n][m] = cost[:, 0, 0]
                continue
            up = D[n - 1][m] if n > 0 else inf
            diagonal = D[n - 1][m - 1] if n > 0 and m > 0 else inf
            left = D[n][m - 1] if m > 0 else inf
            previous = {
                "symmetric": [up, diagonal, left],
                "asymmetric": [up, diagonal],
                "orthogonal": [up, left],
            }[step_pattern]
            D[n][m] = cost[:, n, m] + _softmin(previous, gamma)
    return D[N - 1][M - 1]


@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
@pytest.mark.parametrize("gamma", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("N, M", [(1, 1), (1, 4), (4, 1), (5, 5), (9, 6), (6, 9)])
def test_dtw_matches_sequential(step_pattern, gamma, N, M):
    cost = _cost(3, N, M)
    expected = _sequential(cost, gamma, step_pattern)
    # Without a path (asymmetric steps with N < M), the distance is inf.
    expected = torch.where(expected > 1e9, float("inf"), expected)
    torch.testing.assert_close(dtw(cost, gamma, step_pattern), expected)


@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_dtw_alignment_is_the_gradient(step_pattern, gamma):
    # For DTW the gradient is the optimal path's indicator; for soft-DTW the
    # expected alignment. Both match the gradient of the sequential recursion.
    cost = _cost(2, 7, 5).requires_grad_()
    (grad,) = torch.autograd.grad(dtw(cost, gamma, step_pattern).sum(), cost)
    (expected,) = torch.autograd.grad(_sequential(cost, gamma, step_pattern).sum(), cost)
    torch.testing.assert_close(grad, expected)
    if gamma == 0:
        assert set(grad.unique().tolist()) <= {0.0, 1.0}
        assert grad[:, 0, 0].eq(1).all() and grad[:, -1, -1].eq(1).all()


@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
def test_soft_dtw_second_derivatives(step_pattern):
    cost = _cost(2, 4, 3).requires_grad_()
    assert torch.autograd.gradgradcheck(lambda c: dtw(c, 0.5, step_pattern), (cost,))


def test_dtw_needs_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        dtw(_cost(1, 3, 3).cpu())


@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
@pytest.mark.parametrize("gamma", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("N, M", [(1, 1), (1, 4), (4, 1), (5, 5), (9, 6), (6, 9), (40, 3)])
def test_dtw_rowwise_matches_sequential(step_pattern, gamma, N, M):
    cost = _cost(3, N, M).requires_grad_()
    expected = _sequential(cost, gamma, step_pattern)
    actual = dtw_rowwise(cost, gamma, step_pattern)
    finite = expected < 1e9
    torch.testing.assert_close(actual[finite], expected[finite])
    assert actual[~finite].isinf().all()
    if finite.all():
        (grad,) = torch.autograd.grad(actual.sum(), cost)
        (expected_grad,) = torch.autograd.grad(expected.sum(), cost)
        torch.testing.assert_close(grad, expected_grad)


def test_soft_dtw_rowwise_second_derivatives():
    cost = _cost(2, 4, 5).requires_grad_()
    assert torch.autograd.gradgradcheck(lambda c: dtw_rowwise(c, 0.5), (cost,))


@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
@pytest.mark.parametrize("gamma", [0.0, 0.5])
@pytest.mark.parametrize("N, M", [(1, 4), (4, 1), (9, 6), (6, 9), (300, 40)])
def test_dtw_fused_matches_rowwise(step_pattern, gamma, N, M):
    cost = _cost(3, N, M, dtype=torch.float32).requires_grad_()
    expected = dtw_rowwise(cost, gamma, step_pattern)
    actual = dtw_fused(cost, gamma, step_pattern)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
    if expected.isfinite().all():
        (grad,) = torch.autograd.grad(actual.sum(), cost)
        (expected_grad,) = torch.autograd.grad(expected.sum(), cost)
        torch.testing.assert_close(grad, expected_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
@pytest.mark.parametrize("N, M", [(4, 5), (5, 3)])
def test_dtw_fused_second_derivatives(step_pattern, N, M):
    if step_pattern == "asymmetric" and N < M:
        pytest.skip("no asymmetric path when N < M")
    cost = _cost(2, N, M).requires_grad_()
    assert torch.autograd.gradgradcheck(lambda c: dtw_fused(c, 0.5, step_pattern), (cost,))
