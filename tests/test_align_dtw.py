import pytest
import torch

from philtorch.align import dtw, soft_dtw_divergence

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


def _sequential(cost, gamma, step_pattern="symmetric", diagonal_weight=1.0, allowed=None):
    """The classical DTW recursion over each (N, M) grid; differentiable.

    ``allowed`` masks the cells a path may use. Returns the whole D, (B, N, M).
    """
    _, N, M = cost.shape
    # A large finite cost for unreachable cells: paths through them get zero
    # weight, while inf would give soft-min a NaN gradient.
    big = torch.full_like(cost[:, 0, 0], 1e10)
    D = [[None] * M for _ in range(N)]
    for n in range(N):
        for m in range(M):
            c = cost[:, n, m]
            if n == 0 and m == 0:
                D[n][m] = c
            else:
                up = D[n - 1][m] + c if n > 0 else big
                diagonal = D[n - 1][m - 1] + diagonal_weight * c if n > 0 and m > 0 else big
                left = D[n][m - 1] + c if m > 0 else big
                previous = {
                    "symmetric": [up, diagonal, left],
                    "asymmetric": [up, diagonal],
                    "orthogonal": [up, left],
                }[step_pattern]
                D[n][m] = _softmin(previous, gamma)
            if allowed is not None:
                D[n][m] = torch.where(allowed[:, n, m], D[n][m], big)
    return torch.stack([torch.stack(row, -1) for row in D], -2)


def _distance(cost, gamma, step_pattern="symmetric", diagonal_weight=1.0):
    D = _sequential(cost, gamma, step_pattern, diagonal_weight)[:, -1, -1]
    return torch.where(D > 1e9, float("inf"), D)


@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
@pytest.mark.parametrize("gamma", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("N, M", [(1, 1), (1, 4), (4, 1), (5, 5), (9, 6), (6, 9), (40, 3)])
def test_dtw_matches_sequential(step_pattern, gamma, N, M):
    cost = _cost(3, N, M).requires_grad_()
    expected = _distance(cost, gamma, step_pattern)
    actual = dtw(cost, gamma, step_pattern=step_pattern)
    torch.testing.assert_close(actual, expected)
    if expected.isfinite().all():
        (grad,) = torch.autograd.grad(actual.sum(), cost)
        (expected_grad,) = torch.autograd.grad(expected.sum(), cost)
        torch.testing.assert_close(grad, expected_grad)


@pytest.mark.parametrize("gamma", [0.0, 0.5])
@pytest.mark.parametrize("diagonal_weight", [2.0, 0.5])
def test_dtw_diagonal_weight(gamma, diagonal_weight):
    cost = _cost(3, 9, 7).requires_grad_()
    expected = _distance(cost, gamma, diagonal_weight=diagonal_weight)
    actual = dtw(cost, gamma, diagonal_weight=diagonal_weight)
    torch.testing.assert_close(actual, expected)
    (grad,) = torch.autograd.grad(actual.sum(), cost)
    (expected_grad,) = torch.autograd.grad(expected.sum(), cost)
    torch.testing.assert_close(grad, expected_grad)


@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_dtw_alignment_is_the_gradient(step_pattern, gamma):
    # For DTW the gradient is the optimal path's indicator.
    cost = _cost(2, 7, 5).requires_grad_()
    (grad,) = torch.autograd.grad(dtw(cost, gamma, step_pattern=step_pattern).sum(), cost)
    if gamma == 0:
        assert set(grad.unique().tolist()) <= {0.0, 1.0}
        assert grad[:, 0, 0].eq(1).all() and grad[:, -1, -1].eq(1).all()
    else:
        # The expected alignment: every path passes through both corners.
        torch.testing.assert_close(grad[:, 0, 0], torch.ones_like(grad[:, 0, 0]))
        assert (grad >= 0).all() and (grad <= 1 + 1e-12).all()


@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
@pytest.mark.parametrize("diagonal_weight", [1.0, 2.0])
def test_soft_dtw_second_derivatives(step_pattern, diagonal_weight):
    if diagonal_weight != 1.0 and step_pattern != "symmetric":
        pytest.skip("diagonal weights need the symmetric steps")
    cost = _cost(2, 5, 4).requires_grad_()

    def run(c):
        return dtw(c, 0.5, step_pattern=step_pattern, diagonal_weight=diagonal_weight)

    assert torch.autograd.gradgradcheck(run, (cost,))


@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_dtw_lengths(step_pattern, gamma):
    cost = _cost(4, 9, 7).requires_grad_()
    n_len = torch.tensor([9, 4, 6, 1])
    m_len = torch.tensor([7, 3, 7, 1])
    actual = dtw(cost, gamma, step_pattern=step_pattern, lengths=(n_len, m_len))
    expected = torch.stack(
        [
            _distance(cost[b : b + 1, :n, :m], gamma, step_pattern)[0]
            for b, (n, m) in enumerate(zip(n_len.tolist(), m_len.tolist()))
        ]
    )
    torch.testing.assert_close(actual, expected)
    finite = expected.isfinite()
    (grad,) = torch.autograd.grad(actual[finite].sum(), cost)
    (expected_grad,) = torch.autograd.grad(expected[finite].sum(), cost)
    torch.testing.assert_close(grad, expected_grad)


@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_dtw_band(gamma):
    cost = _cost(2, 12, 9).requires_grad_()
    band = 2.0
    i = torch.arange(12, dtype=torch.float64)[:, None]
    j = torch.arange(9, dtype=torch.float64)
    allowed = ((j - i * 8 / 11).abs() <= band).expand(2, -1, -1).cuda()
    D = _sequential(cost, gamma, allowed=allowed)[:, -1, -1]
    actual = dtw(cost, gamma, band=band)
    torch.testing.assert_close(actual, D)
    (grad,) = torch.autograd.grad(actual.sum(), cost)
    (expected_grad,) = torch.autograd.grad(D.sum(), cost)
    torch.testing.assert_close(grad, expected_grad)
    assert grad[~allowed].abs().max() == 0
    # On a square grid, a zero band leaves the diagonal alone.
    square = _cost(2, 6, 6)
    diagonal = square.diagonal(dim1=1, dim2=2).sum(-1)
    torch.testing.assert_close(dtw(square, band=0.0), diagonal)
    # Too narrow a band for the grid leaves no path.
    assert dtw(cost, gamma, band=0.0).isinf().all()


def test_dtw_asymmetric_without_a_path():
    cost = _cost(2, 3, 5)
    assert dtw(cost, step_pattern="asymmetric").isinf().all()


def test_soft_dtw_divergence():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(3, 8, 2, dtype=torch.float64, generator=gen).cuda()
    y = torch.randn(3, 6, 2, dtype=torch.float64, generator=gen).cuda()

    def cost(a, b):
        return torch.cdist(a, b) ** 2

    divergence = soft_dtw_divergence(cost(x, y), cost(x, x), cost(y, y), 0.5)
    expected = dtw(cost(x, y), 0.5) - 0.5 * (dtw(cost(x, x), 0.5) + dtw(cost(y, y), 0.5))
    torch.testing.assert_close(divergence, expected)
    assert (divergence >= 0).all()
    zero = soft_dtw_divergence(cost(x, x), cost(x, x), cost(x, x), 0.5)
    torch.testing.assert_close(zero, torch.zeros_like(zero))


@pytest.mark.parametrize("gamma", [0.0, 0.5])
@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
def test_dtw_float32_long(gamma, step_pattern):
    cost = _cost(2, 300, 40)
    expected = _sequential(cost[:, :60, :12], gamma, step_pattern)[:, -1, -1]
    actual = dtw(
        cost.float(),
        gamma,
        step_pattern=step_pattern,
        lengths=(torch.tensor([60, 60]), torch.tensor([12, 12])),
    )
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-4)


def test_dtw_needs_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        dtw(_cost(1, 3, 3).cpu())


def test_dtw_rejects_diagonal_weight_without_diagonal():
    with pytest.raises(ValueError, match="diagonal_weight"):
        dtw(_cost(1, 3, 3), diagonal_weight=2.0, step_pattern="orthogonal")


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("diag, diagonal_weight", [(True, 1.0), (True, 2.0), (False, 1.0)])
def test_dtw_backward_implementations_agree(soft, diag, diagonal_weight):
    from philtorch.align import _dtw_kernels as kernels

    cost = _cost(3, 7, 9)
    D = kernels.dtw_dp(cost, soft, diag, diagonal_weight)
    grad = torch.randn_like(D)
    fused = kernels._dtw_backward(D, cost, grad, soft, diag, diagonal_weight)
    parallel = kernels._dtw_backward_parallel(D, cost, grad, soft, diag, diagonal_weight)
    torch.testing.assert_close(fused, parallel)
