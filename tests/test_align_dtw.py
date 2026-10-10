import subprocess
import sys

import pytest
import torch

from philtorch.align import dtw, dtw_path, soft_dtw_divergence

# dtw runs Triton kernels, so the tests that run it need CUDA; the validation
# and import tests run anywhere.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

STEP_PATTERNS = ["symmetric", "asymmetric", "orthogonal"]


def _cost(batch_size, N, M, *, seed=0, dtype=torch.float64, device="cuda"):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(batch_size, N, M, dtype=dtype, generator=gen).to(device)


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


@requires_cuda
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


@requires_cuda
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


@requires_cuda
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


@requires_cuda
@pytest.mark.parametrize(
    "step_pattern, diagonal_weight",
    [("symmetric", 1.0), ("symmetric", 2.0), ("asymmetric", 1.0), ("orthogonal", 1.0)],
)
def test_soft_dtw_second_derivatives(step_pattern, diagonal_weight):
    cost = _cost(2, 5, 4).requires_grad_()

    def run(c):
        return dtw(c, 0.5, step_pattern=step_pattern, diagonal_weight=diagonal_weight)

    assert torch.autograd.gradgradcheck(run, (cost,))


@requires_cuda
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


def _band(N, M, band):
    """The band of dtw's docstring, by its definition, for one (N, M) pair."""
    i = torch.arange(N, dtype=torch.float64)[:, None]
    j = torch.arange(M, dtype=torch.float64)
    if min(N, M) == 1:
        return torch.ones(N, M, dtype=torch.bool)
    if N > M:
        slope, offset = (N - 1) / (M - 1), (i - j * (N - 1) / (M - 1)).abs()
    else:
        slope, offset = (M - 1) / (N - 1), (j - i * (M - 1) / (N - 1)).abs()
    return offset <= max(band, slope / 2)


@requires_cuda
@pytest.mark.parametrize("gamma", [0.0, 0.5])
@pytest.mark.parametrize("N, M, band", [(12, 9, 2.0), (9, 12, 2.0), (10, 40, 1.0), (1, 5, 1.0)])
def test_dtw_band(gamma, N, M, band):
    cost = _cost(2, N, M).requires_grad_()
    allowed = _band(N, M, band).expand(2, -1, -1).cuda()
    D = _sequential(cost, gamma, allowed=allowed)[:, -1, -1]
    actual = dtw(cost, gamma, band=band)
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, D)
    (grad,) = torch.autograd.grad(actual.sum(), cost)
    (expected_grad,) = torch.autograd.grad(D.sum(), cost)
    torch.testing.assert_close(grad, expected_grad)
    assert (grad * ~allowed).abs().max() == 0
    if gamma == 0:
        # The path stays in the band, and is the one the gradient marks.
        _, path = dtw_path(cost, band=band)
        for b in range(2):
            cells = path[b][path[b, :, 0] >= 0]
            assert allowed[b][tuple(cells.T)].all()
            assert torch.equal(cells, grad[b].nonzero())


@requires_cuda
def test_dtw_zero_band_on_a_square_grid_is_the_diagonal():
    square = _cost(2, 6, 6)
    diagonal = square.diagonal(dim1=1, dim2=2).sum(-1)
    torch.testing.assert_close(dtw(square, band=0.0), diagonal)


@requires_cuda
@pytest.mark.parametrize("gamma", [0.0, 0.5])
@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
def test_dtw_inf_costs_forbid_cells(gamma, step_pattern):
    cost = _cost(2, 6, 5)
    blocked = torch.zeros_like(cost, dtype=torch.bool)
    blocked[:, 2, 1] = blocked[:, 3, 3] = True
    expected = _sequential(cost, gamma, step_pattern, allowed=~blocked)[:, -1, -1]
    cost = cost.masked_fill(blocked, float("inf")).requires_grad_()
    actual = dtw(cost, gamma, step_pattern=step_pattern)
    torch.testing.assert_close(actual, expected)
    for create_graph in (False, True):
        (grad,) = torch.autograd.grad(
            actual.sum(), cost, retain_graph=True, create_graph=create_graph
        )
        assert grad.isfinite().all() and grad[blocked].abs().max() == 0


@requires_cuda
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_dtw_inf_padding_with_lengths(gamma):
    cost = _cost(2, 4, 4)
    padded = cost.clone()
    padded[:, 2:] = padded[:, :, 2:] = float("inf")
    padded.requires_grad_()
    lengths = (torch.tensor([2, 2]), torch.tensor([2, 2]))
    actual = dtw(padded, gamma, lengths=lengths)
    torch.testing.assert_close(actual, dtw(cost[:, :2, :2], gamma))
    (grad,) = torch.autograd.grad(actual.sum(), padded, create_graph=True)
    assert grad.isfinite().all()


@requires_cuda
def test_dtw_asymmetric_without_a_path():
    cost = _cost(2, 3, 5).requires_grad_()
    distance = dtw(cost, step_pattern="asymmetric")
    assert distance.isinf().all()
    (grad,) = torch.autograd.grad(distance.sum(), cost)
    assert grad.eq(0).all()


def test_dtw_rejects_bad_arguments():
    cost = _cost(1, 3, 3, device="cpu")
    with pytest.raises(ValueError, match="cost must be"):
        dtw(cost[0])
    with pytest.raises(ValueError, match="step_pattern"):
        dtw(cost, step_pattern="symmetric2")
    with pytest.raises(ValueError, match="empty"):
        dtw(cost[:, :0])
    with pytest.raises(ValueError, match="diagonal_weight"):
        dtw(cost, diagonal_weight=2.0, step_pattern="orthogonal")
    with pytest.raises(ValueError, match="lengths must be within 1 to 3"):
        dtw(cost, lengths=(torch.tensor([4]), torch.tensor([3])))
    with pytest.raises(ValueError, match="lengths must be within"):
        dtw(cost, lengths=(torch.tensor([3]), torch.tensor([0])))
    with pytest.raises(ValueError, match="gamma > 0"):
        soft_dtw_divergence(cost, cost, cost, 0.0)
    with pytest.raises(ValueError, match="cost_xx and cost_yy must be"):
        soft_dtw_divergence(cost.expand(2, -1, -1), cost, cost.expand(2, -1, -1), 0.5)


@requires_cuda
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


@requires_cuda
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
    with pytest.raises(ValueError, match="dtw runs Triton kernels on CUDA GPUs only"):
        dtw(_cost(1, 3, 3, device="cpu"))


def test_align_imports_without_triton(tmp_path):
    # The kernels are imported on first use. Run outside the checkout, whose
    # philtorch/ has no built extension, so the subprocess imports the
    # installed package as the tests do.
    code = "import sys; sys.modules['triton'] = None; import philtorch.align"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@requires_cuda
@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize(
    "steps, diagonal_weight",
    [("symmetric", 1.0), ("symmetric", 2.0), ("orthogonal", 1.0), ("ctc", 1.0)],
)
def test_dtw_backward_implementations_agree(soft, steps, diagonal_weight):
    from philtorch.align import _dtw_kernels as kernels

    cost = _cost(3, 7, 9)
    skip = None
    if steps == "ctc":
        skip = (torch.rand(3, 7, generator=torch.Generator().manual_seed(1)) < 0.5).cuda()
        skip = skip.to(torch.int8)
    D = kernels.dtw_dp(cost, skip, soft, steps, diagonal_weight)
    grad = torch.randn_like(D)
    args = (D, cost, skip, grad, soft, steps, diagonal_weight)
    torch.testing.assert_close(kernels._dtw_backward(*args), kernels._dtw_backward_parallel(*args))


_STEPS = {
    "symmetric": {(1, 0), (0, 1), (1, 1)},
    "asymmetric": {(1, 0), (1, 1)},
    "orthogonal": {(1, 0), (0, 1)},
}


@requires_cuda
@pytest.mark.parametrize(
    "step_pattern, diagonal_weight",
    [("symmetric", 1.0), ("symmetric", 2.0), ("asymmetric", 1.0), ("orthogonal", 1.0)],
)
def test_dtw_path(step_pattern, diagonal_weight):
    # The path runs from (0, 0) to each pair's last cell in the pattern's
    # steps, its cost is DTW's distance, and its distance's gradient is it.
    cost = _cost(4, 9, 7).requires_grad_()
    n_len, m_len = torch.tensor([9, 4, 6, 3]), torch.tensor([7, 3, 7, 5])
    options = dict(
        step_pattern=step_pattern, diagonal_weight=diagonal_weight, lengths=(n_len, m_len)
    )
    distance, path = dtw_path(cost, **options)
    expected = dtw(cost, **options)
    torch.testing.assert_close(distance, expected)
    for b in range(4):
        cells = path[b][path[b, :, 0] >= 0].tolist()
        if expected[b].isinf():  # the asymmetric steps with N < M
            assert cells == []
            continue
        assert cells[0] == [0, 0] and cells[-1] == [n_len[b] - 1, m_len[b] - 1]
        steps = [(n1 - n0, m1 - m0) for (n0, m0), (n1, m1) in zip(cells, cells[1:])]
        assert set(steps) <= _STEPS[step_pattern]
        weights = torch.tensor([1.0] + [diagonal_weight if s == (1, 1) else 1.0 for s in steps])
        on_path = weights.to(cost.dtype) @ cost[b][tuple(zip(*cells))].cpu()
        torch.testing.assert_close(on_path, expected[b].detach().cpu())
    possible = expected.isfinite()
    (grad,) = torch.autograd.grad(distance[possible].sum(), cost)
    (expected_grad,) = torch.autograd.grad(expected[possible].sum(), cost)
    torch.testing.assert_close(grad, expected_grad)


@requires_cuda
def test_dtw_rejects_rows_too_long_to_compile():
    with pytest.raises(ValueError, match="2\\^18 per row"):
        dtw(_cost(1, 2, 2**18 + 1, dtype=torch.float32))
