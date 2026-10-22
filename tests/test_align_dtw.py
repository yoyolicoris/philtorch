import importlib
import itertools
import math
import subprocess
import sys
from functools import partial

import pytest
import torch

from philtorch.align import dtw, dtw_path, soft_dtw_divergence

# dtw runs Triton kernels, so the tests that run it need CUDA; the validation
# and import tests run anywhere.

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


@pytest.mark.cuda
@pytest.mark.parametrize("step_pattern", STEP_PATTERNS)
@pytest.mark.parametrize("gamma", [0.0, 0.1])
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


@pytest.mark.cuda
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


@pytest.mark.cuda
@pytest.mark.parametrize(
    "step_pattern, diagonal_weight",
    [("symmetric", 1.0), ("symmetric", 2.0), ("asymmetric", 1.0), ("orthogonal", 1.0)],
)
def test_soft_dtw_second_derivatives(step_pattern, diagonal_weight):
    cost = _cost(2, 5, 4).requires_grad_()

    def run(c):
        return dtw(c, 0.5, step_pattern=step_pattern, diagonal_weight=diagonal_weight)

    assert torch.autograd.gradgradcheck(run, (cost,))


@pytest.mark.cuda
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


@pytest.mark.cuda
@pytest.mark.parametrize("gamma", [0.0, 0.5])
@pytest.mark.parametrize("N, M, band", [(12, 9, 2.0), (9, 12, 2.0), (10, 40, 1.0), (1, 5, 1.0)])
def test_dtw_band(gamma, N, M, band):
    x, y = _features(2, N, M)
    cost = _squared_distances(x, y)
    allowed = _band(N, M, band).expand(2, -1, -1).cuda()
    D = _sequential(cost, gamma, allowed=allowed)[:, -1, -1]
    actual = dtw(x=x, y=y, gamma=gamma, band=band)
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, D)
    grads = torch.autograd.grad(actual.sum(), (x, y))
    torch.testing.assert_close(grads, torch.autograd.grad(D.sum(), (x, y)))


@pytest.mark.cuda
def test_dtw_zero_band_on_a_square_grid_is_the_diagonal():
    x, y = _features(2, 6, 6)
    diagonal = _squared_distances(x, y).diagonal(dim1=1, dim2=2).sum(-1)
    torch.testing.assert_close(dtw(x=x, y=y, band=0.0), diagonal)


@pytest.mark.cuda
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


@pytest.mark.cuda
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


@pytest.mark.cuda
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
    with pytest.raises(ValueError, match="diagonal_weight must be positive"):
        dtw(cost, diagonal_weight=0.0)
    for gamma in (-0.5, math.nan):
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            dtw(cost, gamma=gamma)
    frames = torch.zeros(1, 3, 1)
    for call, band in itertools.product((dtw, dtw_path), (-1.0, math.nan)):
        with pytest.raises(ValueError, match="band must be non-negative"):
            call(x=frames, y=frames, band=band)
    with pytest.raises(ValueError, match="lengths must be within 1 to 3"):
        dtw(cost, lengths=(torch.tensor([4]), torch.tensor([3])))
    with pytest.raises(ValueError, match="lengths must be within"):
        dtw(cost, lengths=(torch.tensor([3]), torch.tensor([0])))
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="checkpoint must be a positive integer"):
            dtw(x=frames, y=frames, checkpoint=bad)
    with pytest.raises(ValueError, match="with x and y"):
        dtw(cost, checkpoint=3)
    with pytest.raises(ValueError, match="gamma > 0"):
        soft_dtw_divergence(cost, cost, cost, 0.0)
    with pytest.raises(ValueError, match="cost_xx and cost_yy must be"):
        soft_dtw_divergence(cost.expand(2, -1, -1), cost, cost.expand(2, -1, -1), 0.5)


@pytest.mark.cuda
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


@pytest.mark.cuda
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


@pytest.mark.cuda
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_dtw_kernels_own_tiles_agree_with_one_tile(monkeypatch, gamma):
    from philtorch.align import _dtw_kernels

    # Enough tiles side by side for the kernels to tile the grid themselves.
    cost = _cost(4, 128, 1024).requires_grad_()
    assert _dtw_kernels._tile(*cost.shape)[0] < 128

    def run():
        value = dtw(cost, gamma, diagonal_weight=2.0)
        return value, *torch.autograd.grad(value.sum(), cost)

    actual = run()
    monkeypatch.setattr(_dtw_kernels, "_tile", lambda B, R, L, *_: (R, 1024))
    torch.testing.assert_close(actual, run())


@pytest.mark.cuda
def test_dtw_kernels_reject_too_many_cells():
    from philtorch.align import _dtw_kernels

    # An expanded view: the check comes before any allocation.
    cost = torch.zeros(1, 1, 1, device="cuda").expand(1, 46341, 46341)
    with pytest.raises(ValueError, match="at most 2"):
        _dtw_kernels.dtw_dp(cost, None, False, "symmetric", 1.0)


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


@pytest.mark.cuda
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
    torch.testing.assert_close(kernels._dtw_backward(*args), kernels._decomposed_backward(*args))


_STEPS = {
    "symmetric": {(1, 0), (0, 1), (1, 1)},
    "asymmetric": {(1, 0), (1, 1)},
    "orthogonal": {(1, 0), (0, 1)},
}


@pytest.mark.cuda
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


def _derivatives(run, inputs, second):
    """The values of ``run(**inputs)`` and their gradients, and with ``second``
    (for soft-DTW) second derivatives."""
    inputs = {name: t.detach().requires_grad_() for name, t in inputs.items()}
    value = run(**inputs)
    grads = torch.autograd.grad(value.sum(), list(inputs.values()), create_graph=second)
    if not second:
        return value, *grads
    weights = (torch.arange(g.numel(), device=g.device).reshape(g.shape).sin() for g in grads)
    second = torch.autograd.grad(
        sum((g * w).sum() for g, w in zip(grads, weights)), list(inputs.values())
    )
    return value, *grads, *second


@pytest.mark.cuda
@pytest.mark.parametrize("features", [False, True])
@pytest.mark.parametrize("gamma", [0.0, 0.5])
@pytest.mark.parametrize(
    "step_pattern, diagonal_weight",
    [("symmetric", 1.0), ("symmetric", 2.0), ("orthogonal", 1.0)],
)
def test_dtw_tiles_agree_with_one_tile(monkeypatch, features, gamma, step_pattern, diagonal_weight):
    from philtorch.align import _dtw_kernels

    # A grid of many tiles; integer costs for DTW, whose sums are then exact.
    if features:
        frames = (torch.randint(-2, 3, (2, n, 3), device="cuda").double() for n in (70, 37))
        inputs = dict(zip("xy", frames))
    else:
        cost = _cost(2, 70, 37)
        inputs = dict(cost=(cost * 5).floor() if gamma == 0 else cost)
    options = dict(step_pattern=step_pattern, diagonal_weight=diagonal_weight)
    if features and gamma == 0:  # tiles outside the band are skipped
        options["band"] = 3.0
    run = partial(dtw, gamma=gamma, **options)
    expected = _derivatives(run, inputs, gamma > 0)
    monkeypatch.setattr(_dtw_kernels, "_tile", lambda *args: (3, 16))
    # The costs of one anti-diagonal at a time.
    monkeypatch.setattr(_dtw_kernels, "_SCRATCH_CELLS", 1)
    actual = _derivatives(run, inputs, gamma > 0)
    for tiled, whole in zip(actual, expected):
        if gamma == 0:
            torch.testing.assert_close(tiled, whole, rtol=0, atol=0)
        else:
            torch.testing.assert_close(tiled, whole)


def _features(batch_size=3, N=9, M=7, F=2):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(batch_size, N, F, dtype=torch.float64, generator=gen)
    y = torch.randn(batch_size, M, F, dtype=torch.float64, generator=gen)
    return x.cuda().requires_grad_(), y.cuda().requires_grad_()


def _squared_distances(x, y):
    return (x[:, :, None] - y[:, None]).pow(2).sum(-1)


def _banded(x, y, options):
    """The squared distances, inf outside the band of ``options``, and the other
    options: the cost's own way to a band."""
    cost = _squared_distances(x, y)
    options = dict(options)
    band = options.pop("band", None)
    if band is not None:
        B, N, M = cost.shape
        lengths = options.get("lengths") or (torch.full((B,), N), torch.full((B,), M))
        module = importlib.import_module("philtorch.align.dtw")
        orthogonal = options.get("step_pattern") == "orthogonal"
        windows = module._band_windows(N, M, *(t.cuda() for t in lengths), band, orthogonal)
        lo, hi = windows.split(1, -1)
        j = torch.arange(M, device=cost.device)
        cost = torch.where((j >= lo) & (j <= hi), cost, torch.inf)
    return cost, options


@pytest.mark.cuda
def test_soft_dtw_divergence_features():
    x, y = _features()
    lengths = (torch.tensor([9, 4, 6]), torch.tensor([7, 3, 7]))
    actual = soft_dtw_divergence(gamma=0.5, x=x, y=y, lengths=lengths)
    expected = soft_dtw_divergence(
        _squared_distances(x, y),
        _squared_distances(x, x),
        _squared_distances(y, y),
        0.5,
        lengths=lengths,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "call, match",
    [
        (lambda c, x, y: dtw(c, x=x, y=y), "either the cost or x and y"),
        (lambda c, x, y: dtw(c, band=2.0), "band needs x and y"),
        (lambda c, x, y: dtw(), "either the cost or x and y"),
        (lambda c, x, y: dtw_path(x=x), "x and y together"),
        (lambda c, x, y: dtw(x=x, y=y[:, :, :1]), "must be"),
        (lambda c, x, y: dtw(x=x.long(), y=y.long()), "one floating dtype"),
        (lambda c, x, y: soft_dtw_divergence(x=x, y=y), "gamma > 0"),
        (lambda c, x, y: soft_dtw_divergence(c, c, c, 0.5, x=x, y=y), "either the cost"),
    ],
)
def test_dtw_features_reject_bad_arguments(call, match):
    x, y = _features()
    cost = _squared_distances(x, y).detach()
    with pytest.raises(ValueError, match=match):
        call(cost, x, y)


@pytest.mark.cuda
def test_dtw_features_from_differences_under_large_offsets():
    # |x|^2 + |y|^2 - 2 x.y would lose every digit here in float32, in the
    # distance and in its gradient.
    x, y = (1e4 + torch.randn(2, n, 16, device="cuda") for n in (50, 40))
    x64, y64 = (t.double().requires_grad_() for t in (x, y))
    x, y = (t.requires_grad_() for t in (x, y))
    distance = dtw(x=x, y=y)
    expected = dtw(_squared_distances(x64, y64))
    torch.testing.assert_close(distance.double(), expected, rtol=1e-4, atol=1e-3)
    with torch.no_grad():  # the kernels' differences, not the path's cost
        torch.testing.assert_close(dtw(x=x, y=y).double(), expected, rtol=1e-4, atol=1e-3)
    grads = torch.autograd.grad(distance.sum(), (x, y))
    for grad, want in zip(grads, torch.autograd.grad(expected.sum(), (x64, y64))):
        torch.testing.assert_close(grad.double(), want, rtol=1e-3, atol=1e-3)


@pytest.mark.cuda
def test_dtw_features_ignore_frames_past_the_lengths():
    # Even inf there: the gradients' products would turn 0 * inf into NaN.
    x = torch.ones(1, 1, 1, device="cuda", requires_grad=True)
    y = torch.tensor([[[2.0], [torch.inf]]], device="cuda", requires_grad=True)
    lengths = (torch.tensor([1]), torch.tensor([1]))
    distance = dtw(x=x, y=y, lengths=lengths)
    grads = torch.autograd.grad(distance.sum(), (x, y))
    torch.testing.assert_close(distance, torch.ones(1, device="cuda"))
    torch.testing.assert_close(
        grads, (-2 * torch.ones_like(x), torch.tensor([[[2.0], [0.0]]], device="cuda"))
    )


@pytest.mark.cuda
def test_dtw_features_padding_keeps_the_gradients_centring():
    # The backward centres the frames on their mean, padding included: zeros
    # for 100 frames would move it from 1e4 to 2e2, and float32's products
    # then give -0.0078125 for x's gradient, not -0.005859375.
    x = torch.full((1, 1, 1), 1e4, device="cuda")
    y = x + 2.0**-10 * torch.ones(1, 3, 1, device="cuda")

    def grads(x, y, **options):
        x, y = x.clone().requires_grad_(), y.clone().requires_grad_()
        return torch.autograd.grad(dtw(x=x, y=y, gamma=1.0, **options).sum(), (x, y))

    padded = [torch.cat([t, torch.zeros(1, 100 - t.size(1), 1, device="cuda")], 1) for t in (x, y)]
    lengths = (torch.tensor([1]), torch.tensor([3]))
    for actual, want in zip(grads(*padded, lengths=lengths), grads(x, y)):
        torch.testing.assert_close(actual[:, : want.size(1)], want)
        assert actual[:, want.size(1) :].eq(0).all()
    torch.testing.assert_close(grads(x, y)[0], torch.tensor([[[-0.005859375]]], device="cuda"))


@pytest.mark.cuda
def test_dtw_split_chunks_skewed_items(monkeypatch):
    # Padding every item of a round to the widest, one rectangle of 600 columns
    # among 16 of 4, would give frontiers of 17 x 600, not about the sum of
    # the widths.
    from philtorch.align import _dtw_kernels

    module = importlib.import_module("philtorch.align.dtw")
    x, y = torch.randn(1, 40, 2, device="cuda"), torch.randn(1, 700, 2, device="cuda")
    g = module._grid(None, x, y, 0.0, "symmetric", 1.0, None, None)
    narrow = [[0, 8 + 2 * i, 600 + 6 * i, 2, 4] for i in range(16)]
    items = torch.tensor([[0, 0, 0, 8, 600]] + narrow, device="cuda")
    expected = module._cross(g, x, y, items)
    calls, original = [], _dtw_kernels.frontier

    def record(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(_dtw_kernels, "frontier", record)
    actual = module._split(g, x, y, items)
    assert len(calls) == 4  # two chunks, a forward and a reverse frontier each
    for call in calls:
        chunk, h_max, w_max = call[5], call[12], call[13]
        assert len(chunk) * w_max <= 2 * chunk[:, 4].sum()
        assert len(chunk) * h_max <= 2 * chunk[:, 3].sum()
    torch.testing.assert_close(actual.unique(dim=0), expected.unique(dim=0))


@pytest.mark.cuda
def test_dtw_features_without_a_path_in_the_band_give_inf():
    # Pair 0 has an inf frame; pair 1 has a path.
    x, y = (t.detach() for t in _features(2, 6, 6))
    x[0, 2] = x[0, 0] = torch.inf  # the first frame too: no NaN gradients
    distance = dtw(x=x, y=y, band=1.0)  # no gradients
    x, y = x.requires_grad_(), y.requires_grad_()
    actual = dtw(x=x, y=y, band=1.0)
    cost, _ = _banded(x, y, dict(band=1.0))
    expected = dtw(cost)
    assert distance[0].isinf() and distance[1].isfinite()
    torch.testing.assert_close(actual, distance)
    torch.testing.assert_close(actual, expected)
    grads = torch.autograd.grad(actual.sum(), (x, y))
    assert all(g.isfinite().all() and g[0].eq(0).all() for g in grads)
    for grad, want in zip(grads, torch.autograd.grad(expected[1], (x, y))):
        torch.testing.assert_close(grad[1], want[1])  # the cost's has 0 * inf in pair 0
    path = dtw_path(x=x, y=y, band=1.0)[1]
    assert path[0].eq(-1).all() and path[1, 0].ge(0).all()


@pytest.mark.cuda
@pytest.mark.parametrize("grad", [False, True])
def test_dtw_features_with_a_zero_orthogonal_band_keep_a_walk(grad):
    # The band widens to the diagonal and its neighbours, whose tiles are the
    # only ones run: a walk of the 599 cells of cost 1 from (0, 0) to the end.
    x, y = torch.zeros(1, 300, 1, device="cuda"), torch.ones(1, 300, 1, device="cuda")
    x.requires_grad_(grad)
    options = dict(x=x, y=y, step_pattern="orthogonal", band=0.0)
    assert dtw(**options).item() == 599
    distance, path = dtw_path(**options)
    assert distance.item() == 599 and path.ge(0).all()


@pytest.mark.cuda
def test_dtw_features_without_a_band_take_any_finite_distance():
    # 1e38 is finite in float32, though above the band's sentinel threshold.
    x = torch.full((1, 1, 1), 1e19, device="cuda", requires_grad=True)
    y = torch.zeros(1, 1, 1, device="cuda")
    expected = dtw(x=x.detach(), y=y)
    assert expected.isfinite().all()
    torch.testing.assert_close(dtw(x=x, y=y), expected)
    torch.testing.assert_close(dtw((x - y).pow(2)), expected)
    distance, path = dtw_path(x=x, y=y)
    torch.testing.assert_close(distance, expected)
    assert path.eq(0).all()


@pytest.mark.cuda
def test_dtw_features_with_a_tiny_diagonal_weight():
    # In float32, 1 + (w - 1) would round the weight to 0.
    x = torch.tensor([[[0.0], [1.0]]], device="cuda", requires_grad=True)
    y = torch.zeros(1, 2, 1, device="cuda", requires_grad=True)
    cost = _squared_distances(x, y)
    actual = dtw(x=x, y=y, diagonal_weight=1e-8)
    expected = dtw(cost, diagonal_weight=1e-8)
    torch.testing.assert_close(actual, expected)
    assert actual.item() > 0
    grads = torch.autograd.grad(actual.sum(), (x, y))
    torch.testing.assert_close(grads, torch.autograd.grad(expected.sum(), (x, y)))
    torch.testing.assert_close(dtw_path(x=x, y=y, diagonal_weight=1e-8)[0], expected)


@pytest.mark.cuda
def test_dtw_features_keep_the_frames_dtype_under_another_default():
    x, y = (t.detach().float() for t in _features())
    expected = dtw(x=x, y=y, diagonal_weight=2.0)
    expected_path = dtw_path(x=x, y=y, diagonal_weight=2.0)
    x.requires_grad_()
    torch.set_default_dtype(torch.float64)
    try:
        distance = dtw(x=x, y=y, diagonal_weight=2.0)
        path_distance, path = dtw_path(x=x, y=y, diagonal_weight=2.0)
    finally:
        torch.set_default_dtype(torch.float32)
    assert distance.dtype == path_distance.dtype == torch.float32
    torch.testing.assert_close(distance, expected)
    torch.testing.assert_close(path_distance, expected_path[0])
    assert path.equal(expected_path[1])


@pytest.mark.cuda
@pytest.mark.parametrize("step_pattern", ["symmetric", "asymmetric"])
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_dtw_features_of_an_empty_batch(gamma, step_pattern):
    x = torch.empty(0, 3, 1, device="cuda", requires_grad=True)
    options = dict(gamma=gamma, step_pattern=step_pattern)
    assert dtw(x=x.detach(), y=x.detach(), **options).shape == (0,)
    distance = dtw(x=x, y=x, **options)
    assert distance.shape == (0,)
    (grad,) = torch.autograd.grad(distance.sum(), x)
    assert grad.shape == x.shape
    if gamma == 0 and step_pattern == "symmetric":
        distance, path = dtw_path(x=x, y=x)
        assert distance.shape == (0,) and path.shape == (0, 5, 2)


@pytest.mark.cuda
@pytest.mark.parametrize("soft", [False, True])
def test_dtw_backward_features_zeroes_the_skipped_tiles(monkeypatch, soft):
    from philtorch.align import _dtw_kernels

    # A weighted diagonal has an output of its own; the window skips tiles.
    monkeypatch.setattr(_dtw_kernels, "_tile", lambda *args: (3, 16))
    B, R, L, big = 2, 8, 40, 1e3
    x, y = (torch.randn(B, n, 2, device="cuda", dtype=torch.float64) for n in (R, L))
    rows = torch.arange(R, device="cuda")
    window = torch.stack([rows, rows + 3], -1).int().expand(B, R, 2).contiguous()
    options = (window, big, 1.0)
    steps = ("symmetric", 2.0, False, False)
    D = _dtw_kernels.dtw_dp_features(x, y, window, big, 1.0, soft, *steps, R, L)
    grad = torch.ones_like(D)

    def run():
        # The cache hands the NaNs' block back to the next empty tensor.
        junk = torch.full_like(D, torch.nan)
        del junk
        return _dtw_kernels.dtw_dp_backward_features(
            D, x, y, *options, grad, soft, *steps, False, 0
        )

    out = run()
    cols = torch.arange(L, device="cuda")
    inside = ((cols >= rows[:, None]) & (cols <= rows[:, None] + 3)).flip(0, 1)  # reversed
    skipped = torch.zeros_like(inside)
    for i in range(0, R, 3):
        for j in range(0, L, 16):
            skipped[i : i + 3, j : j + 16] = not inside[i : i + 3, j : j + 16].any()
    skipped = skipped.flip(0, 1)
    assert skipped.any()
    assert out.isfinite().all() and out[:, skipped].eq(0).all()
    torch.testing.assert_close(run(), out, rtol=0, atol=0)


@pytest.mark.cuda
def test_dtw_takes_integer_costs():
    cost = torch.tensor([[[0, 1], [1, 0]]], device="cuda")
    torch.testing.assert_close(dtw(cost), torch.zeros(1, device="cuda"))


_FEATURE_CASES = [
    # step_pattern, diagonal_weight, N, M, band, with lengths
    ("symmetric", 1.0, 40, 30, None, False),
    ("symmetric", 2.0, 30, 40, 3.0, True),
    ("orthogonal", 1.0, 40, 30, 3.0, False),
    ("asymmetric", 1.0, 40, 30, 3.0, False),  # sheared, then transposed
    ("asymmetric", 1.0, 30, 8, 2.0, True),  # sheared, not transposed
    ("orthogonal", 1.0, 26, 40, 0.5, False),  # a band widened to keep a path
]


def _feature_options(step_pattern, diagonal_weight, N, M, band, with_lengths):
    options = dict(step_pattern=step_pattern, diagonal_weight=diagonal_weight, band=band)
    if with_lengths:
        options["lengths"] = (torch.tensor([N, N - 5, N - 2]), torch.tensor([M, M - 3, M]))
    return options


@pytest.mark.cuda
@pytest.mark.parametrize("gamma", [0.0, 0.5])
@pytest.mark.parametrize("case", _FEATURE_CASES)
def test_dtw_features_derivatives_equal_the_cost(gamma, case):
    x, y = _features(3, *case[2:4])
    options = _feature_options(*case)

    def banded(x, y):
        cost, rest = _banded(x, y, options)
        return dtw(cost, gamma, **rest)

    inputs = dict(x=x, y=y)
    actual = _derivatives(partial(dtw, gamma=gamma, **options), inputs, gamma > 0)
    assert actual[0].isfinite().all()
    torch.testing.assert_close(actual, _derivatives(banded, inputs, gamma > 0))


@pytest.mark.cuda
@pytest.mark.parametrize("gamma", [0.0, 0.5])
@pytest.mark.parametrize("case", _FEATURE_CASES)
def test_dtw_features_without_gradients_equal_the_cost(monkeypatch, gamma, case):
    from philtorch.align import _dtw_kernels

    monkeypatch.setattr(_dtw_kernels, "_frontier_tile", lambda S, R, L: (3, 4))  # many tiles
    x, y = (t.detach() for t in _features(3, *case[2:4]))
    options = _feature_options(*case)
    actual = dtw(x=x, y=y, gamma=gamma, **options)
    cost, rest = _banded(x, y, options)
    expected = dtw(cost, gamma, **rest)
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
def test_dtw_features_without_gradients_stay_linear_in_memory():
    N = M = 4096
    x, y = torch.randn(2, 1, N, 3, device="cuda")
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    distance = dtw(x=x, y=y, gamma=0.5)
    assert torch.cuda.max_memory_allocated() - base < N * M * 4 / 8
    # Float32 sums of 8191 costs each, in another order than the cost's.
    expected = dtw(_squared_distances(x, y), 0.5)
    torch.testing.assert_close(distance, expected, rtol=1e-5, atol=0)


@pytest.mark.cuda
def test_dtw_features_gradient_stays_linear_in_memory():
    # DTW's gradient is its path's: no accumulated costs are stored, only the
    # divide and conquer's leaves, of bounded size.
    N = M = 8192
    x, y = (t.requires_grad_() for t in torch.randn(2, 1, N, 3, device="cuda"))
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.autograd.grad(dtw(x=x, y=y).sum(), (x, y))
    assert torch.cuda.max_memory_allocated() - base < N * M * 4 / 4


@pytest.mark.cuda
@pytest.mark.parametrize("case", _FEATURE_CASES)
def test_dtw_path_features_by_divide_and_conquer(monkeypatch, case):
    from philtorch.align import _dtw_kernels

    # Leaves of at most 2 rows, in chunks of 64 cells, and frontiers of many tiles.
    module = importlib.import_module("philtorch.align.dtw")
    for name, value in dict(_LEAF_ROWS=2, _LEAF_CELLS=0, _CHUNK_CELLS=64).items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(_dtw_kernels, "_frontier_tile", lambda S, R, L: (3, 4))
    x, y = (t.detach() for t in _features(3, *case[2:4]))
    options = _feature_options(*case)
    distance, path = dtw_path(x=x, y=y, **options)
    cost, rest = _banded(x, y, options)
    expected, expected_path = dtw_path(cost, **rest)
    torch.testing.assert_close(distance, expected)
    torch.testing.assert_close(path, expected_path)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "step_pattern, diagonal_weight, band",
    [("symmetric", 2.0, None), ("asymmetric", 1.0, None), ("orthogonal", 1.0, 4.0)],
)
def test_dtw_checkpoint_equals_storing_every_row(step_pattern, diagonal_weight, band):
    # Soft-DTW only: DTW's gradient is its path's, found without the blocks.
    # A grid of 20 rows, in blocks of 3; the pairs' ends fall in different blocks.
    x, y = (t.detach() for t in _features(3, 26, 20))
    inputs = dict(x=x, y=y)
    options = dict(
        step_pattern=step_pattern,
        diagonal_weight=diagonal_weight,
        band=band,
        lengths=(torch.tensor([26, 15, 24]), torch.tensor([20, 14, 9])),
    )
    run = partial(dtw, gamma=0.5, **options)
    expected = _derivatives(run, inputs, True)
    actual = _derivatives(partial(run, checkpoint=3), inputs, True)
    for checkpointed, stored in zip(actual, expected):
        torch.testing.assert_close(checkpointed, stored)


# (N, M, step_pattern, diagonal_weight, band, tile, dtype, checkpoint): grids of
# 3 tile columns, of 16 and tile rows of 3 (the kernels' own for the last two),
# one transposed and one sheared; the last in float32, to first order.
_STRIP_CASES = [
    (26, 40, "symmetric", 1.0, None, (3, 16), torch.float64, 3),
    (40, 26, "symmetric", 2.0, 5.0, (3, 16), torch.float64, 3),
    (26, 40, "orthogonal", 1.0, 4.0, (3, 16), torch.float64, 3),
    (60, 20, "asymmetric", 1.0, None, (3, 16), torch.float64, 3),
    (40, 30, "asymmetric", 1.0, 3.0, (3, 16), torch.float64, 3),
    (100, 600, "symmetric", 2.0, None, None, torch.float64, 40),
    (100, 600, "symmetric", 1.0, 40.0, None, torch.float32, 40),
]


@pytest.mark.cuda
@pytest.mark.parametrize("N, M, step_pattern, diagonal_weight, band, tile, dtype, k", _STRIP_CASES)
def test_dtw_strips_equal_storing_every_row(
    monkeypatch, N, M, step_pattern, diagonal_weight, band, tile, dtype, k
):
    from philtorch.align import _dtw_kernels

    # Strips of 2 tile anti-diagonals, each pair's end in another; with second
    # derivatives, through the strips' own backward.
    if tile is not None:
        monkeypatch.setattr(_dtw_kernels, "_tile", lambda *args: tile)
    B = 3 if tile else 1
    x, y = (t.detach().to(dtype) for t in _features(B, N, M, 2 if tile else 3))
    lengths = ([N, N * 3 // 5, N * 4 // 5][:B], [M, M * 2 // 3, M // 2][:B])
    options = dict(
        step_pattern=step_pattern,
        diagonal_weight=diagonal_weight,
        band=band,
        lengths=tuple(map(torch.tensor, lengths)),
    )
    run = partial(dtw, gamma=0.5, **options)
    second = dtype == torch.float64  # float32: first order only
    expected = _derivatives(run, dict(x=x, y=y), second)
    actual = _derivatives(partial(run, checkpoint=k), dict(x=x, y=y), second)
    tolerance = {} if dtype == torch.float64 else dict(rtol=1e-4, atol=1e-4)
    for checkpointed, stored in zip(actual, expected):
        torch.testing.assert_close(checkpointed, stored, **tolerance)


@pytest.mark.cuda
def test_dtw_strips_third_derivatives(monkeypatch):
    from philtorch.align import _dtw_kernels

    # 4 tile rows of 2 by 3 tile columns of 8: 3 strips.
    monkeypatch.setattr(_dtw_kernels, "_tile", lambda *args: (2, 8))
    x, y = _features(1, 8, 20, 1)
    run = partial(dtw, gamma=0.5, step_pattern="symmetric", diagonal_weight=2.0, band=6.0)

    def third(checkpoint):
        loss = run(x=x, y=y, checkpoint=checkpoint).sum()
        grads = torch.autograd.grad(loss, (x, y), create_graph=True)
        second = torch.autograd.grad(sum(g.sin().sum() for g in grads), (x, y), create_graph=True)
        return torch.autograd.grad(sum(g.cos().sum() for g in second), (x, y))

    torch.testing.assert_close(third(2), third(None))


@pytest.mark.cuda
def test_dtw_strips_of_an_empty_batch(monkeypatch):
    from philtorch.align import _dtw_kernels

    monkeypatch.setattr(_dtw_kernels, "_tile", lambda *args: (3, 16))
    x, y = (torch.empty(0, n, 1, device="cuda", requires_grad=True) for n in (12, 36))
    distance = dtw(x=x, y=y, gamma=0.5, checkpoint=3)
    assert distance.shape == (0,)
    grads = torch.autograd.grad(distance.sum(), (x, y))
    assert [g.shape for g in grads] == [x.shape, y.shape]


@pytest.mark.cuda
def test_dtw_strips_reject_too_many_cells(monkeypatch):
    from philtorch.align import _dtw_kernels

    # A strip's buffer is checked before it is allocated.
    monkeypatch.setattr(_dtw_kernels, "_tile", lambda *args: (3, 16))
    monkeypatch.setattr(_dtw_kernels, "_MAX_CELLS", 100)
    x, y = (torch.zeros(1, n, 1, device="cuda", requires_grad=True) for n in (12, 36))
    with pytest.raises(ValueError, match="at most 2"):
        dtw(x=x, y=y, gamma=0.5, checkpoint=3)


@pytest.mark.cuda
def test_dtw_checkpoint_lowers_the_backward_peak_memory():
    x, y = torch.randn(2, 1, 2048, 2, device="cuda").unbind(0)
    x.requires_grad_(), y.requires_grad_()
    base = torch.cuda.memory_allocated()
    peaks = {}
    for checkpoint in (None, 64):
        torch.cuda.reset_peak_memory_stats()
        distance = dtw(x=x, y=y, gamma=0.5, checkpoint=checkpoint)
        torch.autograd.grad(distance.sum(), (x, y))
        peaks[checkpoint] = torch.cuda.max_memory_allocated() - base
    assert peaks[64] < peaks[None] / 2


@pytest.mark.cuda
@pytest.mark.parametrize("soft", [False, True])
def test_dtw_features_seeded_row_survives_the_window(monkeypatch, soft):
    from philtorch.align import _dtw_kernels

    # The seed's row spans three tiles of columns, the window only the first.
    monkeypatch.setattr(_dtw_kernels, "_tile", lambda *args: (3, 16))
    B, R, L, big = 2, 8, 40, 1e3
    x, y, seed = (torch.randn(B, n, 2, device="cuda", dtype=torch.float64) for n in (R, L, L))
    seed = seed[..., 0].requires_grad_()
    x, y = x.requires_grad_(), y.requires_grad_()
    rows = torch.arange(R, device="cuda")
    window = torch.stack([rows, rows + 3], -1).int().expand(B, R, 2).contiguous()
    weights = torch.randn(B, L, device="cuda", dtype=torch.float64)

    def loss(D):
        # Finite on the reference, whose skipped tiles are not inf.
        return (D[:, 0] * weights).sum() + D[:, -1, :16].sum()

    def minimum(a, b):
        both = torch.stack([a, b])
        return -torch.logsumexp(-both * math.log(2), 0) / math.log(2) if soft else both.min(0)[0]

    # The recursion in PyTorch, over the orthogonal steps.
    cost = (x[:, :, None] - y[:, None]).pow(2).sum(-1)
    cols = torch.arange(L, device="cuda")
    cost = torch.where((cols < rows[:, None]) | (cols > rows[:, None] + 3), big, cost)
    expected = [seed]
    for i in range(1, R):
        row = []
        for j in range(L):
            up = expected[i - 1][:, j]
            row.append((minimum(up, row[-1]) if j else up) + cost[:, i, j])
        expected.append(torch.stack(row, 1))
    expected = torch.stack(expected, 1)
    D = _dtw_kernels.dtw_dp_features(
        x, y, window, big, 1.0, soft, "orthogonal", 1.0, False, False, R, L, seed, 0
    )
    inputs = (seed, x, y)
    torch.testing.assert_close(D[:, 0], seed)
    torch.testing.assert_close(
        torch.autograd.grad(loss(D), inputs), torch.autograd.grad(loss(expected), inputs)
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "case", [("asymmetric", 1.0, 40, 30, 3.0, False), ("asymmetric", 1.0, 24, 20, 2.0, True)]
)
def test_dtw_asymmetric_gradients_in_chunks(monkeypatch, case):
    from philtorch.align import _dtw_kernels

    # Chunks of the shear's width, the last one partly empty, and past x's end.
    monkeypatch.setattr(_dtw_kernels, "_MIN_ROWS", 2)
    x, y = _features(3, *case[2:4])
    options = _feature_options(*case)
    inputs = dict(x=x, y=y)

    def banded(x, y):
        cost, rest = _banded(x, y, options)
        return dtw(cost, 0.5, **rest)

    actual = _derivatives(partial(dtw, gamma=0.5, checkpoint=5, **options), inputs, True)
    torch.testing.assert_close(actual, _derivatives(banded, inputs, True))


@pytest.mark.cuda
def test_dtw_asymmetric_gradient_stays_linear_in_the_block():
    # A transposed grid: rows of the 3072 columns of y, each block of 17 of the
    # shear's 1025 diagonals. Their cells in a dense 3088 by 3072 block of the
    # rows of x by the columns of y would be 36 MiB.
    N, M, checkpoint = 4096, 3072, 16
    x, y = (torch.randn(1, n, 2, device="cuda").requires_grad_() for n in (N, M))

    def run():
        distance = dtw(x=x, y=y, gamma=0.5, step_pattern="asymmetric", checkpoint=checkpoint)
        return torch.autograd.grad(distance.sum(), (x, y))

    run()  # warm up: the kernels and the matrix products' workspaces
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    run()
    assert torch.cuda.max_memory_allocated() - base < (M + checkpoint) * M * 4 / 4
