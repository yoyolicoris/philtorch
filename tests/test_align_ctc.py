import pytest
import torch
import torch.nn.functional as F

from philtorch.align import ctc_loss, forced_align

# ctc_loss runs Triton kernels, so the tests that run it need CUDA; the
# validation tests run anywhere.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _problem(N, T, C, S, *, blank=0, seed=0, dtype=torch.float64, device="cuda"):
    """Logits of shape (N, T, C), and padded targets (N, S) of every class but the blank."""
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(N, T, C, dtype=dtype, generator=gen)
    labels = torch.randint(0, C - 1, (N, S), generator=gen)
    return logits.to(device).requires_grad_(), (labels + (labels >= blank)).to(device)


def _pytorch(log_probs, *args, **kwargs):
    """F.ctc_loss on batch-first log-probabilities."""
    return F.ctc_loss(log_probs.transpose(0, 1), *args, **kwargs)


def _losses_and_grads(fn, logits, *args, **kwargs):
    """fn's loss through a log_softmax, and its gradient with respect to the logits."""
    loss = fn(logits.log_softmax(-1), *args, **kwargs)
    (grad,) = torch.autograd.grad(loss.sum(), logits)
    return loss, grad


@requires_cuda
@pytest.mark.parametrize("blank", [0, 5])
@pytest.mark.parametrize(
    "T, frames, labels",
    [
        (30, [30, 25, 17, 12], [8, 5, 0, 6]),
        (9, [6, 3, 4, 1], [4, 3, 2, 2]),  # two exact fits, and a target too long
        (5000, [5000, 3000], [40, 7]),  # rows long enough for the backward over all cells
    ],
)
def test_ctc_matches_pytorch(blank, T, frames, labels):
    logits, targets = _problem(len(frames), T, 6, max(labels), blank=blank)
    targets[0, 1:3] = targets[0, 0]  # repeats, which need a blank between them
    args = (targets, torch.tensor(frames), torch.tensor(labels), blank)
    loss = ctc_loss(logits.log_softmax(-1), *args, reduction="none")
    expected = _pytorch(logits.log_softmax(-1), *args, reduction="none")
    torch.testing.assert_close(loss, expected)
    possible = expected.isfinite()
    (grad,) = torch.autograd.grad(loss[possible].sum(), logits)
    (expected_grad,) = torch.autograd.grad(expected[possible].sum(), logits)
    # F.ctc_loss leaves the impossible targets' gradients nonzero; ours are zero.
    torch.testing.assert_close(grad[possible], expected_grad[possible])
    assert grad[~possible].eq(0).all()


@requires_cuda
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("zero_infinity", [False, True])
def test_ctc_reductions(reduction, zero_infinity):
    # The last target is too long for its frames: an infinite loss.
    logits, targets = _problem(4, 9, 6, 4)
    args = (targets, torch.tensor([6, 3, 4, 1]), torch.tensor([4, 3, 0, 2]))
    kwargs = dict(reduction=reduction, zero_infinity=zero_infinity)
    loss, grad = _losses_and_grads(ctc_loss, logits, *args, **kwargs)
    expected, expected_grad = _losses_and_grads(_pytorch, logits, *args, **kwargs)
    torch.testing.assert_close(loss, expected)
    if zero_infinity:
        torch.testing.assert_close(grad, expected_grad)


@requires_cuda
def test_ctc_target_formats():
    # Concatenated targets with lengths as tuples, and one unbatched sequence.
    logits, targets = _problem(3, 20, 5, 6)
    frames, labels = (20, 18, 15), (6, 0, 4)
    concatenated = torch.cat([targets[n, :s] for n, s in enumerate(labels)])
    padded = ctc_loss(logits.log_softmax(-1), targets, frames, labels, reduction="none")
    torch.testing.assert_close(
        ctc_loss(logits.log_softmax(-1), concatenated, frames, labels, reduction="none"), padded
    )
    unbatched = ctc_loss(logits[2].log_softmax(-1), targets[2, :4], 15, 4, reduction="none")
    torch.testing.assert_close(unbatched, padded[2])


@requires_cuda
def test_ctc_gradient_is_the_occupancy():
    # With respect to the log-probabilities, the gradient is minus each
    # frame's expected occupancy of each class, which sums to 1 per frame.
    logits, targets = _problem(3, 20, 5, 6)
    log_probs = logits.log_softmax(-1).detach().requires_grad_()
    loss = ctc_loss(log_probs, targets, (20,) * 3, (6,) * 3, reduction="sum")
    (grad,) = torch.autograd.grad(loss, log_probs)
    assert (grad <= 0).all()
    torch.testing.assert_close(grad.sum(-1), -torch.ones(3, 20, dtype=grad.dtype, device="cuda"))


@requires_cuda
def test_ctc_second_derivatives():
    logits, targets = _problem(2, 7, 4, 3)
    targets[0, 2] = targets[0, 1]
    log_probs = logits.log_softmax(-1).detach().requires_grad_()

    def loss(x):
        return ctc_loss(x, targets, (7, 7), (3, 2), reduction="none")

    # The gather's backward sums with atomics, so repeated runs differ in the last bits.
    assert torch.autograd.gradgradcheck(loss, (log_probs,), nondet_tol=1e-12)


@requires_cuda
def test_ctc_float32():
    logits, targets = _problem(4, 500, 30, 60, dtype=torch.float32)
    args = (targets, (500,) * 4, (60,) * 4)
    torch.testing.assert_close(
        ctc_loss(logits.log_softmax(-1), *args, reduction="none"),
        _pytorch(logits.log_softmax(-1), *args, reduction="none"),
        rtol=1e-5,
        atol=0,
    )


def test_ctc_rejects_bad_arguments():
    logits, targets = _problem(2, 5, 3, 2, device="cpu")
    lengths = ((5, 5), (2, 2))
    with pytest.raises(ValueError, match="log_probs must be"):
        ctc_loss(logits[0, 0], targets, *lengths)
    with pytest.raises(ValueError, match="unknown reduction"):
        ctc_loss(logits, targets, *lengths, reduction="max")
    with pytest.raises(ValueError, match="must have 2 entries"):
        ctc_loss(logits, targets, (5,), (2, 2))
    with pytest.raises(ValueError, match="targets must be"):
        ctc_loss(logits, targets[:1], *lengths)
    with pytest.raises(ValueError, match="input_lengths must be within 0 to 5"):
        ctc_loss(logits, targets, (5, 6), (2, 2))
    with pytest.raises(ValueError, match="target_lengths must be within 0 to 2"):
        ctc_loss(logits, targets, (5, 5), (-1, 2))
    with pytest.raises(ValueError, match="sum to the 4 concatenated targets"):
        ctc_loss(logits, targets.flatten(), (5, 5), (2, 1))
    with pytest.raises(ValueError, match="ctc_loss runs Triton kernels on CUDA GPUs only"):
        ctc_loss(logits, targets, *lengths)


def _viterbi(log_probs, target, blank):
    """CTC's most probable path for one sequence, by the sequential recursion: its
    states' labels, one per frame, and its log-probability."""
    states = [blank]
    for label in target:
        states += [label, blank]
    T, S = log_probs.size(0), len(states)
    best = torch.full((T, S), float("-inf"), dtype=log_probs.dtype)
    back = torch.zeros((T, S), dtype=torch.long)
    best[0, 0] = log_probs[0, states[0]]
    if S > 1:
        best[0, 1] = log_probs[0, states[1]]
    for t in range(1, T):
        for s in range(S):
            choices = [(best[t - 1, s], s)]
            if s > 0:
                choices.append((best[t - 1, s - 1], s - 1))
            if s > 1 and states[s] != blank and states[s] != states[s - 2]:
                choices.append((best[t - 1, s - 2], s - 2))
            value, back[t, s] = max(choices, key=lambda c: c[0])
            best[t, s] = value + log_probs[t, states[s]]
    end = S - 1 if S == 1 or best[T - 1, S - 1] >= best[T - 1, S - 2] else S - 2
    path = [end]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return [states[s] for s in reversed(path)], best[T - 1, end]


@requires_cuda
@pytest.mark.parametrize("blank", [0, 5])
def test_forced_align_matches_viterbi(blank):
    logits, targets = _problem(4, 12, 6, 4, blank=blank)
    targets[0, 1] = targets[0, 0]  # a repeat, which needs a blank between them
    frames, labels = (12, 9, 6, 2), (4, 3, 0, 3)  # the last cannot fit its frames
    log_probs = logits.log_softmax(-1).detach()
    path, scores = forced_align(log_probs, targets, frames, labels, blank)
    for n in range(4):
        T = frames[n]
        if n == 3:
            assert path[n].eq(-1).all() and scores[n].eq(0).all()
            continue
        expected, best = _viterbi(log_probs[n, :T].cpu(), targets[n, : labels[n]].tolist(), blank)
        assert path[n, :T].tolist() == expected and path[n, T:].eq(-1).all()
        torch.testing.assert_close(scores[n, :T].sum().cpu(), best)
    one_path, one_scores = forced_align(log_probs[1], targets[1, :3], 9, 3, blank)
    assert torch.equal(one_path, path[1]) and torch.equal(one_scores, scores[1])
