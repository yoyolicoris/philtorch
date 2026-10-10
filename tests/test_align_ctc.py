import pytest
import torch
import torch.nn.functional as F

from philtorch.align import ctc_loss

# ctc_loss runs Triton kernels, so the tests that run it need CUDA; the
# validation tests run anywhere.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _problem(B, N, C, U, *, blank=0, seed=0, dtype=torch.float64, device="cuda"):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, N, C, dtype=dtype, generator=gen)
    labels = torch.randint(1, C, (B, U), generator=gen)
    targets = torch.where(labels <= blank, labels - 1, labels)  # every class but the blank
    return logits.to(device).requires_grad_(), targets.to(device)


def _reference(logits, targets, input_lengths, target_lengths, blank):
    """F.ctc_loss, per sequence, and its gradient with respect to the logits."""
    log_probs = logits.log_softmax(-1).transpose(0, 1)
    loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction="none")
    (grad,) = torch.autograd.grad(loss.sum(), logits)
    return loss, grad


@requires_cuda
@pytest.mark.parametrize("blank", [0, 5])
@pytest.mark.parametrize(
    "N, frames, labels",
    [
        (30, [30, 25, 17, 12], [8, 5, 0, 6]),
        (9, [6, 3, 4, 1], [4, 3, 2, 2]),  # two exact fits, and a target too long
        (5000, [5000, 3000], [40, 7]),  # rows long enough for the backward over all cells
    ],
)
def test_ctc_matches_pytorch(blank, N, frames, labels):
    B, U = len(frames), max(labels)
    logits, targets = _problem(B, N, 6, U, blank=blank)
    targets[0, 1:3] = targets[0, 0]  # repeats, which need a blank between them
    frames, labels = torch.tensor(frames), torch.tensor(labels)
    expected, expected_grad = _reference(logits, targets, frames, labels, blank)
    loss = ctc_loss(logits.log_softmax(-1), targets, frames, labels, blank)
    torch.testing.assert_close(loss, expected)
    possible = expected.isfinite()
    (grad,) = torch.autograd.grad(loss[possible].sum(), logits)
    torch.testing.assert_close(grad[possible], expected_grad[possible])
    assert grad[~possible].eq(0).all()


@requires_cuda
def test_ctc_gradient_is_the_occupancy():
    # With respect to the log-probabilities, the gradient is minus each
    # frame's expected occupancy of each class, which sums to 1 per frame.
    logits, targets = _problem(3, 20, 5, 6)
    log_probs = logits.log_softmax(-1).detach().requires_grad_()
    (grad,) = torch.autograd.grad(ctc_loss(log_probs, targets).sum(), log_probs)
    assert (grad <= 0).all()
    torch.testing.assert_close(grad.sum(-1), -torch.ones(3, 20, dtype=grad.dtype, device="cuda"))


@requires_cuda
def test_ctc_second_derivatives():
    logits, targets = _problem(2, 7, 4, 3)
    targets[0, 2] = targets[0, 1]
    lengths = torch.tensor([3, 2])
    log_probs = logits.log_softmax(-1).detach().requires_grad_()

    def loss(x):
        return ctc_loss(x, targets, None, lengths)

    # The gather's backward sums with atomics, so repeated runs differ in the last bits.
    assert torch.autograd.gradgradcheck(loss, (log_probs,), nondet_tol=1e-12)


@requires_cuda
def test_ctc_float32():
    logits, targets = _problem(4, 500, 30, 60, dtype=torch.float32)
    expected, _ = _reference(logits, targets, torch.full((4,), 500), torch.full((4,), 60), 0)
    torch.testing.assert_close(
        ctc_loss(logits.log_softmax(-1), targets), expected, rtol=1e-5, atol=0
    )


def test_ctc_rejects_bad_arguments():
    logits, targets = _problem(2, 5, 3, 2, device="cpu")
    with pytest.raises(ValueError, match="log_probs must be"):
        ctc_loss(logits[0], targets)
    with pytest.raises(ValueError, match="targets must be"):
        ctc_loss(logits, targets[0])
    with pytest.raises(ValueError, match="ctc_loss runs Triton kernels on CUDA GPUs only"):
        ctc_loss(logits, targets)
