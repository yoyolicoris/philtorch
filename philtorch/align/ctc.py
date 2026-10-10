"""Connectionist temporal classification (CTC) as soft-DTW."""

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .._triton import check_cuda_triton


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor | Sequence[int],
    target_lengths: Tensor | Sequence[int],
    blank: int = 0,
    reduction: Literal["none", "mean", "sum"] = "mean",
    zero_infinity: bool = False,
) -> Tensor:
    r"""The connectionist temporal classification loss, with the arguments of
    :func:`torch.nn.functional.ctc_loss` but batch first.

    For per-frame log-probabilities :math:`\log p(k \mid t)` over :math:`C`
    classes and a target of :math:`S` labels, `Connectionist Temporal
    Classification`_ (Graves et al., 2006) sums the probabilities of every
    path of :math:`T` frames that reads as the target once repeats are merged
    and blanks dropped. That sum is soft-DTW with :math:`\gamma = 1` between
    the frames and the target's :math:`2S + 1` states, its labels with a
    blank before, between and after them, at the cost :math:`-\log p(k \mid
    t)` of the state's class at each frame: each frame stays in its state,
    advances to the next or skips a blank between two different labels. So
    it runs on the kernels of :func:`dtw`, one state after another, each a
    parallel scan over the frames: :math:`2S + 1` sequential steps and
    :math:`O(TS)` work and memory.

    The loss is differentiable to any order in reverse mode. Its gradient
    with respect to :attr:`log_probs` is minus each frame's expected
    occupancy of each class; :func:`torch.nn.functional.ctc_loss` returns
    instead that plus the probabilities, which agree once composed with a
    ``log_softmax``.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        log_probs (Tensor): :math:`\log p(k \mid t)`, of shape
            :math:`(N, T, C)`, or :math:`(T, C)` for one sequence, such as a
            ``log_softmax`` over the classes.
        targets (Tensor): the labels, an integer tensor of shape
            :math:`(N, S)`, padded past each target's length, or of the
            targets' total length, the targets concatenated; :math:`(S)` for
            one sequence. Not :attr:`blank`.
        input_lengths (Tensor or tuple of int): each sequence's number of
            frames, of shape :math:`(N)`, or :math:`()` for one sequence.
        target_lengths (Tensor or tuple of int): each target's number of
            labels, of shape :math:`(N)`, or :math:`()` for one sequence.
        blank (int): the blank class. Default: 0.
        reduction (str): ``"none"`` for the losses, ``"mean"`` for the mean
            over the batch of the losses divided by their target lengths, or
            ``"sum"``. Default: ``"mean"``.
        zero_infinity (bool): zero the infinite losses, of targets that no
            path of their frames can read, as their gradients are. Default:
            ``False``.

    Returns:
        Tensor: the losses, of shape :math:`(N)` with ``reduction="none"``,
        else a scalar.

    Raises:
        ValueError: if :attr:`log_probs` is not a CUDA tensor of shape
            :math:`(N, T, C)` or :math:`(T, C)`, if the other arguments'
            shapes do not match it, or if :attr:`reduction` is unknown.
        RuntimeError: if Triton is not installed.

    Example::

        >>> import torch
        >>> from philtorch.align import ctc_loss
        >>> logits = torch.randn(4, 100, 20, device="cuda", requires_grad=True)
        >>> targets = torch.randint(1, 20, (4, 30), device="cuda")
        >>> loss = ctc_loss(logits.log_softmax(-1), targets, (100,) * 4, (30,) * 4)
        >>> loss.backward()

    .. _Connectionist Temporal Classification:
        https://doi.org/10.1145/1143844.1143891
    """
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"unknown reduction {reduction!r}")
    log_probs, states, skip, frames, labels, unbatched = _prepare(
        "ctc_loss", log_probs, targets, input_lengths, target_lengths, blank
    )
    N, T, _ = log_probs.shape
    ends = _ends(log_probs.gather(2, states[:, None].expand(N, T, -1)), skip, frames, labels, True)
    none = ends.isinf().all(0)
    loss = -torch.logsumexp(-torch.where(none, 0.0, ends), dim=0)
    loss = torch.where(none, 0.0 if zero_infinity else float("inf"), loss)
    if reduction == "mean":
        return (loss / labels.clamp(min=1)).mean()
    if reduction == "sum":
        return loss.sum()
    return loss[0] if unbatched else loss


def forced_align(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor | Sequence[int],
    target_lengths: Tensor | Sequence[int],
    blank: int = 0,
) -> tuple[Tensor, Tensor]:
    r"""The most probable alignment of each target to its frames, as
    ``torchaudio.functional.forced_align``, batch first and batched.

    Of the paths that :func:`ctc_loss` sums over, the most probable one: CTC
    Viterbi decoding, the same grid with a hard minimum, the path read from
    its gradient, one-hot at each frame, in one backward pass. Under ties, a
    frame stays in its state rather than advances, and advances rather than
    skips a blank.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        log_probs (Tensor): as in :func:`ctc_loss`, :math:`(N, T, C)` or
            :math:`(T, C)`.
        targets (Tensor): as in :func:`ctc_loss`.
        input_lengths (Tensor or tuple of int): as in :func:`ctc_loss`.
        target_lengths (Tensor or tuple of int): as in :func:`ctc_loss`.
        blank (int): the blank class. Default: 0.

    Returns:
        tuple of Tensor: each frame's label, :attr:`blank` included, an
        integer tensor of shape :math:`(N, T)`, and its log-probability, of
        shape :math:`(N, T)`; :math:`(T)` each for one sequence. Past each
        sequence's frames, and for a target that no path of its frames can
        read, the labels are -1 and the log-probabilities 0.

    Raises:
        ValueError: as :func:`ctc_loss` does.
        RuntimeError: if Triton is not installed.

    Example::

        >>> import torch
        >>> from philtorch.align import forced_align
        >>> log_probs = torch.randn(2, 50, 20, device="cuda").log_softmax(-1)
        >>> targets = torch.randint(1, 20, (2, 10), device="cuda")
        >>> labels, scores = forced_align(log_probs, targets, (50, 40), (10, 8))
    """
    log_probs, states, skip, frames, labels, unbatched = _prepare(
        "forced_align", log_probs, targets, input_lengths, target_lengths, blank
    )
    N, T, _ = log_probs.shape
    with torch.enable_grad():
        state_log_probs = log_probs.detach().gather(2, states[:, None].expand(N, T, -1))
        state_log_probs.requires_grad_()
        ends = _ends(state_log_probs, skip, frames, labels, False)
        # The better end, the last label first under a tie.
        best = ends.gather(0, ends.argmin(0, keepdim=True))[0]
        possible = best.isfinite()
        (grad,) = torch.autograd.grad(torch.where(possible, best, 0.0).sum(), state_log_probs)
    # The gradient is minus the path's one-hot state at each frame.
    state = grad.argmin(-1)
    path = states.gather(1, state)
    scores = log_probs.gather(2, path[..., None])[..., 0]
    valid = possible[:, None] & (torch.arange(T, device=frames.device) < frames[:, None])
    path, scores = torch.where(valid, path, -1), torch.where(valid, scores, 0.0)
    return (path[0], scores[0]) if unbatched else (path, scores)


def _prepare(name, log_probs, targets, input_lengths, target_lengths, blank):
    """The arguments as (N, T, C) log-probabilities, the (N, 2S + 1) states and the
    kernels' skip mask, (N) lengths, and whether there was a batch."""
    if log_probs.dim() not in (2, 3):
        raise ValueError(f"log_probs must be (N, T, C) or (T, C), got {tuple(log_probs.shape)}")
    unbatched = log_probs.dim() == 2
    device = log_probs.device
    frames = torch.as_tensor(input_lengths, device=device, dtype=torch.long).reshape(-1)
    labels = torch.as_tensor(target_lengths, device=device, dtype=torch.long).reshape(-1)
    if unbatched:
        log_probs, targets = log_probs[None], targets.reshape(1, -1)
    N = log_probs.size(0)
    if frames.shape != (N,) or labels.shape != (N,):
        raise ValueError(f"input_lengths and target_lengths must have {N} entries")
    if targets.dim() not in (1, 2) or (targets.dim() == 2 and targets.size(0) != N):
        raise ValueError(f"targets must be of shape ({N}, S) or 1-D, got {tuple(targets.shape)}")
    check_cuda_triton(name, log_probs)

    targets = targets.to(device, torch.long)
    if targets.dim() == 1:
        # Concatenated targets: each one's labels, in order, into a padded row.
        padded = targets.new_zeros(N, int(labels.max()) if N else 0)
        padded[torch.arange(padded.size(1), device=device) < labels[:, None]] = targets
        targets = padded
    # The states: the labels with a blank before, between and after them. A
    # path may skip a blank into a label that differs from the one before.
    states = targets.new_full((N, 2 * targets.size(1) + 1), blank)
    states[:, 1::2] = targets
    skip = torch.zeros_like(states, dtype=torch.bool)
    skip[:, 1] = True
    skip[:, 3::2] = targets[:, 1:] != targets[:, :-1]
    skip = F.pad(skip, (1, 0)).to(torch.int8)
    return log_probs, states, skip, frames, labels, unbatched


def _ends(state_log_probs, skip, frames, labels, soft):
    """-log p of the paths ending in each sequence's last label and in the blank after
    it, (2, N), from each frame's log-probabilities of the states, (N, T, 2S + 1):
    of all paths, or with ``soft`` False the best one."""
    # Imported here so that philtorch.align imports without Triton.
    from ._dtw_kernels import dtw_dp

    # (N, 2S + 1, T), in bits, as the kernels' soft-min is in base 2: one
    # multiply both negates and converts.
    cost = state_log_probs.mT * (-1 / math.log(2))
    # A virtual first state and frame, at no cost where they meet, from which
    # both the first blank and the first label begin.
    cost = F.pad(cost, (1, 0, 1, 0), value=float("inf"))
    cost[:, 0, 0] = 0.0
    D = dtw_dp(cost, skip, soft, "ctc", 1.0)
    batch = torch.arange(D.size(0), device=D.device)
    ends = torch.stack([D[batch, 2 * labels + 1, frames], D[batch, 2 * labels, frames]])
    return ends * math.log(2)
