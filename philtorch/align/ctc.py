"""Connectionist temporal classification (CTC) as soft-DTW."""

import math
import numbers
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
    r"""Compute the connectionist temporal classification (CTC) loss.

    It takes the arguments of :func:`torch.nn.functional.ctc_loss`, but batch
    first. For per-frame log-probabilities :math:`\log p(k \mid t)` over
    :math:`C` classes and a target of :math:`S` labels, `Connectionist
    Temporal Classification`_ (Graves et al., 2006) sums the probabilities of
    every path of :math:`T` frames that reads as the target once repeats are
    merged and blanks dropped. That sum is soft-DTW with :math:`\gamma = 1`
    between the frames and the target's :math:`2S + 1` states, its labels with
    a blank before, between and after them, at the cost :math:`-\log p(k \mid
    t)` of the state's class at each frame: each frame stays in its state,
    advances to the next or skips a blank between two different labels. So
    it runs on the kernels of :func:`dtw`, one state after another, each a
    parallel scan over the frames: :math:`2S + 1` sequential steps and
    :math:`O(BTS)` work and memory.

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
            :math:`(B, T, C)`, or :math:`(T, C)` for one sequence, such as a
            ``log_softmax`` over the classes.
        targets (Tensor): the labels, an integer tensor of shape
            :math:`(B, S)`, padded past each target's length, or of shape
            :math:`(\sum_b S_b)`, the targets concatenated; :math:`(S)` for
            one sequence. Not :attr:`blank`.
        input_lengths (Tensor or tuple of int): each sequence's number of
            frames, of shape :math:`(B)`, or :math:`()` for one sequence.
        target_lengths (Tensor or tuple of int): each target's number of
            labels, of shape :math:`(B)`, or :math:`()` for one sequence.
            With concatenated targets, a tensor's values size the padded
            targets, which breaks a :func:`torch.compile` graph.
        blank (int, optional): the blank class. Default: 0.
        reduction (str, optional): ``"none"`` for the losses, ``"mean"`` for
            the mean over the batch of the losses divided by their target
            lengths, or ``"sum"``. Default: ``"mean"``.
        zero_infinity (bool, optional): if ``True``, zeroes the infinite
            losses, of targets that no path of their frames can read; their
            gradients are zero either way. Default: ``False``.

    Returns:
        Tensor: the losses, of shape :math:`(B)` with ``reduction="none"``
        (:math:`()` for one sequence), else a scalar.

    Raises:
        ValueError: if :attr:`log_probs` is not a CUDA tensor of shape
            :math:`(B, T, C)` or :math:`(T, C)`, if the other arguments'
            shapes do not match it, if a length is negative or past its
            sequence (or the concatenated targets' lengths do not sum to
            theirs), if a sequence's grid, :math:`(2S + 2)(T + 1)` cells with
            :math:`S` the padded targets' width, exceeds :math:`2^{31} - 1`
            cells, or if :attr:`reduction` is unknown. Lengths given as
            tensors are checked only in eager mode: their values would break
            a :func:`torch.compile` graph.
        RuntimeError: if Triton is not installed.

    Example::

        >>> import torch
        >>> from philtorch.align import ctc_loss
        >>> logits = torch.randn(4, 100, 20, device="cuda", requires_grad=True)
        >>> targets = torch.randint(1, 20, (4, 30), device="cuda")
        >>> log_probs = logits.log_softmax(-1)
        >>> loss = ctc_loss(log_probs, targets, (100,) * 4, (30,) * 4)
        >>> loss.backward()

    .. _Connectionist Temporal Classification:
        https://doi.org/10.1145/1143844.1143891
    """
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"unknown reduction {reduction!r}")
    log_probs, states, skip, frames, labels, unbatched = _prepare(
        "ctc_loss", log_probs, targets, input_lengths, target_lengths, blank
    )
    ends = _grid(log_probs, states, skip, frames, labels, False)[1]
    none = ends.isinf().all(0)
    loss = -torch.logsumexp(-torch.where(none, 0.0, ends), dim=0)
    loss = torch.where(none, 0.0 if zero_infinity else float("inf"), loss)
    if reduction == "mean":
        loss = (loss / labels.clamp(min=1)).mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif unbatched:
        loss = loss[0]
    return loss.to(log_probs.dtype)


def forced_align(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor | Sequence[int],
    target_lengths: Tensor | Sequence[int],
    blank: int = 0,
) -> tuple[Tensor, Tensor]:
    r"""Find each target's most probable CTC alignment to its frames.

    As ``torchaudio.functional.forced_align``, but for any batch size, not
    only 1. Of the paths that :func:`ctc_loss` sums over, the most probable
    one: CTC Viterbi decoding, the same grid with a hard minimum, each cell's
    best step back found afterwards from its accumulated costs, over all cells
    at once, and the traceback of :func:`~philtorch.estimation.hmm_viterbi`,
    parallel over the frames. Under ties, a frame stays in its state rather
    than advances, and advances rather than skips a blank.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        log_probs (Tensor): as in :func:`ctc_loss`.
        targets (Tensor): as in :func:`ctc_loss`.
        input_lengths (Tensor or tuple of int): as in :func:`ctc_loss`.
        target_lengths (Tensor or tuple of int): as in :func:`ctc_loss`.
        blank (int, optional): as in :func:`ctc_loss`.

    Returns:
        tuple of Tensor: each frame's label, :attr:`blank` included, an
        integer tensor of shape :math:`(B, T)`, and its log-probability, of
        shape :math:`(B, T)`; :math:`(T)` each for one sequence. Past each
        sequence's frames, and for a target that no path of its frames can
        read, the labels are -1 and the log-probabilities 0. Neither carries a
        gradient.

    Raises:
        ValueError: as :func:`ctc_loss` does for these arguments.
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
    T = log_probs.size(1)
    from .._trace_kernels import trace

    with torch.no_grad():
        steps_back, ends = _grid(log_probs, states, skip, frames, labels, True)
        # From the better end, the final blank first under a tie. Each frame's
        # steps back are offsets to the row before; past a sequence's frames
        # they stay, so its traceback starts at its own end.
        rows = (2 * labels + 1 - ends.argmin(0)).int()
        offsets = steps_back[:, :, 1:].mT
        past = torch.arange(T, device=frames.device) >= frames[:, None]
        offsets = torch.where(past[..., None], 0, offsets).contiguous()
        # The row at each frame, from the virtual cell's: frame t's at t + 1.
        before = trace(rows, offsets, reverse=True, relative=True)
        state = torch.cat([before[:, 1:], rows[:, None]], 1).long() - 1
        possible = ends.min(0).values.isfinite()
        valid = possible[:, None] & ~past
        path = states.gather(1, state.clamp(min=0))
        scores = log_probs.gather(2, path[..., None])[..., 0]
        path, scores = torch.where(valid, path, -1), torch.where(valid, scores, 0.0)
    return (path[0], scores[0]) if unbatched else (path, scores)


def _lengths(lengths: Tensor | Sequence[int] | int) -> tuple[Tensor, list[int] | None]:
    """Lengths as a 1-D long tensor, and as a list of ints if they were given
    as ints."""
    if isinstance(lengths, Tensor):
        return lengths.long().reshape(-1), None
    if isinstance(lengths, numbers.Integral):
        lengths = (lengths,)
    values = [int(n) for n in lengths]  # a sequence, or a NumPy array
    return torch.tensor(values, dtype=torch.long), values


def _prepare(name, log_probs, targets, input_lengths, target_lengths, blank):
    """The arguments as (B, T, C) log-probabilities, the (B, 2S + 1) states and
    the kernels' skip mask, (B) lengths, and whether there was a batch."""
    if log_probs.dim() not in (2, 3):
        raise ValueError(f"log_probs must be (B, T, C) or (T, C), got {tuple(log_probs.shape)}")
    unbatched = log_probs.dim() == 2
    device = log_probs.device
    frames, frame_values = _lengths(input_lengths)
    labels, label_values = _lengths(target_lengths)
    if unbatched:
        log_probs, targets = log_probs[None], targets.reshape(1, -1)
    N, T, _ = log_probs.shape
    if frames.shape != (N,) or labels.shape != (N,):
        raise ValueError(f"input_lengths and target_lengths must have {N} entries")
    if targets.dim() not in (1, 2) or (targets.dim() == 2 and targets.size(0) != N):
        raise ValueError(f"targets must be of shape ({N}, S) or 1-D, got {tuple(targets.shape)}")
    # Lengths given as ints are checked in Python, which torch.compile reads as
    # constants; as tensors, only in eager mode, as their values would break the
    # graph.
    compiling = torch.compiler.is_compiling()
    if frame_values is not None:
        bad_frames = any(not 0 <= f <= T for f in frame_values)
    else:
        bad_frames = not compiling and bool(((frames < 0) | (frames > T)).any())
    if bad_frames:
        raise ValueError(f"input_lengths must be within 0 to {T}")
    if targets.dim() == 2:
        S = targets.size(1)
        if label_values is not None:
            bad_labels = any(not 0 <= n <= S for n in label_values)
        else:
            bad_labels = not compiling and bool(((labels < 0) | (labels > S)).any())
        if bad_labels:
            raise ValueError(f"target_lengths must be within 0 to {S}")
    else:
        if label_values is not None:
            bad_labels = min(label_values, default=0) < 0 or sum(label_values) != targets.numel()
        else:
            bad_labels = not compiling and bool(
                (labels < 0).any() or labels.sum() != targets.numel()
            )
        if bad_labels:
            raise ValueError(
                f"target_lengths must be non-negative and sum to the {targets.numel()} "
                "concatenated targets"
            )
    check_cuda_triton(name, log_probs)

    frames, labels = frames.to(device), labels.to(device)
    targets = targets.to(device, torch.long)
    if targets.dim() == 1:
        # Concatenated targets: each one's labels, in order, into a padded row.
        if label_values is not None:
            width = max(label_values, default=0)
        else:
            width = int(labels.max()) if N else 0
        padded = targets.new_zeros(N, width)
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


def _grid(log_probs, states, skip, frames, labels, viterbi):
    """The kernels' grid from each frame's log-probabilities of the states
    (B, 2S + 1), gathered from ``log_probs`` (B, T, C): with ``viterbi`` its
    cells' best steps back (else None), and -log p of the paths ending in each
    sequence's final blank and in its last label, rows 2S + 1 and 2S, (2, B):
    of all paths, or with ``viterbi`` the best one."""
    # Imported here so that philtorch.align imports without Triton.
    from ._dtw_kernels import ctc_viterbi, dtw_dp

    # (N, 2S + 1, T), in bits, as the kernels' soft-min is in base 2: one
    # multiply both negates and converts.
    states = states[:, None].expand(-1, log_probs.size(1), -1)
    # Half and bfloat16 log-probabilities accumulate in float32.
    acc = torch.promote_types(log_probs.dtype, torch.float32)
    cost = log_probs.gather(2, states).mT.to(acc) * (-1 / math.log(2))
    # A virtual first state and frame, at no cost where they meet, from which
    # both the first blank and the first label begin.
    cost = F.pad(cost, (1, 0, 1, 0), value=float("inf"))
    cost[:, 0, 0] = 0.0
    if viterbi:
        D, steps_back = ctc_viterbi(cost, skip)
    else:
        D, steps_back = dtw_dp(cost, skip, True, "ctc", 1.0), None
    batch = torch.arange(D.size(0), device=D.device)
    ends = torch.stack([D[batch, 2 * labels + 1, frames], D[batch, 2 * labels, frames]])
    return steps_back, ends * math.log(2)
