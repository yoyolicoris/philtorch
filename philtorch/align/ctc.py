"""Connectionist temporal classification (CTC) as soft-DTW."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .._triton import check_cuda_triton


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor | None = None,
    target_lengths: Tensor | None = None,
    blank: int = 0,
) -> Tensor:
    r"""The CTC loss of each sequence: minus the log-probability of its target.

    For per-frame log-probabilities :math:`\log p(k \mid n)` over :math:`C`
    classes and a target of :math:`U` labels, `Connectionist Temporal
    Classification`_ (Graves et al., 2006) sums the probabilities of every
    path of :math:`N` frames that reads as the target once repeats are merged
    and blanks dropped. That sum is soft-DTW with :math:`\gamma = 1` between
    the frames and the target's :math:`2U + 1` states, its labels with a
    blank before, between and after them, at the cost :math:`-\log p(k \mid
    n)` of the state's class at each frame: each frame stays in its state,
    advances to the next or skips a blank between two different labels. So
    it runs on the kernels of :func:`dtw`, one state after another, each a
    parallel scan over the frames: :math:`2U + 1` sequential steps and
    :math:`O(NU)` work and memory.

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
        log_probs (Tensor): :math:`\log p(k \mid n)`, of shape
            :math:`(B, N, C)`, such as a ``log_softmax`` over the classes.
        targets (Tensor): the labels, an integer tensor of shape
            :math:`(B, U)`, padded past each target's length; not
            :attr:`blank`.
        input_lengths (Tensor, optional): each sequence's number of frames,
            of shape :math:`(B)`. Default: :math:`N`.
        target_lengths (Tensor, optional): each target's number of labels,
            of shape :math:`(B)`. Default: :math:`U`.
        blank (int): the blank class. Default: 0.

    Returns:
        Tensor: the losses, of shape :math:`(B)`; ``inf`` for a target that
        no path of its frames can read.

    Raises:
        ValueError: if :attr:`log_probs` is not a CUDA tensor of shape
            :math:`(B, N, C)` or :attr:`targets` not of shape :math:`(B, U)`.
        RuntimeError: if Triton is not installed.

    Example::

        >>> import torch
        >>> from philtorch.align import ctc_loss
        >>> logits = torch.randn(4, 100, 20, device="cuda", requires_grad=True)
        >>> targets = torch.randint(1, 20, (4, 30), device="cuda")
        >>> loss = ctc_loss(logits.log_softmax(-1), targets)
        >>> loss.mean().backward()

    .. _Connectionist Temporal Classification:
        https://doi.org/10.1145/1143844.1143891
    """
    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must be (B, N, C), got {tuple(log_probs.shape)}")
    B, N, _ = log_probs.shape
    if targets.dim() != 2 or targets.size(0) != B:
        raise ValueError(f"targets must be of shape (B, U) = ({B}, U), got {tuple(targets.shape)}")
    check_cuda_triton("ctc_loss", log_probs)
    # Imported here so that philtorch.align imports without Triton.
    from ._dtw_kernels import dtw_dp

    device = log_probs.device
    targets = targets.to(device, torch.long)
    U = targets.size(1)
    frames = torch.full((B,), N, device=device) if input_lengths is None else input_lengths
    labels = torch.full((B,), U, device=device) if target_lengths is None else target_lengths
    frames, labels = frames.to(device, torch.long), labels.to(device, torch.long)

    # The states: the labels with a blank before, between and after them. A
    # path may skip a blank into a label that differs from the one before.
    states = targets.new_full((B, 2 * U + 1), blank)
    states[:, 1::2] = targets
    skip = torch.zeros_like(states, dtype=torch.bool)
    skip[:, 1] = True
    skip[:, 3::2] = targets[:, 1:] != targets[:, :-1]
    # (B, 2U + 1, N), in bits, as the kernels' soft-min is in base 2: one
    # multiply both negates and converts.
    cost = log_probs.gather(2, states[:, None].expand(B, N, -1)).mT * (-1 / math.log(2))
    # A virtual first state and frame, at no cost where they meet, from which
    # both the first blank and the first label begin.
    cost = F.pad(cost, (1, 0, 1, 0), value=float("inf"))
    cost[:, 0, 0] = 0.0
    skip = F.pad(skip, (1, 0)).to(torch.int8)
    D = dtw_dp(cost, skip, True, "ctc", 1.0)
    # A path ends in the last label or the blank after it, at the last frame.
    batch = torch.arange(B, device=device)
    ends = torch.stack([D[batch, 2 * labels + 1, frames], D[batch, 2 * labels, frames]])
    ends = ends * math.log(2)
    none = ends.isinf().all(0)
    loss = -torch.logsumexp(-torch.where(none, 0.0, ends), dim=0)
    return torch.where(none, float("inf"), loss)
