"""Hidden Markov model inference parallelized over time.

The forward algorithm, forward-backward smoothing and Viterbi decoding as
scans over time of per-step K x K matrices in the log or max-plus semiring,
following Hassan, Särkkä and García-Fernández, "Temporal parallelization of
inference in hidden Markov models" (IEEE TSP, 2021).

The model follows :func:`~philtorch.estimation.kalman_filter`'s convention:
the prior describes z[0], the state before the first step; step n moves to
z[n + 1] through ``log_trans[n]``, and ``log_emit[:, n]`` scores y[n] against
z[n + 1]. So all inputs have N steps, and the outputs describe z[1], ...,
z[N]. A classical HMM, whose initial distribution is that of the first state
that emits, is the case ``log_trans[0][i, :] = log_init`` for every i.

Two implementations, chosen per call:

* Without gradients and with K <= 32 states, a chunked scan in Triton
  kernels (:mod:`._hmm_kernels`): each chunk of steps is multiplied in one
  program, so a call takes a few kernel launches whatever N is.
* With gradients, or more states, a scan of whole matrices whose log-space
  and max-plus products (:mod:`._semiring`) are differentiable to any order.
"""

import math

import torch
from torch import Tensor


def _parse(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    assert log_emit.dim() == 3, f"log_emit must be (B, N, K), got {tuple(log_emit.shape)}"
    batch_size, N, K = log_emit.shape
    assert log_trans.shape[-2:] == (K, K), f"log_trans must end in {(K, K)}"
    match log_trans.dim():
        case 2:
            pass
        case 3 if log_trans.size(0) == N:
            pass
        case 4 if log_trans.shape[:2] == (batch_size, N):
            pass
        case _:
            raise ValueError(
                f"log_trans must be of shape {(K, K)}, {(N, K, K)} or {(batch_size, N, K, K)}, "
                f"got {tuple(log_trans.shape)}"
            )
    assert log_init.shape in ((K,), (batch_size, K)), (
        f"log_init must be {(K,)} or {(batch_size, K)}, got {tuple(log_init.shape)}"
    )
    if not log_emit.is_cuda:
        raise ValueError("The HMM scans run Triton kernels, which need CUDA tensors.")
    # The kernels read each K x K matrix and each step's K emissions as
    # contiguous blocks; batch and time may broadcast.
    if log_trans.stride(-1) != 1 or log_trans.stride(-2) != K:
        log_trans = log_trans.contiguous()
    return log_trans.expand(batch_size, N, K, K), log_init.expand(batch_size, K)


def _logsumexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """torch.logsumexp with a zero gradient, not NaN, where all inputs are -inf.

    Unreachable states, such as those a left-to-right model starts outside,
    make whole slices -inf. Those slices are reduced over zeros instead and
    their result put back to -inf, so no NaN reaches the gradient.
    """
    empty = torch.isneginf(x).all(dim=dim, keepdim=True)
    out = torch.logsumexp(torch.where(empty, torch.zeros_like(x), x), dim=dim, keepdim=True)
    out = torch.where(empty, float("-inf"), out)
    return out if keepdim else out.squeeze(dim)


def _amax(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    return x.amax(dim=dim, keepdim=keepdim)


def _scan(combine_fn, x: Tensor, reverse: bool = False) -> Tensor:
    """Inclusive scan of (B, N, K, K) matrices over dimension 1.

    combine_fn(earlier, later) combines adjacent prefixes; in reverse it gets
    (later, earlier), as in torch's associative_scan. This is the odd/even
    recursion of torch's generic associative_scan, O(N) combines in O(log N)
    rounds, but without its vmap: the products see whole batches.
    """
    if reverse:
        # Scanning the flipped sequence hands combine_fn (later, earlier).
        return _scan(combine_fn, x.flip(1)).flip(1)
    N = x.size(1)
    if N < 2:
        return x
    # Combine adjacent pairs, scan the pairs, then fill in the even positions.
    odd = _scan(combine_fn, combine_fn(x[:, 0:-1:2], x[:, 1::2]))
    even = combine_fn(odd[:, : (N - 1) // 2], x[:, 2::2])
    out = torch.empty_like(x)
    out[:, 0] = x[:, 0]
    out[:, 2::2] = even
    out[:, 1::2] = odd
    return out


def _matmul(is_max: bool):
    """The (B, n, K, K) x (B, n, K, K) product of the semiring, differentiable."""
    from ._semiring import log_bmm, max_bmm

    op = max_bmm if is_max else log_bmm

    def apply(a: Tensor, b: Tensor) -> Tensor:
        lead = a.shape[:-2]
        # Explicit sizes, not -1: under vmap an empty batch makes -1 ambiguous.
        n_p = math.prod(lead)
        out = op(a.reshape(n_p, *a.shape[-2:]), b.reshape(n_p, *b.shape[-2:]))
        return out.reshape(*lead, a.size(-2), b.size(-1))

    return apply


def _use_chunked(K: int, *tensors: Tensor) -> bool:
    """Whether the chunked kernels, which have no autograd, can serve the call."""
    from ._hmm_kernels import MAX_STATES

    needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    # Forward-mode dual tensors don't set requires_grad.
    dual = any(torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in tensors)
    return K <= MAX_STATES and not needs_grad and not dual


def _forward_messages(
    log_emit: Tensor, log_trans: Tensor, log_init: Tensor, is_max: bool
) -> Tensor:
    """alpha[n][j] = log p(y[0..n], z[n + 1] = j), or its max over paths, (B, N, K)."""
    if _use_chunked(log_emit.size(-1), log_emit, log_trans, log_init):
        from ._hmm_kernels import chain

        return chain(log_init.contiguous(), log_trans, log_emit.contiguous(), is_max)
    reduce = _amax if is_max else _logsumexp
    M = log_trans + log_emit.unsqueeze(-2)  # M[n][i, j] = log p(z[n + 1] = j, y[n] | z[n] = i)
    # Fold the prior into the first step: its rows all become that step's
    # message, so every prefix product has equal rows, the forward messages.
    first = reduce(log_init.unsqueeze(-1) + M[:, 0], dim=-2)
    M = torch.cat([first.unsqueeze(-2).expand_as(M[:, 0]).unsqueeze(1), M[:, 1:]], dim=1)
    return _scan(_matmul(is_max), M)[..., 0, :]


def _backward_messages(log_emit: Tensor, log_trans: Tensor, is_max: bool) -> Tensor:
    """beta[n][i] = log p(y[n + 1..N - 1] | z[n + 1] = i), or its max, (B, N, K)."""
    batch_size, N, K = log_emit.shape
    last = log_emit.new_zeros(batch_size, 1, K)
    if N < 2:
        return last[:, :N]
    if _use_chunked(K, log_emit, log_trans):
        from ._hmm_kernels import chain

        # beta[n - 1] = M[n] (x) beta[n]: a chain from the last step back,
        # through each matrix's transpose.
        beta = chain(
            last[:, 0],
            log_trans[:, 1:],
            log_emit[:, 1:].contiguous(),
            is_max,
            reverse=True,
            transpose=True,
        )
        return torch.cat([beta, last], dim=1)
    M = log_trans[:, 1:] + log_emit[:, 1:].unsqueeze(-2)
    product = _matmul(is_max)
    # suffix[n] = M[n + 1] (x) ... (x) M[N - 1]; its row reductions are the messages.
    suffix = _scan(lambda later, earlier: product(earlier, later), M, reverse=True)
    reduce = _amax if is_max else _logsumexp
    return torch.cat([reduce(suffix, dim=-1), last], dim=1)


def hmm_filter(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    r"""Filter a hidden Markov model: its log-likelihood and filtered state probabilities.

    For a hidden Markov model with states :math:`z[0], \dots, z[N]` and
    measurements :math:`y[0], \dots, y[N - 1]`, where step :math:`n` moves
    :math:`z[n]` to :math:`z[n + 1]` and :math:`y[n]` is emitted from
    :math:`z[n + 1]`, this returns :math:`\log p(y[0], \dots, y[N - 1])` and
    :math:`\log p(z[n + 1] \mid y[0], \dots, y[n])`: the forward algorithm,
    computed as a scan over time instead of step by step.

    The results are differentiable to any order with respect to every input,
    so the log-likelihood can train the model's probabilities, or a network
    that predicts them; its gradient with respect to :attr:`log_emit` is the
    posteriors of :func:`hmm_smoother`. Without gradients and with at most 32
    states, a chunked scan runs in a few Triton kernels; with gradients or
    more states, a scan of whole matrices runs O(log N) rounds of
    differentiable log-space matrix products, with O(N K^3) work.

    Note:
        As in :func:`kalman_filter`, the prior is the state before the first
        step. A classical HMM, whose initial distribution is that of the
        first state that emits, is the case ``log_trans[0][i, :] = log_init``
        for every :math:`i`.

    Args:
        log_emit (Tensor): :math:`\log p(y[n] \mid z[n + 1] = k)`, of shape
            :math:`(B, N, K)`, on a CUDA device.
        log_trans (Tensor): :math:`\log p(z[n + 1] = j \mid z[n] = i)` at
            index :math:`[i, j]`, of shape :math:`(K, K)`, :math:`(N, K, K)`
            or :math:`(B, N, K, K)`.
        log_init (Tensor): :math:`\log p(z[0] = k)`, of shape :math:`(K)` or
            :math:`(B, K)`.

    Returns:
        tuple of Tensor: the log-likelihood, of shape :math:`(B)`, and the
        filtered log-probabilities, of shape :math:`(B, N, K)`.

    Raises:
        ValueError: if the inputs are not CUDA tensors or :attr:`log_trans`
            has an unsupported shape.
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    if log_emit.size(1) == 0:
        return log_emit.new_zeros(log_emit.size(0)), log_emit
    alpha = _forward_messages(log_emit, log_trans, log_init, is_max=False)
    norm = _logsumexp(alpha, dim=-1, keepdim=True)
    return norm[:, -1, 0], alpha - norm


def hmm_smoother(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    r"""Smooth a hidden Markov model: forward-backward state posteriors.

    For the model of :func:`hmm_filter`, this returns the log-likelihood and
    :math:`\log p(z[n + 1] \mid y[0], \dots, y[N - 1])`, from a forward and an
    independent backward scan. The arguments, implementations and
    differentiability are those of :func:`hmm_filter`.

    Args:
        log_emit (Tensor): :math:`\log p(y[n] \mid z[n + 1] = k)`, of shape
            :math:`(B, N, K)`, on a CUDA device.
        log_trans (Tensor): :math:`\log p(z[n + 1] = j \mid z[n] = i)` at
            index :math:`[i, j]`, of shape :math:`(K, K)`, :math:`(N, K, K)`
            or :math:`(B, N, K, K)`.
        log_init (Tensor): :math:`\log p(z[0] = k)`, of shape :math:`(K)` or
            :math:`(B, K)`.

    Returns:
        tuple of Tensor: the log-likelihood, of shape :math:`(B)`, and the
        posterior log-probabilities, of shape :math:`(B, N, K)`.

    Raises:
        ValueError: if the inputs are not CUDA tensors or :attr:`log_trans`
            has an unsupported shape.
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    if log_emit.size(1) == 0:
        return log_emit.new_zeros(log_emit.size(0)), log_emit
    alpha = _forward_messages(log_emit, log_trans, log_init, is_max=False)
    beta = _backward_messages(log_emit, log_trans, is_max=False)
    log_likelihood = _logsumexp(alpha[:, -1], dim=-1)
    # Normalize each step by its own sum rather than by the likelihood: the
    # messages grow to thousands over long inputs, and most of their rounding
    # error is shared by all states at a step, so this cancels it.
    joint = alpha + beta
    return log_likelihood, joint - _logsumexp(joint, dim=-1, keepdim=True)


def hmm_viterbi(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    r"""Decode a hidden Markov model: its most probable state sequence.

    For the model of :func:`hmm_filter`, this is Hassan et al.'s max-product
    form of the Viterbi algorithm: a forward scan gives the best score of
    any path ending in each state, a backward scan the best score of any
    continuation, and each step takes the state whose sum is largest. That
    is the exact Viterbi path when it is unique; with ties, the steps can
    pick states from different optimal paths. The score is differentiable to
    any order; the implementations are those of :func:`hmm_filter`.

    Args:
        log_emit (Tensor): :math:`\log p(y[n] \mid z[n + 1] = k)`, of shape
            :math:`(B, N, K)`, on a CUDA device.
        log_trans (Tensor): :math:`\log p(z[n + 1] = j \mid z[n] = i)` at
            index :math:`[i, j]`, of shape :math:`(K, K)`, :math:`(N, K, K)`
            or :math:`(B, N, K, K)`.
        log_init (Tensor): :math:`\log p(z[0] = k)`, of shape :math:`(K)` or
            :math:`(B, K)`.

    The path maximizes over every state, the prior state :math:`z[0]`
    included. For a classical HMM written with ``log_trans[0][i, :]`` set to
    its initial distribution, give :math:`z[0]` a point mass, such as
    ``log_init = [0, -inf, ..., -inf]`` with only ``log_trans[0][0, :]``
    set, so that the score is that of the classical Viterbi path.

    Returns:
        tuple of Tensor: the best joint log-probability
        :math:`\max \log p(z[0], \dots, z[N], y[0], \dots, y[N - 1])`, of
        shape :math:`(B)`, and the states :math:`z[1], \dots, z[N]` of that
        path, of shape :math:`(B, N)`.

    Raises:
        ValueError: if the inputs are not CUDA tensors or :attr:`log_trans`
            has an unsupported shape.
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    batch_size, N, _ = log_emit.shape
    if N == 0:
        return log_init.amax(-1), log_emit.new_zeros(batch_size, 0, dtype=torch.long)
    delta = _forward_messages(log_emit, log_trans, log_init, is_max=True)
    future = _backward_messages(log_emit, log_trans, is_max=True)
    return delta[:, -1].amax(dim=-1), (delta + future).argmax(dim=-1)
