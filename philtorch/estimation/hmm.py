"""Hidden Markov model inference parallelized over time.

The forward algorithm, forward-backward smoothing and Viterbi decoding as
scans over time of per-step K x K matrices in the log or max-plus semiring,
following Hassan, Särkkä and García-Fernández, "Temporal parallelization of
inference in hidden Markov models" (IEEE TSP, 2021).

The model is the classical one, as in
:func:`~philtorch.estimation.kalman_filter`: the prior describes the first
state z[0], ``log_emit[:, n]`` scores y[n] against z[n], and ``log_trans[n]``
moves z[n] to z[n + 1]. So there are N states and N - 1 transitions, and the
outputs describe z[0], ..., z[N - 1].

Two implementations, chosen per call:

* Without gradients and with K <= 32 states, a chunked scan in Triton
  kernels (:mod:`._hmm_kernels`): each chunk of steps is multiplied in one
  program, so a call takes a few kernel launches whatever N is.
* With gradients, or more states, a scan of whole matrices whose log-space
  and max-plus products (:mod:`._semiring`) are differentiable to any order.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor


def _parse(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    """Validate the model: log_trans of its N - 1 transitions, log_emit of its N states."""
    assert log_emit.dim() == 3, f"log_emit must be (B, N, K), got {tuple(log_emit.shape)}"
    batch_size, N, K = log_emit.shape
    transitions = max(N - 1, 0)
    assert log_trans.shape[-2:] == (K, K), f"log_trans must end in {(K, K)}"
    match log_trans.dim():
        case 2:
            pass
        case 3 if log_trans.size(0) == transitions:
            pass
        case 4 if log_trans.shape[:2] == (batch_size, transitions):
            pass
        case _:
            raise ValueError(
                f"log_trans must be of shape {(K, K)}, {(transitions, K, K)} or "
                f"{(batch_size, transitions, K, K)}, got {tuple(log_trans.shape)}"
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
    return log_trans.expand(batch_size, transitions, K, K), log_init.expand(batch_size, K)


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
    """alpha[n][j] = log p(y[0..n], z[n] = j), or its max over paths, (B, N, K)."""
    first = log_init + log_emit[:, 0]
    if _use_chunked(log_emit.size(-1), log_emit, log_trans, log_init):
        from ._hmm_kernels import chain

        alpha = log_emit.new_empty(log_emit.shape)
        alpha[:, 0] = first
        chain(first, log_trans, log_emit[:, 1:].contiguous(), is_max, out=alpha[:, 1:])
        return alpha
    # M[n][i, j] = log p(z[n + 1] = j, y[n + 1] | z[n] = i). A first element
    # whose rows all equal the first message makes every prefix product's
    # rows equal too: the forward messages.
    M = log_trans + log_emit[:, 1:].unsqueeze(-2)
    K = first.size(-1)
    M = torch.cat([first[:, None, None].expand(-1, 1, K, K), M], dim=1)
    return _scan(_matmul(is_max), M)[..., 0, :]


def _backward_messages(log_emit: Tensor, log_trans: Tensor, is_max: bool) -> Tensor:
    """beta[n][i] = log p(y[n + 1..N - 1] | z[n] = i), or its max, (B, N, K)."""
    batch_size, N, K = log_emit.shape
    last = log_emit.new_zeros(batch_size, 1, K)
    if N < 2:
        return last[:, :N]
    if _use_chunked(K, log_emit, log_trans):
        from ._hmm_kernels import chain

        # beta[n] = M[n] (x) beta[n + 1]: a chain from the last step back,
        # through each matrix's transpose.
        beta = log_emit.new_empty(batch_size, N, K)
        beta[:, -1] = 0
        chain(
            last[:, 0],
            log_trans,
            log_emit[:, 1:].contiguous(),
            is_max,
            reverse=True,
            transpose=True,
            out=beta[:, :-1],
        )
        return beta
    M = log_trans + log_emit[:, 1:].unsqueeze(-2)
    product = _matmul(is_max)
    # suffix[n] = M[n] (x) ... (x) M[N - 2]; its row reductions are the messages.
    suffix = _scan(lambda later, earlier: product(earlier, later), M, reverse=True)
    reduce = _amax if is_max else _logsumexp
    return torch.cat([reduce(suffix, dim=-1), last], dim=1)


class HMMFilterResult(NamedTuple):
    """The result of :func:`hmm_filter`."""

    #: Filtered log-probabilities :math:`\log p(z[n] = k \mid y[0], \dots, y[n])`, of shape
    #: :math:`(B, N, K)`.
    log_probs: Tensor
    #: Each sequence's log marginal likelihood :math:`\log p(y)`, of shape :math:`(B)`.
    log_likelihood: Tensor


class HMMSmootherResult(NamedTuple):
    """The result of :func:`hmm_smoother`."""

    #: Posterior log-probabilities :math:`\log p(z[n] = k \mid y[0], \dots, y[N - 1])`, of
    #: shape :math:`(B, N, K)`.
    log_probs: Tensor
    #: Each sequence's log marginal likelihood :math:`\log p(y)`, of shape :math:`(B)`.
    log_likelihood: Tensor


class HMMViterbiResult(NamedTuple):
    """The result of :func:`hmm_viterbi`."""

    #: The most probable states :math:`z[0], \dots, z[N - 1]`, of shape :math:`(B, N)`.
    path: Tensor
    #: That path's joint log-probability :math:`\max_z \log p(z, y)`, of shape :math:`(B)`.
    score: Tensor


def hmm_filter(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> HMMFilterResult:
    r"""Filter a hidden Markov model: its log-likelihood and filtered state probabilities.

    For a hidden Markov model with states :math:`z[0], \dots, z[N - 1]` and
    measurements :math:`y[0], \dots, y[N - 1]`, where :math:`y[n]` is emitted
    from :math:`z[n]` and :math:`N - 1` transitions move :math:`z[n]` to
    :math:`z[n + 1]`, this returns :math:`\log p(y[0], \dots, y[N - 1])` and
    :math:`\log p(z[n] \mid y[0], \dots, y[n])`: the forward algorithm,
    computed as a scan over time instead of step by step, as in `Temporal
    Parallelization of Inference in Hidden Markov Models`_ (Hassan, Särkkä
    and García-Fernández, 2021).

    The results are differentiable to any order with respect to every input,
    so the log-likelihood can train the model's probabilities, or a network
    that predicts them; its gradient with respect to :attr:`log_emit` is the
    posteriors of :func:`hmm_smoother`. Without gradients and with at most 32
    states, a chunked scan runs in a few Triton kernels; with gradients or
    more states, a scan of whole matrices runs O(log N) rounds of
    differentiable log-space matrix products, with O(N K^3) work.

    Args:
        log_emit (Tensor): :math:`\log p(y[n] \mid z[n] = k)`, of shape
            :math:`(B, N, K)`, on a CUDA device.
        log_trans (Tensor): :math:`\log p(z[n + 1] = j \mid z[n] = i)` at
            index :math:`[i, j]`, of shape :math:`(K, K)`,
            :math:`(N - 1, K, K)` or :math:`(B, N - 1, K, K)`.
        log_init (Tensor): :math:`\log p(z[0] = k)`, of shape :math:`(K)` or
            :math:`(B, K)`.

    Returns:
        HMMFilterResult: the filtered log-probabilities, of shape
        :math:`(B, N, K)`, and the log-likelihood, of shape :math:`(B)`.

    Raises:
        ValueError: if the inputs are not CUDA tensors or :attr:`log_trans`
            has an unsupported shape.

    .. _Temporal Parallelization of Inference in Hidden Markov Models:
        https://doi.org/10.1109/TSP.2021.3103338
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    if log_emit.size(1) == 0:
        return HMMFilterResult(log_emit, log_emit.new_zeros(log_emit.size(0)))
    alpha = _forward_messages(log_emit, log_trans, log_init, is_max=False)
    norm = _logsumexp(alpha, dim=-1, keepdim=True)
    return HMMFilterResult(alpha - norm, norm[:, -1, 0])


def hmm_smoother(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> HMMSmootherResult:
    r"""Smooth a hidden Markov model: forward-backward state posteriors.

    For the model of :func:`hmm_filter`, this returns the log-likelihood and
    :math:`\log p(z[n] \mid y[0], \dots, y[N - 1])`, from a forward and an
    independent backward scan, as in `Temporal Parallelization of Inference
    in Hidden Markov Models`_ (Hassan et al., 2021). The arguments,
    implementations and differentiability are those of :func:`hmm_filter`.

    Args:
        log_emit (Tensor): :math:`\log p(y[n] \mid z[n] = k)`, of shape
            :math:`(B, N, K)`, on a CUDA device.
        log_trans (Tensor): :math:`\log p(z[n + 1] = j \mid z[n] = i)` at
            index :math:`[i, j]`, of shape :math:`(K, K)`,
            :math:`(N - 1, K, K)` or :math:`(B, N - 1, K, K)`.
        log_init (Tensor): :math:`\log p(z[0] = k)`, of shape :math:`(K)` or
            :math:`(B, K)`.

    Returns:
        HMMSmootherResult: the posterior log-probabilities, of shape
        :math:`(B, N, K)`, and the log-likelihood, of shape :math:`(B)`.

    Raises:
        ValueError: if the inputs are not CUDA tensors or :attr:`log_trans`
            has an unsupported shape.

    .. _Temporal Parallelization of Inference in Hidden Markov Models:
        https://doi.org/10.1109/TSP.2021.3103338
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    if log_emit.size(1) == 0:
        return HMMSmootherResult(log_emit, log_emit.new_zeros(log_emit.size(0)))
    alpha = _forward_messages(log_emit, log_trans, log_init, is_max=False)
    beta = _backward_messages(log_emit, log_trans, is_max=False)
    log_likelihood = _logsumexp(alpha[:, -1], dim=-1)
    # Normalize each step by its own sum rather than by the likelihood: the
    # messages grow to thousands over long inputs, and most of their rounding
    # error is shared by all states at a step, so this cancels it.
    joint = alpha + beta
    return HMMSmootherResult(joint - _logsumexp(joint, dim=-1, keepdim=True), log_likelihood)


def hmm_viterbi(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> HMMViterbiResult:
    r"""Decode a hidden Markov model: its most probable state sequence.

    For the model of :func:`hmm_filter`, this is the max-product form of the
    Viterbi algorithm from `Temporal Parallelization of Inference in Hidden
    Markov Models`_ (Hassan et al., 2021): a forward scan gives the best
    score of any path ending in each state, a backward scan the best score
    of any continuation, and each step takes the state whose sum is largest.
    That is the exact Viterbi path when it is unique; with ties, the steps
    can pick states from different optimal paths. The score is
    differentiable to any order; the implementations are those of
    :func:`hmm_filter`.

    Args:
        log_emit (Tensor): :math:`\log p(y[n] \mid z[n] = k)`, of shape
            :math:`(B, N, K)`, on a CUDA device.
        log_trans (Tensor): :math:`\log p(z[n + 1] = j \mid z[n] = i)` at
            index :math:`[i, j]`, of shape :math:`(K, K)`,
            :math:`(N - 1, K, K)` or :math:`(B, N - 1, K, K)`.
        log_init (Tensor): :math:`\log p(z[0] = k)`, of shape :math:`(K)` or
            :math:`(B, K)`.

    Returns:
        HMMViterbiResult: the states :math:`z[0], \dots, z[N - 1]` of the
        most probable path, of shape :math:`(B, N)`, and its joint
        log-probability
        :math:`\max \log p(z[0], \dots, z[N - 1], y[0], \dots, y[N - 1])`,
        of shape :math:`(B)`.

    Raises:
        ValueError: if the inputs are not CUDA tensors or :attr:`log_trans`
            has an unsupported shape.

    .. _Temporal Parallelization of Inference in Hidden Markov Models:
        https://doi.org/10.1109/TSP.2021.3103338
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    batch_size, N, _ = log_emit.shape
    if N == 0:
        path = log_emit.new_zeros(batch_size, 0, dtype=torch.long)
        return HMMViterbiResult(path, log_emit.new_zeros(batch_size))
    delta = _forward_messages(log_emit, log_trans, log_init, is_max=True)
    future = _backward_messages(log_emit, log_trans, is_max=True)
    return HMMViterbiResult((delta + future).argmax(dim=-1), delta[:, -1].amax(dim=-1))
