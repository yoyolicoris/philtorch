"""Hidden Markov model inference parallelized over time (prototype).

The forward algorithm, forward-backward smoothing and Viterbi decoding as
associative scans of per-step K x K matrices in log space, following Hassan,
Särkkä and García-Fernández, "Temporal parallelization of inference in hidden
Markov models" (IEEE TSP, 2021). The log-space and max-plus matrix products
are Helion kernels, so these need CUDA tensors.

The model follows :mod:`philtorch.estimation`'s Kalman filter: the prior
describes z[0], the state before the first step, step n moves to z[n + 1]
through ``log_trans[n]``, and ``log_emit[:, n]`` scores y[n] against z[n + 1].
So all inputs have N steps, and the outputs describe z[1], ..., z[N]. A
classical HMM, whose initial distribution is that of the first state that
emits, is the case ``log_trans[0][i, :] = log_init`` for every i.
"""

import math

import torch
from torch import Tensor

from ._semiring_helion import log_bmm, max_bmm


def _batched(op):
    """Apply a (P, K, K) x (P, K, K) product to (B, n, K, K) tensors."""

    def apply(a: Tensor, b: Tensor) -> Tensor:
        lead = a.shape[:-2]
        # Explicit sizes, not -1: under vmap an empty batch makes -1 ambiguous.
        n_p = math.prod(lead)
        out = op(a.reshape(n_p, *a.shape[-2:]), b.reshape(n_p, *b.shape[-2:]))
        return out.reshape(*lead, a.size(-2), b.size(-1))

    return apply


# Exact log-space and max-plus matrix products, as Helion kernels (CUDA only).
_log_matmul = _batched(log_bmm)
_max_matmul = _batched(max_bmm)


def _scan(combine_fn, x: Tensor, reverse: bool = False) -> Tensor:
    """Inclusive scan of (B, N, K, K) matrices over dimension 1.

    combine_fn(earlier, later) combines adjacent prefixes; in reverse it gets
    (later, earlier), as in torch's associative_scan. This is the odd/even
    recursion of torch's generic associative_scan, O(N) combines in O(log N)
    rounds, but without its vmap: the products see whole batches, which is
    over twice as fast when launch overhead dominates.
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
        raise ValueError("The HMM scans run Helion kernels, which need CUDA tensors.")
    return log_trans.expand(batch_size, N, K, K), log_init.expand(batch_size, K)


def _step_matrices(log_emit: Tensor, log_trans: Tensor) -> Tensor:
    """M[n][i, j] = log p(z[n + 1] = j, y[n] | z[n] = i), of shape (B, N, K, K)."""
    return log_trans + log_emit.unsqueeze(-2)


def _fold_prior(M: Tensor, log_init: Tensor, reduce) -> Tensor:
    """Fold the prior into the first step: its rows all become that step's message.

    Then every prefix product has equal rows, and its first row is the
    forward message.
    """
    first = reduce(log_init.unsqueeze(-1) + M[:, 0], dim=-2)
    return torch.cat([first.unsqueeze(-2).expand_as(M[:, 0]).unsqueeze(1), M[:, 1:]], dim=1)


def _forward_messages(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> Tensor:
    """alpha[n][j] = log p(y[0..n], z[n + 1] = j), of shape (B, N, K)."""
    M = _step_matrices(log_emit, log_trans)
    scanned = _scan(_log_matmul, _fold_prior(M, log_init, torch.logsumexp))
    return scanned[..., 0, :]


def _backward_messages(log_emit: Tensor, log_trans: Tensor) -> Tensor:
    """beta[n][i] = log p(y[n + 1..N - 1] | z[n + 1] = i), of shape (B, N, K)."""
    batch_size, _, K = log_emit.shape
    M = _step_matrices(log_emit[:, 1:], log_trans[:, 1:])
    # suffix[n] = M[n + 1] ⊗ ... ⊗ M[N - 1]; its row sums are the messages.
    suffix = _scan(lambda later, earlier: _log_matmul(earlier, later), M, reverse=True)
    return torch.cat([torch.logsumexp(suffix, dim=-1), log_emit.new_zeros(batch_size, 1, K)], dim=1)


def hmm_forward(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    """Filter a hidden Markov model: its log-likelihood and filtered state probabilities.

    The forward messages are a scan of the per-step matrices, multiplied
    exactly in log space with Helion kernels, so the inputs must be CUDA
    tensors.

    Args:
        log_emit (Tensor): log p(y[n] | z[n + 1] = k), of shape (B, N, K).
        log_trans (Tensor): log p(z[n + 1] = j | z[n] = i) at index [i, j], of
            shape (K, K), (N, K, K) or (B, N, K, K).
        log_init (Tensor): log p(z[0] = k), of shape (K,) or (B, K).

    Returns:
        tuple of Tensor: log p(y[0..N - 1]), of shape (B,), and the filtered
        log-probabilities log p(z[n + 1] | y[0..n]), of shape (B, N, K).
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    alpha = _forward_messages(log_emit, log_trans, log_init)
    norm = torch.logsumexp(alpha, dim=-1, keepdim=True)
    return norm[:, -1, 0], alpha - norm


def hmm_posteriors(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    """Smooth a hidden Markov model: forward-backward state posteriors.

    The arguments are those of :func:`hmm_forward`. A forward and a reverse
    scan, which are independent, give the two messages.

    Returns:
        tuple of Tensor: log p(y[0..N - 1]), of shape (B,), and the posterior
        log-probabilities log p(z[n + 1] | y[0..N - 1]), of shape (B, N, K).
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    alpha = _forward_messages(log_emit, log_trans, log_init)
    beta = _backward_messages(log_emit, log_trans)
    log_likelihood = torch.logsumexp(alpha[:, -1], dim=-1)
    # Normalize each step by its own sum rather than by the likelihood: the
    # messages grow to thousands over long inputs, and most of their rounding
    # error is shared by all states at a step, so this cancels it.
    joint = alpha + beta
    return log_likelihood, joint - torch.logsumexp(joint, dim=-1, keepdim=True)


def viterbi(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    """Decode a hidden Markov model: its most probable state sequence.

    The arguments are those of :func:`hmm_forward`. This is Hassan et al.'s
    max-product form: a forward scan gives the best score of any path ending
    in each state, a reverse scan the best score of any continuation, and each
    step takes the state whose sum is largest. That is the exact Viterbi path
    when it is unique; with ties, the steps can pick states from different
    optimal paths.

    Returns:
        tuple of Tensor: the best joint log-probability
        max log p(z[1..N], y[0..N - 1]), of shape (B,), and the states
        z[1], ..., z[N] of that path, of shape (B, N).
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    batch_size, _, K = log_emit.shape
    M = _step_matrices(log_emit, log_trans)
    delta = _scan(_max_matmul, _fold_prior(M, log_init, lambda t, dim: t.amax(dim=dim)))[..., 0, :]
    suffix = _scan(lambda later, earlier: _max_matmul(earlier, later), M[:, 1:], reverse=True)
    future = torch.cat([suffix.amax(dim=-1), M.new_zeros(batch_size, 1, K)], dim=1)
    return delta[:, -1].amax(dim=-1), (delta + future).argmax(dim=-1)
