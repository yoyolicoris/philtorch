"""Hidden Markov model inference parallelized over time (prototype).

The forward algorithm, forward-backward smoothing and Viterbi decoding as
associative scans of per-step K x K matrices in log space, following Hassan,
Särkkä and García-Fernández, "Temporal parallelization of inference in hidden
Markov models" (IEEE TSP, 2021). Each function also has a sequential method,
the classical recursion, for comparison.

The model follows :mod:`philtorch.estimation`'s Kalman filter: the prior
describes z[0], the state before the first step, step n moves to z[n + 1]
through ``log_trans[n]``, and ``log_emit[:, n]`` scores y[n] against z[n + 1].
So all inputs have N steps, and the outputs describe z[1], ..., z[N]. A
classical HMM, whose initial distribution is that of the first state that
emits, is the case ``log_trans[0][i, :] = log_init`` for every i.
"""

from typing import Literal

import torch
from torch import Tensor

try:
    from torch._higher_order_ops.associative_scan import associative_scan
except ImportError:  # pragma: no cover - PyTorch without associative_scan
    associative_scan = None

Method = Literal["parallel", "sequential"]


def _log_matmul(a: Tensor, b: Tensor) -> Tensor:
    """(a ⊗ b)[i, k] = logsumexp_j a[i, j] + b[j, k], as a matrix multiply.

    Each row of a and column of b is shifted by its maximum before
    exponentiating, so the product needs no K^3 temporary. A sum still
    underflows to zero when all its terms are more than about 87 (float32) or
    708 (float64) below those maxima, which turns a finite result into -inf.
    """
    a_max = a.amax(-1, keepdim=True)
    b_max = b.amax(-2, keepdim=True)
    # A row or column of -inf would make the shift -inf and the result NaN.
    a_max = torch.where(torch.isfinite(a_max), a_max, torch.zeros_like(a_max))
    b_max = torch.where(torch.isfinite(b_max), b_max, torch.zeros_like(b_max))
    return torch.log(torch.exp(a - a_max) @ torch.exp(b - b_max)) + a_max + b_max


def _max_matmul(a: Tensor, b: Tensor) -> Tensor:
    """(a ⊗ b)[i, k] = max_j a[i, j] + b[j, k]."""
    return (a.unsqueeze(-1) + b.unsqueeze(-3)).amax(dim=-2)


def _scan(combine_fn, x: Tensor, reverse: bool = False) -> Tensor:
    """Inclusive scan over dimension 1; in reverse, combine_fn gets (later, earlier)."""
    if x.size(1) == 0:
        return x
    return associative_scan(combine_fn, x, dim=1, reverse=reverse, combine_mode="generic")


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
    return log_trans.expand(batch_size, N, K, K), log_init.expand(batch_size, K)


def _step_matrices(log_emit: Tensor, log_trans: Tensor, n: int | slice = slice(None)) -> Tensor:
    """M[n][i, j] = log p(z[n + 1] = j, y[n] | z[n] = i).

    All steps make a (B, N, K, K) tensor, which only the parallel scans need;
    the sequential recursions take one step at a time.
    """
    return log_trans[:, n] + log_emit[:, n].unsqueeze(-2)


def _fold_prior(M: Tensor, log_init: Tensor, reduce) -> Tensor:
    """Fold the prior into the first step: its rows all become that step's message.

    Then every prefix product has equal rows, and its first row is the
    forward message.
    """
    first = reduce(log_init.unsqueeze(-1) + M[:, 0], dim=-2)
    return torch.cat([first.unsqueeze(-2).expand_as(M[:, 0]).unsqueeze(1), M[:, 1:]], dim=1)


def _forward_messages(
    log_emit: Tensor, log_trans: Tensor, log_init: Tensor, method: Method
) -> Tensor:
    """alpha[n][j] = log p(y[0..n], z[n + 1] = j), of shape (B, N, K)."""
    if method == "sequential":
        alpha, out = log_init, []
        for n in range(log_emit.size(1)):
            step = _step_matrices(log_emit, log_trans, n)
            alpha = torch.logsumexp(alpha.unsqueeze(-1) + step, dim=-2)
            out.append(alpha)
        return torch.stack(out, dim=1)
    M = _step_matrices(log_emit, log_trans)
    scanned = _scan(_log_matmul, _fold_prior(M, log_init, torch.logsumexp))
    return scanned[..., 0, :]


def _backward_messages(log_emit: Tensor, log_trans: Tensor, method: Method) -> Tensor:
    """beta[n][i] = log p(y[n + 1..N - 1] | z[n + 1] = i), of shape (B, N, K)."""
    batch_size, N, K = log_emit.shape
    last = log_emit.new_zeros(batch_size, 1, K)
    if method == "sequential":
        beta, out = last[:, 0], [last[:, 0]]
        for n in range(N - 1, 0, -1):
            step = _step_matrices(log_emit, log_trans, n)
            beta = torch.logsumexp(step + beta.unsqueeze(-2), dim=-1)
            out.append(beta)
        return torch.stack(out[::-1], dim=1)
    M = _step_matrices(log_emit[:, 1:], log_trans[:, 1:])
    # suffix[n] = M[n + 1] ⊗ ... ⊗ M[N - 1]; its row sums are the messages.
    suffix = _scan(lambda later, earlier: _log_matmul(earlier, later), M, reverse=True)
    return torch.cat([torch.logsumexp(suffix, dim=-1), last], dim=1)


def hmm_forward(
    log_emit: Tensor,
    log_trans: Tensor,
    log_init: Tensor,
    *,
    method: Method = "parallel",
) -> tuple[Tensor, Tensor]:
    """Filter a hidden Markov model: its log-likelihood and filtered state probabilities.

    Args:
        log_emit (Tensor): log p(y[n] | z[n + 1] = k), of shape (B, N, K).
        log_trans (Tensor): log p(z[n + 1] = j | z[n] = i) at index [i, j], of
            shape (K, K), (N, K, K) or (B, N, K, K).
        log_init (Tensor): log p(z[0] = k), of shape (K,) or (B, K).
        method (str): ``"parallel"`` for associative scans, ``"sequential"``
            for the classical recursion. The parallel scans multiply the
            per-step matrices in log space with rescaled matrix multiplies,
            which underflow to -inf when every term of a sum is more than
            about 87 (float32) below its row's and column's largest terms.

    Returns:
        tuple of Tensor: log p(y[0..N - 1]), of shape (B,), and the filtered
        log-probabilities log p(z[n + 1] | y[0..n]), of shape (B, N, K).
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    alpha = _forward_messages(log_emit, log_trans, log_init, method)
    norm = torch.logsumexp(alpha, dim=-1, keepdim=True)
    return norm[:, -1, 0], alpha - norm


def hmm_posteriors(
    log_emit: Tensor,
    log_trans: Tensor,
    log_init: Tensor,
    *,
    method: Method = "parallel",
) -> tuple[Tensor, Tensor]:
    """Smooth a hidden Markov model: forward-backward state posteriors.

    The arguments are those of :func:`hmm_forward`. The parallel method runs a
    forward and a reverse scan, which are independent.

    Returns:
        tuple of Tensor: log p(y[0..N - 1]), of shape (B,), and the posterior
        log-probabilities log p(z[n + 1] | y[0..N - 1]), of shape (B, N, K).
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    alpha = _forward_messages(log_emit, log_trans, log_init, method)
    beta = _backward_messages(log_emit, log_trans, method)
    log_likelihood = torch.logsumexp(alpha[:, -1], dim=-1)
    # Normalize each step by its own sum rather than by the likelihood: the
    # messages grow to thousands over long inputs, and most of their rounding
    # error is shared by all states at a step, so this cancels it.
    joint = alpha + beta
    return log_likelihood, joint - torch.logsumexp(joint, dim=-1, keepdim=True)


def viterbi(
    log_emit: Tensor,
    log_trans: Tensor,
    log_init: Tensor,
    *,
    method: Method = "parallel",
) -> tuple[Tensor, Tensor]:
    """Decode a hidden Markov model: its most probable state sequence.

    The arguments are those of :func:`hmm_forward`. The parallel method is
    Hassan et al.'s max-product form: a forward scan gives the best score of
    any path ending in each state, a reverse scan the best score of any
    continuation, and each step takes the state whose sum is largest. That is
    the exact Viterbi path when it is unique; with ties, the steps can pick
    states from different optimal paths. The sequential method backtracks.

    Returns:
        tuple of Tensor: the best joint log-probability
        max log p(z[1..N], y[0..N - 1]), of shape (B,), and the states
        z[1], ..., z[N] of that path, of shape (B, N).
    """
    log_trans, log_init = _parse(log_emit, log_trans, log_init)
    batch_size, N, K = log_emit.shape
    if method == "sequential":
        delta, pointers = log_init, []
        for n in range(N):
            scores = delta.unsqueeze(-1) + _step_matrices(log_emit, log_trans, n)
            delta, best = scores.max(dim=-2)
            pointers.append(best)
        score, state = delta.max(dim=-1)
        path = [state]
        for n in range(N - 1, 0, -1):
            state = pointers[n].gather(-1, state.unsqueeze(-1)).squeeze(-1)
            path.append(state)
        return score, torch.stack(path[::-1], dim=1)

    M = _step_matrices(log_emit, log_trans)
    delta = _scan(_max_matmul, _fold_prior(M, log_init, lambda t, dim: t.amax(dim=dim)))[..., 0, :]
    suffix = _scan(lambda later, earlier: _max_matmul(earlier, later), M[:, 1:], reverse=True)
    future = torch.cat([suffix.amax(dim=-1), M.new_zeros(batch_size, 1, K)], dim=1)
    return delta[:, -1].amax(dim=-1), (delta + future).argmax(dim=-1)
