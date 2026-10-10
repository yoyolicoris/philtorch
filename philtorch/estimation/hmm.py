"""Hidden Markov model inference parallelized over time.

The forward algorithm, forward-backward smoothing and Viterbi decoding as
parallel scans over time of per-step K x K matrices in the log or max-plus
semiring, following Hassan, Särkkä and García-Fernández, "Temporal
parallelization of inference in hidden Markov models" (IEEE TSP, 2021).

The model is the classical one, as in
:func:`~philtorch.estimation.kalman_filter`: the prior describes the first
state z[0], ``log_emit[:, n]`` scores y[n] against z[n], and ``log_trans[n]``
moves z[n] to z[n + 1]. So there are N states and N - 1 transitions, and the
outputs describe z[0], ..., z[N - 1].

Every message is a chain y[t] = y[t - 1] (x) M[t], computed by the chunked
parallel scan of :mod:`._hmm_kernels`, with M[t][i, j] = log_trans[i, j] +
log_emit[i], the emission of the state a transition leaves. So the forward
chain gives the predicted messages log p(y[0..n - 1], z[n]), and the
backward messages are the same chain over flipped time and transposed
matrices.

A log chain's derivative is a linear chain backwards over the same steps,
with weights W[r, c] = exp(y[t - 1][r] + M[t][r, c] - y[t][c]) in [0, 1]
that the kernel builds on the fly, and a linear chain's derivative is again
one. So :class:`_LogChain` and :class:`_LinearChain` are differentiable to
any order in reverse mode; they have no forward-mode rules. The gradients
of shared transition matrices sum the weighted terms over the batch and
time with :func:`._contract.weighted_contract`.
"""

import torch
from torch import Tensor

from .._triton import check_cuda_triton


def _parse(
    name: str, log_emit: Tensor, log_trans: Tensor, log_init: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate the model; return log_emit, log_trans as (1 or B, 1 or N - 1, K, K), and
    log_init as (B, K)."""
    if log_emit.dim() != 3:
        raise ValueError(f"log_emit must be (B, N, K), got {tuple(log_emit.shape)}")
    batch_size, N, K = log_emit.shape
    transitions = max(N - 1, 0)
    match log_trans.dim() if log_trans.shape[-2:] == (K, K) else None:
        case 2:
            log_trans = log_trans[None, None]
        # Per step before per signal when both fit, as in kalman_filter.
        case 3 if log_trans.size(0) == transitions:
            log_trans = log_trans[None]
        case 3 if log_trans.size(0) == batch_size:
            log_trans = log_trans[:, None]
        case 4 if log_trans.shape[:2] == (batch_size, transitions):
            pass
        case _:
            raise ValueError(
                f"log_trans must be of shape {(K, K)}, {(transitions, K, K)}, "
                f"{(batch_size, K, K)} or {(batch_size, transitions, K, K)}, "
                f"got {tuple(log_trans.shape)}"
            )
    if log_init.shape not in ((K,), (batch_size, K)):
        raise ValueError(
            f"log_init must be of shape {(K,)} or {(batch_size, K)}, got {tuple(log_init.shape)}"
        )
    check_cuda_triton(name, log_emit, log_trans, log_init)
    # The kernels read each K x K matrix as a contiguous block, where batch
    # and time may broadcast, and the emissions as one contiguous block.
    if log_trans.stride(-1) != 1 or log_trans.stride(-2) != K:
        log_trans = log_trans.contiguous()
    log_emit = log_emit.contiguous()
    return log_emit, log_trans, log_init.expand(batch_size, K)


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


# The transition gradient's sum over the batch and time runs over blocks of
# this many terms in parallel, then over the blocks.
_REDUCE_BLOCK = 256


def _trans_grad(log_trans: Tensor, alpha: Tensor, beta: Tensor, fi: Tensor, fj: Tensor) -> Tensor:
    """The sum of exp(log_trans[i, j] + alpha[i] + beta[j]) fi[i] fj[j] over shared steps.

    alpha, beta and the factors fi and fj are (B, T, K). The result has
    log_trans's shape, (1 or B, 1 or T, K, K), summed over the batch and time
    that it is shared over, or (2, ...) for a stacked pair of chains, each
    half of the batch summed into its own matrices.
    """
    from ._contract import weighted_contract

    if log_trans.dim() == 5:
        halves = zip(log_trans, *(t.chunk(2) for t in (alpha, beta, fi, fj)), strict=True)
        return torch.stack([_trans_grad(*half) for half in halves])
    B, T, K = alpha.shape
    groups = log_trans.shape[:2]
    if alpha.numel() == 0:
        return torch.zeros_like(log_trans)
    if groups == (B, T):
        weights = torch.exp(log_trans + alpha.unsqueeze(-1) + beta.unsqueeze(-2))
        return weights * fi.unsqueeze(-1) * fj.unsqueeze(-2)
    # (P groups, M terms, K): time-varying matrices shared by the batch sum
    # over it, the matrices of each sequence over its time.
    terms = (alpha, beta, fi, fj)
    if groups[1] == T:
        terms = tuple(t.transpose(0, 1) for t in terms)
    elif groups[0] != B:
        terms = tuple(t.reshape(1, B * T, K) for t in terms)
    P, M = terms[0].shape[:2]
    size = min(M, _REDUCE_BLOCK)
    blocks = -(-M // size)
    # Pad the terms to whole blocks with zero weights: -inf exponents, zero factors.
    alpha, beta, fi, fj = (
        torch.nn.functional.pad(t, (0, 0, 0, blocks * size - M), value=value).reshape(
            P * blocks, size, K
        )
        for t, value in zip(terms, (float("-inf"), float("-inf"), 0.0, 0.0))
    )
    # W[i, m, j] = exp(alpha[m, i] + beta[m, j] + log_trans[i, j]).
    c = log_trans.reshape(P, 1, K, K).expand(P, blocks, K, K).reshape(P * blocks, K, K)
    ones = torch.ones_like(c)
    out = weighted_contract(alpha.mT, beta, c, fi.mT, ones, fj, "j")
    return out.reshape(P, blocks, K, K).sum(1).reshape(*groups, K, K)


def _positions(adjoint: bool) -> tuple[int, int, slice, slice]:
    """In a chain's (B, T + 1, K) messages: the start, the message no step reads,
    and the steps' inputs and outputs."""
    if adjoint:
        return -1, 0, slice(1, None), slice(0, -1)
    return 0, -1, slice(0, -1), slice(1, None)


def _run_chain(
    semiring, y0, log_trans, log_emit, inj=None, adjoint=False, weights=None, argmax=None
):
    """The chain's messages with y0 included: (B, T + 1, K), y0 first, or last if adjoint.

    The semiring is "log", "max" or "linear"; see :func:`._hmm_kernels.chain`.
    """
    # Imported here so that philtorch.estimation imports without Triton.
    from ._hmm_kernels import chain

    B, T, K = log_emit.shape
    out = y0.new_empty(B, T + 1, K)
    start, _, _, steps = _positions(adjoint)
    out[:, start] = y0
    if log_trans.dim() == 5:
        trans = log_trans.expand(2, B // 2, T, K, K)
    else:
        trans = log_trans.expand(B, T, K, K)
    chain(y0, trans, log_emit, inj, semiring, adjoint, out[:, steps], weights, argmax)
    return out


class _LogChain(torch.autograd.Function):
    """The log chain y[t] = y[t - 1] (x) M[t] of :func:`_run_chain`, differentiable.

    M[t][i, j] is log_trans[t][i, j] + log_emit[t][i]. log_trans is
    (1 or B, 1 or T, K, K), or (2, ...) for two chains stacked in the batch,
    each half with its own matrices.
    """

    @staticmethod
    def forward(y0, log_trans, log_emit):
        return _run_chain("log", y0, log_trans, log_emit)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, log_trans, log_emit = inputs
        ctx.save_for_backward(log_trans, log_emit, output)

    @staticmethod
    def backward(ctx, grad):
        log_trans, log_emit, y = ctx.saved_tensors
        start, last, prev, steps = _positions(False)
        # The chain kernels take time slices of contiguous messages.
        grad = grad.contiguous()
        # A step's weights W[r, c] = exp(y_in[r] + M[r, c] - y_out[c]); an
        # unreachable output, -inf, takes none.
        y_in = y[:, prev]
        neg_out = torch.where(torch.isfinite(y), -y, float("-inf"))[:, steps]
        # The adjoints a = grad + W a_out: the adjoint chain over the same
        # steps, from the message no step reads.
        a = _LinearChain.apply(
            grad[:, last], log_trans, log_emit, grad[:, prev], neg_out, y_in, True
        )
        a_out = a[:, steps]
        # The emissions, on W's rows, take the inputs' adjoint less their own
        # gradient.
        grad_emit = a[:, prev] - grad[:, prev]
        grad_trans = None
        if ctx.needs_input_grad[1]:
            ones = torch.ones_like(a_out)
            grad_trans = _trans_grad(log_trans, y_in + log_emit, neg_out, ones, a_out)
        return a[:, start], grad_trans, grad_emit


class _LinearChain(torch.autograd.Function):
    """The linear chain x[t] = x[t - 1] A[t] + j[t] of :func:`_run_chain`, differentiable.

    A[t][r, c] = exp(M[t][r, c] + p[t][r] + q[t][c]), with M[t] as in
    :class:`_LogChain` and log weights p and q of shape (B, T, K). With
    ``adjoint``, the chain runs from the last step back over each M[t]'s
    transpose; the emissions stay on log_trans's rows i, now A's columns.
    """

    @staticmethod
    def forward(x0, log_trans, log_emit, inj, p, q, adjoint):
        return _run_chain("linear", x0, log_trans, log_emit, inj, adjoint, (p, q))

    @staticmethod
    def setup_context(ctx, inputs, output):
        *_, adjoint = inputs
        ctx.adjoint = adjoint
        ctx.save_for_backward(*inputs[1:6], output)

    @staticmethod
    def backward(ctx, grad):
        log_trans, log_emit, inj, p, q, x = ctx.saved_tensors
        adjoint = ctx.adjoint
        start, last, prev, steps = _positions(adjoint)
        grad = grad.contiguous()
        # The adjoints h = grad + A h_out: the adjoint chain the other way,
        # whose weights swap p and q.
        h = _LinearChain.apply(grad[:, last], log_trans, log_emit, grad[:, prev], q, p, not adjoint)
        x_in, h_out = x[:, prev], h[:, steps]
        # Each step's A[r, c] gets x_in[r] h_out[c], and so p[r] and q[c] its
        # row and column sums, which the chains already hold.
        grad_p = x_in * (h[:, prev] - grad[:, prev])
        grad_q = h_out * (x[:, steps] - inj)
        # The emissions are on the stored matrices' rows: A's rows, or with
        # ``adjoint`` its columns.
        grad_emit = grad_q if adjoint else grad_p
        grad_trans = None
        if ctx.needs_input_grad[1]:
            if adjoint:
                grad_trans = _trans_grad(log_trans, q + log_emit, p, h_out, x_in)
            else:
                grad_trans = _trans_grad(log_trans, p + log_emit, q, x_in, h_out)
        grad_inj = h_out if ctx.needs_input_grad[3] else None
        return h[:, start], grad_trans, grad_emit, grad_inj, grad_p, grad_q, None


class _Viterbi(torch.autograd.Function):
    """The Viterbi score and path; the score's gradient is the path's indicator."""

    @staticmethod
    def forward(log_emit, log_trans, log_init):
        from ._hmm_kernels import trace

        B, N, K = log_emit.shape
        # Each state's best predecessor per step, as time slices of (B, N, K):
        # the kernels need the messages' batch stride.
        pointers = log_emit.new_empty(B, N, K, dtype=torch.int32)[:, 1:]
        # The best score of a path to each state, without its emission.
        predicted = _run_chain("max", log_init, log_trans, log_emit[:, :-1], argmax=pointers)
        score, last = (predicted[:, -1] + log_emit[:, -1]).max(dim=-1)
        # Follow the pointers back from the best last state.
        path = trace(last.int(), pointers, reverse=True)
        return score, torch.cat([path, last.int().unsqueeze(1)], dim=1).long()

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, path = output
        ctx.trans_shape = inputs[1].shape
        ctx.mark_non_differentiable(path)
        ctx.save_for_backward(path)

    @staticmethod
    def backward(ctx, grad, _):
        (path,) = ctx.saved_tensors
        B, N = path.shape
        groups, K = ctx.trans_shape[:2], ctx.trans_shape[-1]
        grad_emit = grad[:, None, None] * torch.nn.functional.one_hot(path, K).to(grad.dtype)
        # Count each transition of the path; the modulo maps a shared batch
        # or time dimension to its single index.
        b = torch.arange(B, device=path.device)[:, None] % groups[0]
        t = torch.arange(N - 1, device=path.device)[None, :] % max(groups[1], 1)
        grad_trans = grad.new_zeros(ctx.trans_shape).index_put(
            (b, t, path[:, :-1], path[:, 1:]), grad[:, None].expand(B, N - 1), accumulate=True
        )
        return grad_emit, grad_trans, grad_emit[:, 0]


def _empty(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    """The results for no steps: a log-likelihood of 0, (B,), and no distributions.

    Built from empty slices of every input, rather than as new tensors, so
    they stay in the autograd graph and give zero gradients; sums over the
    empty slices avoid 0 * -inf.
    """
    zero = log_emit.sum((1, 2)) + log_init[:, :0].sum(-1) + log_trans[..., :0].sum()
    return zero, log_emit + zero[:, None, None]


def _predicted_and_backward(
    log_emit: Tensor, log_trans: Tensor, log_init: Tensor
) -> tuple[Tensor, Tensor]:
    """The predicted messages log p(y[0..n - 1], z[n]) and the backward messages
    beta[n][i] = log p(y[n + 1..N - 1] | z[n] = i), both (B, N, K).

    Two log chains stacked in the batch of one call: beta's runs in flipped
    time over the transposed matrices. The joint is then predicted +
    log_emit + beta, with no subtraction that could give -inf - (-inf) at
    impossible emissions.
    """
    B = log_emit.size(0)
    trans = torch.stack([log_trans, log_trans.flip(1).mT])
    # Time slices of a contiguous (2B, N, K), as the kernels take steps.
    emit = torch.cat([log_emit, log_emit.flip(1)])[:, :-1]
    y0 = torch.cat([log_init, torch.zeros_like(log_init)])
    predicted, beta = _LogChain.apply(y0, trans, emit).split(B)
    return predicted, beta.flip(1)


def hmm_filter(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    r"""Filter a hidden Markov model: its log-likelihood and filtered state probabilities.

    For a hidden Markov model with states :math:`z[0], \dots, z[N - 1]` and
    measurements :math:`y[0], \dots, y[N - 1]`, where :math:`y[n]` is emitted
    from :math:`z[n]` and :math:`N - 1` transitions move :math:`z[n]` to
    :math:`z[n + 1]`, this returns :math:`\log p(y[0], \dots, y[N - 1])` and
    :math:`\log p(z[n] \mid y[0], \dots, y[n])`: the forward algorithm,
    computed as a scan over time instead of step by step, as in `Temporal
    Parallelization of Inference in Hidden Markov Models`_ (Hassan, Särkkä
    and García-Fernández, 2021).

    The results are differentiable to any order in reverse mode, with
    respect to every input, so the log-likelihood can train the model's
    probabilities, or a network that predicts them; its gradient with respect
    to :attr:`log_emit` is the posteriors of :func:`hmm_smoother`. Forward-mode
    automatic differentiation is not supported. The scan runs in chunks of
    steps, a few Triton kernel launches whatever N is, with O(N K^3) work,
    and its gradients are scans of the same kind.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        log_emit (Tensor): :math:`\log p(y[n] \mid z[n] = k)`, of shape
            :math:`(B, N, K)`.
        log_trans (Tensor): :math:`\log p(z[n + 1] = j \mid z[n] = i)` at
            index :math:`[i, j]`: shared, of shape :math:`(K, K)`; per step,
            :math:`(N - 1, K, K)`; per signal, :math:`(B, K, K)`; or both,
            :math:`(B, N - 1, K, K)`. When :math:`B = N - 1`, a 3-D tensor is
            taken per step.
        log_init (Tensor): :math:`\log p(z[0] = k)`, of shape :math:`(K)` or
            :math:`(B, K)`.

    Returns:
        tuple of Tensor: the log-likelihood, of shape :math:`(B)`, and the
        filtered log-probabilities, of shape :math:`(B, N, K)`.

    Raises:
        ValueError: if the inputs are not CUDA tensors or have unsupported
            shapes.
        RuntimeError: if Triton is not installed.

    .. _Temporal Parallelization of Inference in Hidden Markov Models:
        https://doi.org/10.1109/TSP.2021.3103338
    """
    log_emit, log_trans, log_init = _parse("hmm_filter", log_emit, log_trans, log_init)
    if log_emit.size(1) == 0:
        return _empty(log_emit, log_trans, log_init)
    # alpha[n][j] = log p(y[0..n], z[n] = j), from the predicted messages.
    alpha = _LogChain.apply(log_init, log_trans, log_emit[:, :-1]) + log_emit
    norm = _logsumexp(alpha, dim=-1, keepdim=True)
    return norm[:, -1, 0], alpha - norm


def hmm_smoother(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    r"""Smooth a hidden Markov model: forward-backward state posteriors.

    For the model of :func:`hmm_filter`, this returns the log-likelihood and
    :math:`\log p(z[n] \mid y[0], \dots, y[N - 1])`, from a forward and an
    independent backward scan, as in `Temporal Parallelization of Inference
    in Hidden Markov Models`_ (Hassan et al., 2021). Its implementation and
    differentiability are those of :func:`hmm_filter`.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        log_emit (Tensor): :math:`\log p(y[n] \mid z[n] = k)`, of shape
            :math:`(B, N, K)`.
        log_trans (Tensor): :math:`\log p(z[n + 1] = j \mid z[n] = i)` at
            index :math:`[i, j]`: shared, of shape :math:`(K, K)`; per step,
            :math:`(N - 1, K, K)`; per signal, :math:`(B, K, K)`; or both,
            :math:`(B, N - 1, K, K)`. When :math:`B = N - 1`, a 3-D tensor is
            taken per step.
        log_init (Tensor): :math:`\log p(z[0] = k)`, of shape :math:`(K)` or
            :math:`(B, K)`.

    Returns:
        tuple of Tensor: the log-likelihood, of shape :math:`(B)`, and the
        posterior log-probabilities, of shape :math:`(B, N, K)`.

    Raises:
        ValueError: if the inputs are not CUDA tensors or have unsupported
            shapes.
        RuntimeError: if Triton is not installed.

    .. _Temporal Parallelization of Inference in Hidden Markov Models:
        https://doi.org/10.1109/TSP.2021.3103338
    """
    log_emit, log_trans, log_init = _parse("hmm_smoother", log_emit, log_trans, log_init)
    if log_emit.size(1) == 0:
        return _empty(log_emit, log_trans, log_init)
    predicted, beta = _predicted_and_backward(log_emit, log_trans, log_init)
    log_likelihood = _logsumexp(predicted[:, -1] + log_emit[:, -1], dim=-1)
    # Normalize each step by its own sum rather than by the likelihood: the
    # messages grow to thousands over long inputs, and most of their rounding
    # error is shared by all states at a step, so this cancels it.
    joint = predicted + log_emit + beta
    return log_likelihood, joint - _logsumexp(joint, dim=-1, keepdim=True)


def hmm_viterbi(log_emit: Tensor, log_trans: Tensor, log_init: Tensor) -> tuple[Tensor, Tensor]:
    r"""Decode a hidden Markov model: its most probable state sequence.

    For the model of :func:`hmm_filter`, this is the Viterbi algorithm with
    both of its passes parallelized over time: a max-plus scan, as in
    `Temporal Parallelization of Inference in Hidden Markov Models`_ (Hassan
    et al., 2021), gives the best score of any path ending in each state and
    records each state's best predecessor, and a traceback follows those
    backpointers from the best last state, in chunks of steps. So the path is
    always one of the optimal paths: under ties, the one that prefers lower
    state indices, as the sequential algorithm's backpointers do. The score's
    gradient is that of the decoded path: one for its first state's prior
    and for each of its emissions and transitions; its higher derivatives are
    zero.

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.

    Args:
        log_emit (Tensor): :math:`\log p(y[n] \mid z[n] = k)`, of shape
            :math:`(B, N, K)`.
        log_trans (Tensor): :math:`\log p(z[n + 1] = j \mid z[n] = i)` at
            index :math:`[i, j]`: shared, of shape :math:`(K, K)`; per step,
            :math:`(N - 1, K, K)`; per signal, :math:`(B, K, K)`; or both,
            :math:`(B, N - 1, K, K)`. When :math:`B = N - 1`, a 3-D tensor is
            taken per step.
        log_init (Tensor): :math:`\log p(z[0] = k)`, of shape :math:`(K)` or
            :math:`(B, K)`.

    Returns:
        tuple of Tensor: the best joint log-probability
        :math:`\max \log p(z[0], \dots, z[N - 1], y[0], \dots, y[N - 1])`,
        of shape :math:`(B)`, and the states :math:`z[0], \dots, z[N - 1]`
        of that path, of shape :math:`(B, N)`.

    Raises:
        ValueError: if the inputs are not CUDA tensors or have unsupported
            shapes.
        RuntimeError: if Triton is not installed.

    .. _Temporal Parallelization of Inference in Hidden Markov Models:
        https://doi.org/10.1109/TSP.2021.3103338
    """
    log_emit, log_trans, log_init = _parse("hmm_viterbi", log_emit, log_trans, log_init)
    batch_size, N, _ = log_emit.shape
    if N == 0:
        score, _ = _empty(log_emit, log_trans, log_init)
        return score, log_emit.new_zeros(batch_size, 0, dtype=torch.long)
    return _Viterbi.apply(log_emit, log_trans, log_init)
