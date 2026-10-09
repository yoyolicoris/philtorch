"""Chunked scans of HMM messages for small state spaces, in Triton.

A message chain x[t] = x[t - 1] (x) M[t], in the log semiring (logsumexp of
sums) or the max-plus one, with x a vector of K states and M[t] K x K, is
computed in two levels instead of O(log N) rounds of matrix products:

1. ``_chunk_totals_kernel`` multiplies each chunk of T consecutive matrices,
   one program per chunk, with the running product in registers;
2. the chain over the chunk totals gives each chunk's starting vector, by
   the same method one level up;
3. ``_chunk_sweep_kernel`` carries each chunk's starting vector through its
   matrices, one program per chunk.

So a chain costs about N K^3 + N K^2 work, a few launches, and a sequential
depth of a few T. Each matrix is M[t][i, j] = log_trans[n][i, j] +
log_emit[n][j] at time n, built in the kernels from ``log_trans``, which may
be shared over the batch or time through zero strides, so the B x N x K x K
step matrices are never stored. With ``REVERSE``, the chain runs from the
last time back, and with ``TRANSPOSE`` it multiplies by M[t]'s transpose:
together, the backward messages. The kernels hold K x K x K products in
registers, so they serve K up to 32; they have no autograd.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

_NEG_INF = float("-inf")
# Triton kernels can only read module constants made with tl.constexpr.
_KERNEL_NEG_INF = tl.constexpr(_NEG_INF)
# The chunk length: each kernel program takes T sequential steps.
_CHUNK = 64
# The largest K the kernels serve.
MAX_STATES = 32


@triton.jit
def _reduce(x, axis: tl.constexpr, IS_MAX: tl.constexpr):
    """Max or logsumexp along ``axis``, -inf where every term is -inf."""
    top = tl.max(x, axis)
    if IS_MAX:
        return top
    shift = tl.where(top == _KERNEL_NEG_INF, 0.0, top)
    return shift + tl.log(tl.sum(tl.exp(x - tl.expand_dims(shift, axis)), axis))


@triton.jit
def _step_matrix(
    trans_ptr, emit_ptr, b, n, valid, K, stride_tb, stride_tn, stride_eb, stride_en, rows, cols,
    HAS_EMIT: tl.constexpr, TRANSPOSE: tl.constexpr,
):  # fmt: skip
    """M[n] of batch b as a (BK, BK) tile, or its transpose; -inf outside K x K."""
    mask = (rows < K) & (cols < K) & valid
    # (i, j) of the stored matrix for tile entry (rows, cols).
    i, j = (cols, rows) if TRANSPOSE else (rows, cols)
    m = tl.load(
        trans_ptr + b * stride_tb + n * stride_tn + i * K + j, mask=mask, other=_KERNEL_NEG_INF
    )
    if HAS_EMIT:
        m += tl.load(emit_ptr + b * stride_eb + n * stride_en + j, mask=mask, other=0.0)
    return m


@triton.jit
def _chunk_totals_kernel(
    trans_ptr, emit_ptr, out_ptr, N, K, C, stride_tb, stride_tn, stride_eb, stride_en,
    T: tl.constexpr, HAS_EMIT: tl.constexpr, REVERSE: tl.constexpr, TRANSPOSE: tl.constexpr,
    IS_MAX: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    """out[b, c] = M[cT] (x) ... (x) M[cT + T - 1], in chain order; one program per chunk."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = pid % C
    rows = tl.arange(0, BK)[:, None]
    cols = tl.arange(0, BK)[None, :]
    t = c * T
    n = (N - 1 - t) if REVERSE else t
    total = _step_matrix(
        trans_ptr, emit_ptr, b, n, True, K, stride_tb, stride_tn, stride_eb, stride_en,
        rows, cols, HAS_EMIT, TRANSPOSE,
    )  # fmt: skip
    for s in range(1, T):
        t = c * T + s
        valid = t < N
        n = (N - 1 - t) if REVERSE else t
        m = _step_matrix(
            trans_ptr, emit_ptr, b, n, valid, K, stride_tb, stride_tn, stride_eb, stride_en,
            rows, cols, HAS_EMIT, TRANSPOSE,
        )  # fmt: skip
        # (total (x) m)[i, k] = reduce over j of total[i, j] + m[j, k].
        product = _reduce(total[:, :, None] + m[None, :, :], 1, IS_MAX)
        total = tl.where(valid, product, total)
    tl.store(
        out_ptr + pid.to(tl.int64) * K * K + rows * K + cols, total, mask=(rows < K) & (cols < K)
    )


@triton.jit
def _chunk_sweep_kernel(
    start_ptr, trans_ptr, emit_ptr, out_ptr, N, K, C, stride_tb, stride_tn, stride_eb, stride_en,
    T: tl.constexpr, HAS_EMIT: tl.constexpr, REVERSE: tl.constexpr, TRANSPOSE: tl.constexpr,
    IS_MAX: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    """x[t] = x[t - 1] (x) M[t] through chunk c from its start; out at time n of each step."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = pid % C
    states = tl.arange(0, BK)
    rows = states[:, None]
    cols = states[None, :]
    x = tl.load(start_ptr + pid.to(tl.int64) * K + states, mask=states < K, other=_KERNEL_NEG_INF)
    for s in range(0, T):
        t = c * T + s
        valid = t < N
        n = (N - 1 - t) if REVERSE else t
        m = _step_matrix(
            trans_ptr, emit_ptr, b, n, valid, K, stride_tb, stride_tn, stride_eb, stride_en,
            rows, cols, HAS_EMIT, TRANSPOSE,
        )  # fmt: skip
        # (x (x) m)[k] = reduce over i of x[i] + m[i, k].
        x_next = _reduce(x[:, None] + m, 0, IS_MAX)
        tl.store(out_ptr + (b * N + n) * K + states, x_next, mask=(states < K) & valid)
        x = tl.where(valid, x_next, x)


def _num_warps(K: int) -> int:
    """Few warps: more spill the K x K x K products (measured on an RTX 5060 Ti)."""
    return 1 if K <= 4 else 2 if K <= 16 else 4


def _strides(t: Tensor) -> tuple[int, int]:
    """The batch and time strides of a (B, N, ...) view; 0 where broadcast."""
    return t.stride(0), t.stride(1)


def chain(
    x0: Tensor,
    log_trans: Tensor,
    log_emit: Tensor | None,
    is_max: bool,
    reverse: bool = False,
    transpose: bool = False,
) -> Tensor:
    """The messages x[t] = x[t - 1] (x) M[t], t = 0, ..., N - 1, from x[-1] = x0.

    Args:
        x0: the starting vectors, (B, K).
        log_trans: (B, N, K, K), possibly with zero batch or time strides,
            and contiguous K x K matrices.
        log_emit: (B, N, K) with a contiguous last dimension, added to each
            matrix's columns, or None.
        is_max: the max-plus semiring rather than the log one.
        reverse: run the chain from time N - 1 down to 0; the message after
            the step at time n is still written at n.
        transpose: multiply by each matrix's transpose.

    Returns:
        The messages, (B, N, K), each written at its step's time.
    """
    B, N, K = log_trans.shape[0], log_trans.shape[1], log_trans.shape[-1]
    out = x0.new_empty(B, N, K)
    if B == 0 or N == 0 or K == 0:
        return out
    T = _CHUNK
    C = triton.cdiv(N, T)
    has_emit = log_emit is not None
    emit = log_emit if has_emit else x0
    emit_strides = _strides(log_emit) if has_emit else (0, 0)
    flags = dict(T=T, HAS_EMIT=has_emit, REVERSE=reverse, TRANSPOSE=transpose, IS_MAX=is_max)
    flags |= dict(BK=max(triton.next_power_of_2(K), 2), num_warps=_num_warps(K))
    if C == 1:
        starts = x0.contiguous()
    else:
        totals = x0.new_empty(B, C, K, K)
        _chunk_totals_kernel[(B * C,)](
            log_trans, emit, totals, N, K, C, *_strides(log_trans), *emit_strides, **flags
        )
        # The totals are in chain order and already transposed: their own
        # chain runs forward, with no emissions.
        ends = chain(x0, totals, None, is_max)
        starts = torch.cat([x0.unsqueeze(1), ends[:, :-1]], dim=1).contiguous()
    _chunk_sweep_kernel[(B * C,)](
        starts, log_trans, emit, out, N, K, C, *_strides(log_trans), *emit_strides, **flags
    )
    return out
