"""Chunked parallel scans of HMM messages, in Triton.

A message chain

    y[t] = (y[t - 1] (x) A[t]) + j[t],   t = 0, ..., N - 1,

with y a vector of K states, A[t] K x K and, for a linear chain, an optional
injection j[t], is computed as a parallel scan in two levels instead of N
sequential steps:

1. ``_chunk_totals_kernel`` composes each chunk of T consecutive steps into
   one map y_end = (y_start (x) P) + o, one program per chunk;
2. the chain of those maps, whose injections are the o, gives each chunk's
   starting vector, by the same method one level up;
3. ``_chunk_sweep_kernel`` carries each chunk's starting vector through its
   steps, one program per chunk.

So a chain costs about N K^3 work for the totals and N K^2 for the sweep, a
few launches, and a sequential depth of a few T. The semiring is one of

* ``"log"``: (x) is the vector-matrix product with logsumexp in place of
  sums, and A[t] = M[t], log-probabilities;
* ``"max"``: the same with max in place of logsumexp;
* ``"linear"``: the ordinary product, with A[t] = exp(M[t][r, c] + p[t][r] +
  q[t][c]) for log weights p and q, or explicit matrices.

M[t] at time n is log_trans[n][i, j] plus log_emit[n][i], the emission of
the state a transition leaves, built in the kernels from ``log_trans``,
which may be shared over the batch or time through zero strides, so the
B x N x K x K step matrices are never stored. With ``ADJOINT``, the chain
runs from the last time back over each M[t]'s transpose: a chain's
derivative, a linear chain backwards, runs here too.

A max-plus chain can also record each message's maximizing previous state,
and :func:`trace` follows such backpointers in the same two levels: Viterbi
decoding's traceback, in parallel.

Up to ``_REGISTER_STATES`` states, or twice that for a linear chain, a
program holds the K x K x K terms of a product in registers. Above, it
computes them in tiles of rows and columns, and the totals' running product
alternates between two K x K scratch buffers, so registers and shared memory
stay bounded for any K.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

_SEMIRINGS = {"log": 0, "max": 1, "linear": 2}
# Triton kernels can only read module constants made with tl.constexpr.
_MAX, _LINEAR = (tl.constexpr(_SEMIRINGS[name]) for name in ("max", "linear"))
_KERNEL_NEG_INF = tl.constexpr(float("-inf"))
# The chunk length: each kernel program takes T sequential steps.
_CHUNK = 64
# The largest K whose products a program holds in registers; twice that for a
# linear chain (see _config).
_REGISTER_STATES = 16


@triton.jit
def _reduce(x, axis: tl.constexpr, SEMIRING: tl.constexpr):
    """The semiring's sum along ``axis``; logsumexp is -inf where every term is."""
    if SEMIRING == _LINEAR:
        return tl.sum(x, axis)
    top = tl.max(x, axis)
    if SEMIRING == _MAX:
        return top
    shift = tl.where(top == _KERNEL_NEG_INF, 0.0, top)
    return shift + tl.log(tl.sum(tl.exp(x - tl.expand_dims(shift, axis)), axis))


@triton.jit
def _times(a, b, SEMIRING: tl.constexpr):
    if SEMIRING == _LINEAR:
        return a * b
    return a + b


@triton.jit
def _time(c, s, N, T: tl.constexpr, REVERSE: tl.constexpr):
    """The time of step s of chunk c, a 64-bit index."""
    t = c * T + s
    return (N - 1 - t) if REVERSE else t


@triton.jit
def _step_matrix(
    trans, emit, p, q, n, K, stride_tn, rows, cols, ADJOINT: tl.constexpr,
    SEMIRING: tl.constexpr,
):  # fmt: skip
    """Entries (rows, cols) of A at time n, from pointers offset to the batch item.

    ``emit``, ``p`` and ``q`` may be None; each step of them is K contiguous
    values. The semiring's zero outside K x K.
    """
    mask = (rows < K) & (cols < K)
    # (i, j) of the stored matrix for entry (rows, cols).
    i, j = (cols, rows) if ADJOINT else (rows, cols)
    offsets = trans + n * stride_tn + i * K + j
    if SEMIRING == _LINEAR and p is None:
        return tl.load(offsets, mask=mask, other=0.0)
    m = tl.load(offsets, mask=mask, other=_KERNEL_NEG_INF)
    # The emissions of the states i leaves. On an adjoint chain's columns they
    # join q; on rows they take the tile's shape, as a broadcast row vector
    # would cost a layout conversion.
    if emit is not None and not ADJOINT:
        m += tl.load(emit + n * K + i + 0 * j, mask=mask, other=0.0)
    if p is not None:
        p_rows = tl.load(p + n * K + rows, mask=rows < K, other=_KERNEL_NEG_INF)
        q_cols = tl.load(q + n * K + cols, mask=cols < K, other=_KERNEL_NEG_INF)
        if emit is not None and ADJOINT:
            q_cols += tl.load(emit + n * K + cols, mask=cols < K, other=0.0)
        m = tl.exp(m + p_rows + q_cols)
    return m


@triton.jit
def _chunk_totals_kernel(
    trans_ptr, emit_ptr, p_ptr, q_ptr, inj_ptr, total_ptr, offset_ptr, scratch_ptr, N, K, C,
    HB, stride_th, stride_tb, stride_tn, stride_b, T: tl.constexpr, ADJOINT: tl.constexpr,
    SEMIRING: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr, BR: tl.constexpr,
):  # fmt: skip
    """Chunk c's map: total = A[cT] (x) ... (x) A[cT + T - 1], in chain order, and
    for a linear chain with injections, offset, the chain through those steps
    of the injections alone."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, N - c * T)
    # A stacked pair of chains reads its own matrices per half of the batch.
    trans = trans_ptr + (b // HB) * stride_th + (b % HB) * stride_tb
    emit = emit_ptr + b * stride_b if emit_ptr is not None else None
    p = p_ptr + b * stride_b if p_ptr is not None else None
    q = q_ptr + b * stride_b if q_ptr is not None else None
    inj = inj_ptr + b * stride_b if inj_ptr is not None else None
    states = tl.arange(0, BK)
    rows = states[:, None]
    cols = states[None, :]
    n = _time(c, 0, N, T, ADJOINT)
    if inj is not None:
        offset = tl.load(inj + n * K + states, mask=states < K, other=0.0)
    out = total_ptr + pid.to(tl.int64) * K * K
    if BC == BK:
        total = _step_matrix(trans, emit, p, q, n, K, stride_tn, rows, cols, ADJOINT, SEMIRING)
        # A loop of constant length, with steps past the end masked, unrolls
        # and pipelines better for these small matrices.
        for s in range(1, T):
            valid = s < steps
            n = _time(c, tl.where(valid, s, 0), N, T, ADJOINT)
            m = _step_matrix(trans, emit, p, q, n, K, stride_tn, rows, cols, ADJOINT, SEMIRING)
            # (total (x) m)[i, k] = sum over j of total[i, j] (x) m[j, k].
            product = _reduce(_times(total[:, :, None], m[None, :, :], SEMIRING), 1, SEMIRING)
            total = tl.where(valid, product, total)
            if inj is not None:
                j = tl.load(inj + n * K + states, mask=states < K, other=0.0)
                offset = tl.where(valid, tl.sum(offset[:, None] * m, 0) + j, offset)
        tl.store(out + rows * K + cols, total, mask=(rows < K) & (cols < K))
    else:
        # Tiles of BR rows and BC columns: the running product alternates
        # between two K x K scratch buffers, and only tiles are in registers.
        scratch = scratch_ptr + pid.to(tl.int64) * (2 * BK * BK + BK)
        tile_rows = tl.arange(0, BR)[:, None]
        block = tl.arange(0, BC)
        for k0 in range(0, BK, BC):
            columns = k0 + block[None, :]
            m = _step_matrix(trans, emit, p, q, n, K, stride_tn, rows, columns, ADJOINT, SEMIRING)
            tl.store(scratch + rows * BK + columns, m)
        tl.debug_barrier()
        for s in range(1, steps):
            n = _time(c, s, N, T, ADJOINT)
            current = scratch + ((s - 1) % 2) * BK * BK
            following = scratch + (s % 2) * BK * BK
            for k0 in range(0, BK, BC):
                columns = k0 + block
                m = _step_matrix(
                    trans, emit, p, q, n, K, stride_tn, rows, columns[None, :], ADJOINT,
                    SEMIRING,
                )  # fmt: skip
                for i0 in range(0, BK, BR):
                    a = tl.load(current + (i0 + tile_rows) * BK + cols)
                    product = _reduce(_times(a[:, :, None], m[None, :, :], SEMIRING), 1, SEMIRING)
                    tl.store(following + (i0 + tile_rows) * BK + columns[None, :], product)
                if inj is not None:
                    j = tl.load(inj + n * K + columns, mask=columns < K, other=0.0)
                    o = tl.sum(offset[:, None] * m, 0)
                    tl.store(scratch + 2 * BK * BK + columns, o + j)
            tl.debug_barrier()
            if inj is not None:
                offset = tl.load(scratch + 2 * BK * BK + states)
                # Every thread reads the offset before the next step rewrites it.
                tl.debug_barrier()
        last = scratch + ((steps - 1) % 2) * BK * BK
        for k0 in range(0, BK, BC):
            columns = k0 + block[None, :]
            tile = tl.load(last + rows * BK + columns)
            tl.store(out + rows * K + columns, tile, mask=(rows < K) & (columns < K))
    if inj is not None:
        tl.store(offset_ptr + pid.to(tl.int64) * K + states, offset, mask=states < K)


@triton.jit
def _chunk_sweep_kernel(
    start_ptr, trans_ptr, emit_ptr, p_ptr, q_ptr, inj_ptr, out_ptr, argmax_ptr, N, K, C,
    HB, stride_th, stride_tb, stride_tn, stride_b, T: tl.constexpr, ADJOINT: tl.constexpr,
    SEMIRING: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
):  # fmt: skip
    """y[t] = (y[t - 1] (x) A[t]) + j[t] through chunk c from its start, written at time n.

    With ``argmax_ptr``, a max-plus chain also writes, for each state k, the
    lowest-index state i attaining y[t][k] = y[t - 1][i] + A[t][i, k].
    """
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, N - c * T)
    # A stacked pair of chains reads its own matrices per half of the batch.
    trans = trans_ptr + (b // HB) * stride_th + (b % HB) * stride_tb
    emit = emit_ptr + b * stride_b if emit_ptr is not None else None
    p = p_ptr + b * stride_b if p_ptr is not None else None
    q = q_ptr + b * stride_b if q_ptr is not None else None
    inj = inj_ptr + b * stride_b if inj_ptr is not None else None
    argmax = argmax_ptr + b * stride_b if argmax_ptr is not None else None
    out = out_ptr + b * stride_b
    states = tl.arange(0, BK)
    rows = states[:, None]
    # Padded states read 0, which a padded matrix row turns into the zero.
    y = tl.load(start_ptr + pid.to(tl.int64) * K + states, mask=states < K, other=0.0)
    if BC == BK:
        for s in range(0, T):
            valid = s < steps
            n = _time(c, tl.where(valid, s, 0), N, T, ADJOINT)
            m = _step_matrix(
                trans, emit, p, q, n, K, stride_tn, rows, states[None, :], ADJOINT,
                SEMIRING,
            )  # fmt: skip
            # (y (x) m)[k] = sum over i of y[i] (x) m[i, k].
            terms = _times(y[:, None], m, SEMIRING)
            y_next = _reduce(terms, 0, SEMIRING)
            if argmax is not None:
                best = tl.argmax(terms, 0, tie_break_left=True)
                tl.store(argmax + n * K + states, best, mask=(states < K) & valid)
            if inj is not None:
                y_next += tl.load(inj + n * K + states, mask=states < K, other=0.0)
            tl.store(out + n * K + states, y_next, mask=(states < K) & valid)
            y = tl.where(valid, y_next, y)
    else:
        # A block of BC states at a time, through the output itself.
        block = tl.arange(0, BC)
        for s in range(0, steps):
            n = _time(c, s, N, T, ADJOINT)
            for k0 in range(0, BK, BC):
                columns = k0 + block
                m = _step_matrix(
                    trans, emit, p, q, n, K, stride_tn, rows, columns[None, :], ADJOINT,
                    SEMIRING,
                )  # fmt: skip
                terms = _times(y[:, None], m, SEMIRING)
                y_block = _reduce(terms, 0, SEMIRING)
                if argmax is not None:
                    best = tl.argmax(terms, 0, tie_break_left=True)
                    tl.store(argmax + n * K + columns, best, mask=columns < K)
                if inj is not None:
                    y_block += tl.load(inj + n * K + columns, mask=columns < K, other=0.0)
                tl.store(out + n * K + columns, y_block, mask=columns < K)
            tl.debug_barrier()
            y = tl.load(out + n * K + states, mask=states < K, other=0.0)


def _config(K: int, semiring: str) -> tuple[int, int, int, int]:
    """The padded K, the column and row tile sizes, and the warps.

    The paths and warps were measured on an RTX 5060 Ti: at 17 to 32 states,
    tiles are faster for log and max-plus chains, while a linear chain's
    weighted matrices cost more to rebuild per tile. The tiles hold about
    4096 terms, BR rows by BK summed by BC columns, a size chosen to fit,
    not tuned.
    """
    BK = max(triton.next_power_of_2(K), 2)
    registers = _REGISTER_STATES * (2 if semiring == "linear" else 1)
    if BK <= registers:
        return BK, BK, BK, 1 if BK <= 4 else 2 if BK <= 16 else 4
    BC = min(max(1024 // BK, 1), _REGISTER_STATES)
    return BK, BC, max(4096 // (BK * BC), 1), 4 if BK <= 32 else 8


def chain(
    y0: Tensor,
    trans: Tensor,
    log_emit: Tensor | None,
    inj: Tensor | None,
    semiring: str,
    adjoint: bool = False,
    out: Tensor | None = None,
    weights: tuple[Tensor, Tensor] | None = None,
    argmax: Tensor | None = None,
) -> Tensor:
    """The messages y[t] = (y[t - 1] (x) A[t]) + j[t], t = 0, ..., N - 1, from y[-1] = y0.

    The tensors with a value per step, ``log_emit``, ``inj``, ``weights``,
    ``out`` and ``argmax``, share one layout: each step's K values
    contiguous, the steps contiguous, and one batch stride, as for time
    slices of contiguous (B, N or N + 1, K) tensors.

    Args:
        y0: the starting vectors, (B, K).
        trans: (B, N, K, K), possibly with zero batch or time strides, and
            contiguous K x K matrices: log_trans, or explicit matrices for
            an unweighted linear chain. Or (2, B / 2, N, K, K) for two chains
            stacked in the batch, each half of it with its own matrices.
        log_emit: (B, N, K), added to the rows i of each log_trans[n][i, j],
            or None.
        inj: for a linear chain, the injections j, (B, N, K), or None.
        semiring: "log", "max" or "linear".
        adjoint: run the chain from time N - 1 down to 0, over each step's
            transposed matrix; the message after the step at time n is
            still written at n.
        out: where to write the messages, (B, N, K), or None for a new
            tensor.
        weights: for a linear chain, the log weights (p, q), each (B, N, K),
            of A[t][r, c] = exp(M[t][r, c] + p[t][r] + q[t][c]) with r
            summed; None for explicit matrices.
        argmax: for a max-plus chain, a (B, N, K) int32 tensor that receives
            each message's maximizing previous state, the lowest-index one
            under ties; or None.

    Returns:
        The messages, (B, N, K), each written at its step's time.
    """
    halves = trans.dim() == 5
    B = trans.shape[0] * trans.shape[1] if halves else trans.shape[0]
    N, K = trans.shape[-3], trans.shape[-1]
    if out is None:
        out = y0.new_empty(B, N, K)
    if B == 0 or N == 0 or K == 0:
        return out
    linear = semiring == "linear"
    assert linear or weights is None, "weights are for linear chains"
    assert weights is not None or not linear or log_emit is None, (
        "explicit matrices take no emissions"
    )
    assert linear or inj is None, "injections are for linear chains"
    assert not adjoint or weights is not None, "adjoint chains are weighted linear ones"
    assert argmax is None or semiring == "max", "argmax is for max-plus chains"
    p, q = weights if weights is not None else (None, None)
    per_step = [t for t in (out, log_emit, inj, p, q, argmax) if t is not None]
    assert all(t.stride()[1:] == (K, 1) for t in per_step), "steps of K contiguous values"
    stride_b = out.stride(0)
    assert all(t.stride(0) == stride_b for t in per_step), "one batch stride"
    T = _CHUNK
    C = triton.cdiv(N, T)
    # The matrices' half stride and per-half batch size, batch and time strides.
    if halves:
        trans_strides = (trans.shape[1], trans.stride(0), trans.stride(1), trans.stride(2))
    else:
        trans_strides = (B, 0, trans.stride(0), trans.stride(1))
    BK, BC, BR, num_warps = _config(K, semiring)
    flags = dict(T=T, ADJOINT=adjoint, SEMIRING=_SEMIRINGS[semiring])
    flags |= dict(BK=BK, BC=BC, num_warps=num_warps)
    if C == 1:
        starts = y0.contiguous()
    else:
        totals = y0.new_empty(B, C, K, K)
        offsets = y0.new_empty(B, C, K) if inj is not None else None
        scratch = y0.new_empty(B * C, 2 * BK * BK + BK) if BC < BK else None
        _chunk_totals_kernel[(B * C,)](
            trans, log_emit, p, q, inj, totals, offsets, scratch, N, K, C, *trans_strides,
            stride_b, **flags, BR=BR,
        )  # fmt: skip
        # The totals are in chain order and already transposed and weighted:
        # their own chain runs forward on them as explicit matrices, with the
        # offsets as injections.
        ends = chain(y0, totals, None, offsets, semiring)
        starts = torch.cat([y0.unsqueeze(1), ends[:, :-1]], dim=1).contiguous()
    _chunk_sweep_kernel[(B * C,)](
        starts, trans, log_emit, p, q, inj, out, argmax, N, K, C, *trans_strides, stride_b,
        **flags,
    )  # fmt: skip
    return out


@triton.jit
def _trace_totals_kernel(
    maps_ptr, total_ptr, L, K, C, stride_mb, T: tl.constexpr, REVERSE: tl.constexpr,
    BK: tl.constexpr,
):  # fmt: skip
    """Chunk c's composed map, total[k] = F[cT + T - 1](... F[cT](k)), in chain order."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, L - c * T)
    maps = maps_ptr + b * stride_mb
    states = tl.arange(0, BK)
    current = states
    for s in range(0, steps):
        n = _time(c, s, L, T, REVERSE)
        current = tl.load(maps + n * K + current, mask=states < K, other=0)
    tl.store(total_ptr + pid.to(tl.int64) * K + states, current, mask=states < K)


@triton.jit
def _trace_sweep_kernel(
    start_ptr, maps_ptr, out_ptr, L, K, C, stride_mb, T: tl.constexpr, REVERSE: tl.constexpr
):
    """x[t] = F[t](x[t - 1]) through chunk c from its start, written at time n."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, L - c * T)
    maps = maps_ptr + b * stride_mb
    current = tl.load(start_ptr + pid)
    for s in range(0, steps):
        n = _time(c, s, L, T, REVERSE)
        current = tl.load(maps + n * K + current)
        tl.store(out_ptr + b * L + n, current)


def trace(x0: Tensor, maps: Tensor, reverse: bool = False) -> Tensor:
    """The states x[t] = maps[t][x[t - 1]], t = 0, ..., L - 1, from x[-1] = x0.

    Following backpointers is a chain of maps of K states, so it runs as the
    message chains do: each chunk's maps composed, the chain of the
    compositions one level up, then each chunk from its start.

    Args:
        x0: the starting states, (B,) int32.
        maps: (B, L, K) int32, each step's K values contiguous, the steps
            contiguous; maps[b, t, k] is the state that state k leads to at
            step t.
        reverse: run from time L - 1 down to 0; the state after the step at
            time n is still written at n.

    Returns:
        The states, (B, L) int32, each written at its step's time.
    """
    B, L, K = maps.shape
    out = maps.new_empty(B, L)
    if B == 0 or L == 0:
        return out
    assert maps.stride()[1:] == (K, 1), "steps of K contiguous values"
    T = _CHUNK
    C = triton.cdiv(L, T)
    if C == 1:
        starts = x0.contiguous()
    else:
        totals = maps.new_empty(B, C, K)
        BK = max(triton.next_power_of_2(K), 2)
        _trace_totals_kernel[(B * C,)](
            maps, totals, L, K, C, maps.stride(0), T=T, REVERSE=reverse, BK=BK, num_warps=1
        )
        ends = trace(x0, totals)
        starts = torch.cat([x0.unsqueeze(1), ends[:, :-1]], dim=1).contiguous()
    _trace_sweep_kernel[(B * C,)](
        starts, maps, out, L, K, C, maps.stride(0), T=T, REVERSE=reverse, num_warps=1
    )
    return out
