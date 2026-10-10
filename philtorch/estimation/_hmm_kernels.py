"""Chunked parallel scans of HMM messages, in Triton.

A message chain

    y[t] = (y[t - 1] (x) A[t]) (+) j[t],   t = 0, ..., N - 1,

with y a vector of K states, A[t] K x K and an optional injection j[t], is
computed as a parallel scan in two levels instead of N sequential steps:

1. ``_chunk_totals_kernel`` composes each chunk of T consecutive steps into
   one map y_end = (y_start (x) P) (+) o, one program per chunk;
2. the chain of those maps, whose injections are the o, gives each chunk's
   starting vector, by the same method one level up;
3. ``_chunk_sweep_kernel`` carries each chunk's starting vector through its
   steps, one program per chunk.

So a chain costs about N K^3 work for the totals and N K^2 for the sweep, a
few launches, and a sequential depth of a few T. The semiring is one of

* ``"log"``: (x) is the vector-matrix product with logsumexp in place of
  sums, (+) is logsumexp, and A[t] = M[t], log-probabilities;
* ``"max"``: the same with max in place of logsumexp;
* ``"linear"``: the ordinary product and sum, with A[t] = exp(M[t][r, c] +
  p[t][r] + q[t][c]) for log weights p and q, or explicit matrices.

M[t] at time n is log_trans[n] plus log_emit[n] on its output states'
columns, or with ``EMIT_SUMMED`` on its summed states' rows, built in the
kernels from ``log_trans``, which may be shared over the batch or time
through zero strides, so the B x N x K x K step matrices are never stored.
With ``REVERSE``, the chain runs from the last time back, and with
``TRANSPOSE`` it uses each log_trans[n]'s transpose: so a chain's
derivative, a linear chain backwards, runs here too.

A max-plus chain can also record each message's maximizing previous state,
and :func:`trace` follows such backpointers in the same two levels: Viterbi
decoding's traceback, in parallel.

Up to ``_REGISTER_STATES`` states, a program holds the K x K x K terms of a
product in registers. Above, it computes them in tiles of rows and columns,
and the totals' running product alternates between two K x K scratch
buffers, so registers and shared memory stay bounded for any K.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

_SEMIRINGS = {"log": 0, "max": 1, "linear": 2}
# Triton kernels can only read module constants made with tl.constexpr.
_MAX, _LINEAR = (tl.constexpr(_SEMIRINGS[name]) for name in ("max", "linear"))
_NEG_INF = float("-inf")
_KERNEL_NEG_INF = tl.constexpr(_NEG_INF)
# The chunk length: each kernel program takes T sequential steps.
_CHUNK = 64
# The largest K whose products a program holds whole in registers.
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
def _plus(a, b, SEMIRING: tl.constexpr):
    if SEMIRING == _LINEAR:
        return a + b
    top = tl.maximum(a, b)
    if SEMIRING == _MAX:
        return top
    shift = tl.where(top == _KERNEL_NEG_INF, 0.0, top)
    return shift + tl.log(tl.exp(a - shift) + tl.exp(b - shift))


@triton.jit
def _time(c, s, N, T: tl.constexpr, REVERSE: tl.constexpr):
    """The time of step s of chunk c, a 64-bit index."""
    t = c * T + s
    return (N - 1 - t) if REVERSE else t


@triton.jit
def _step_matrix(
    trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn, rows, cols,
    HAS_EMIT: tl.constexpr, EMIT_SUMMED: tl.constexpr, TRANSPOSE: tl.constexpr,
    WEIGHTED: tl.constexpr, SEMIRING: tl.constexpr,
):  # fmt: skip
    """Entries (rows, cols) of A at time n, from pointers offset to the batch item.

    The semiring's zero outside K x K.
    """
    mask = (rows < K) & (cols < K)
    # (i, j) of the stored matrix for entry (rows, cols).
    i, j = (cols, rows) if TRANSPOSE else (rows, cols)
    offsets = trans + n * stride_tn + i * K + j
    if SEMIRING == _LINEAR and not WEIGHTED:
        return tl.load(offsets, mask=mask, other=0.0)
    m = tl.load(offsets, mask=mask, other=_KERNEL_NEG_INF)
    if HAS_EMIT:
        # The emissions of the summed or of the output states, indexed in the
        # tile's own shape: a broadcast vector would cost a layout conversion.
        e = rows + 0 * cols if EMIT_SUMMED else cols + 0 * rows
        m += tl.load(emit + n * stride_en + e, mask=mask, other=0.0)
    if WEIGHTED:
        m += tl.load(p + n * stride_pn + rows, mask=rows < K, other=_KERNEL_NEG_INF)
        m += tl.load(q + n * stride_qn + cols, mask=cols < K, other=_KERNEL_NEG_INF)
        m = tl.exp(m)
    return m


@triton.jit
def _chunk_totals_kernel(
    trans_ptr, emit_ptr, p_ptr, q_ptr, inj_ptr, total_ptr, offset_ptr, scratch_ptr, N, K, C,
    HB, stride_th, stride_tb, stride_tn, stride_eb, stride_en, stride_pb, stride_pn, stride_qb,
    stride_qn,
    stride_jb, stride_jn, ZERO: tl.constexpr, T: tl.constexpr, HAS_EMIT: tl.constexpr,
    EMIT_SUMMED: tl.constexpr, HAS_INJ: tl.constexpr, WEIGHTED: tl.constexpr,
    REVERSE: tl.constexpr,
    TRANSPOSE: tl.constexpr, SEMIRING: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
    BR: tl.constexpr,
):  # fmt: skip
    """Chunk c's map: total = A[cT] (x) ... (x) A[cT + T - 1], in chain order, and
    offset, the chain through those steps of the injections alone."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, N - c * T)
    # A stacked pair of chains reads its own matrices per half of the batch.
    trans = trans_ptr + (b // HB) * stride_th + (b % HB) * stride_tb
    emit = emit_ptr + b * stride_eb
    p, q, inj = p_ptr + b * stride_pb, q_ptr + b * stride_qb, inj_ptr + b * stride_jb
    states = tl.arange(0, BK)
    rows = states[:, None]
    cols = states[None, :]
    n = _time(c, 0, N, T, REVERSE)
    offset = tl.full([BK], ZERO, total_ptr.dtype.element_ty)
    if HAS_INJ:
        offset = tl.load(inj + n * stride_jn + states, mask=states < K, other=ZERO)
    out = total_ptr + pid.to(tl.int64) * K * K
    if BC == BK:
        total = _step_matrix(
            trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn, rows, cols,
            HAS_EMIT, EMIT_SUMMED, TRANSPOSE, WEIGHTED, SEMIRING,
        )  # fmt: skip
        # A loop of constant length, with steps past the end masked, unrolls
        # and pipelines better for these small matrices.
        for s in range(1, T):
            valid = s < steps
            n = _time(c, tl.where(valid, s, 0), N, T, REVERSE)
            m = _step_matrix(
                trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn, rows, cols,
                HAS_EMIT, EMIT_SUMMED, TRANSPOSE, WEIGHTED, SEMIRING,
            )  # fmt: skip
            # (total (x) m)[i, k] = sum over j of total[i, j] (x) m[j, k].
            product = _reduce(_times(total[:, :, None], m[None, :, :], SEMIRING), 1, SEMIRING)
            total = tl.where(valid, product, total)
            if HAS_INJ:
                o = _reduce(_times(offset[:, None], m, SEMIRING), 0, SEMIRING)
                j = tl.load(inj + n * stride_jn + states, mask=states < K, other=ZERO)
                offset = tl.where(valid, _plus(o, j, SEMIRING), offset)
        tl.store(out + rows * K + cols, total, mask=(rows < K) & (cols < K))
    else:
        # Tiles of BR rows and BC columns: the running product alternates
        # between two K x K scratch buffers, and only tiles are in registers.
        scratch = scratch_ptr + pid.to(tl.int64) * (2 * BK * BK + BK)
        tile_rows = tl.arange(0, BR)[:, None]
        block = tl.arange(0, BC)
        for k0 in range(0, BK, BC):
            columns = k0 + block[None, :]
            m = _step_matrix(
                trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn,
                rows, columns, HAS_EMIT, EMIT_SUMMED, TRANSPOSE, WEIGHTED, SEMIRING,
            )  # fmt: skip
            tl.store(scratch + rows * BK + columns, m)
        tl.debug_barrier()
        for s in range(1, steps):
            n = _time(c, s, N, T, REVERSE)
            current = scratch + ((s - 1) % 2) * BK * BK
            following = scratch + (s % 2) * BK * BK
            for k0 in range(0, BK, BC):
                columns = k0 + block
                m = _step_matrix(
                    trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn,
                    rows, columns[None, :], HAS_EMIT, EMIT_SUMMED, TRANSPOSE, WEIGHTED, SEMIRING,
                )  # fmt: skip
                for i0 in range(0, BK, BR):
                    a = tl.load(current + (i0 + tile_rows) * BK + cols)
                    product = _reduce(_times(a[:, :, None], m[None, :, :], SEMIRING), 1, SEMIRING)
                    tl.store(following + (i0 + tile_rows) * BK + columns[None, :], product)
                if HAS_INJ:
                    o = _reduce(_times(offset[:, None], m, SEMIRING), 0, SEMIRING)
                    j = tl.load(inj + n * stride_jn + columns, mask=columns < K, other=ZERO)
                    tl.store(scratch + 2 * BK * BK + columns, _plus(o, j, SEMIRING))
            tl.debug_barrier()
            if HAS_INJ:
                offset = tl.load(scratch + 2 * BK * BK + states)
            # Every thread reads the offset before the next step rewrites it.
            tl.debug_barrier()
        last = scratch + ((steps - 1) % 2) * BK * BK
        for k0 in range(0, BK, BC):
            columns = k0 + block[None, :]
            tile = tl.load(last + rows * BK + columns)
            tl.store(out + rows * K + columns, tile, mask=(rows < K) & (columns < K))
    if HAS_INJ:
        tl.store(offset_ptr + pid.to(tl.int64) * K + states, offset, mask=states < K)


@triton.jit
def _chunk_sweep_kernel(
    start_ptr, trans_ptr, emit_ptr, p_ptr, q_ptr, inj_ptr, out_ptr, argmax_ptr, N, K, C,
    HB, stride_th, stride_tb, stride_tn, stride_eb, stride_en, stride_pb, stride_pn, stride_qb,
    stride_qn,
    stride_jb, stride_jn, stride_ob, ZERO: tl.constexpr, T: tl.constexpr,
    HAS_EMIT: tl.constexpr, EMIT_SUMMED: tl.constexpr, HAS_INJ: tl.constexpr,
    WEIGHTED: tl.constexpr, REVERSE: tl.constexpr, TRANSPOSE: tl.constexpr,
    SEMIRING: tl.constexpr,
    ARGMAX: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
):  # fmt: skip
    """y[t] = (y[t - 1] (x) A[t]) (+) j[t] through chunk c from its start, written at time n.

    With ARGMAX, a max-plus chain also writes, for each state k, the
    lowest-index state i attaining y[t][k] = y[t - 1][i] + A[t][i, k].
    """
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, N - c * T)
    # A stacked pair of chains reads its own matrices per half of the batch.
    trans = trans_ptr + (b // HB) * stride_th + (b % HB) * stride_tb
    emit = emit_ptr + b * stride_eb
    p, q, inj = p_ptr + b * stride_pb, q_ptr + b * stride_qb, inj_ptr + b * stride_jb
    out = out_ptr + b * stride_ob
    argmax = argmax_ptr + b * N * K
    states = tl.arange(0, BK)
    rows = states[:, None]
    y = tl.load(start_ptr + pid.to(tl.int64) * K + states, mask=states < K, other=ZERO)
    if BC == BK:
        for s in range(0, T):
            valid = s < steps
            n = _time(c, tl.where(valid, s, 0), N, T, REVERSE)
            m = _step_matrix(
                trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn,
                rows, states[None, :], HAS_EMIT, EMIT_SUMMED, TRANSPOSE, WEIGHTED, SEMIRING,
            )  # fmt: skip
            # (y (x) m)[k] = sum over i of y[i] (x) m[i, k].
            terms = _times(y[:, None], m, SEMIRING)
            y_next = _reduce(terms, 0, SEMIRING)
            if ARGMAX:
                best = tl.argmax(terms, 0, tie_break_left=True)
                tl.store(argmax + n * K + states, best, mask=(states < K) & valid)
            if HAS_INJ:
                j = tl.load(inj + n * stride_jn + states, mask=states < K, other=ZERO)
                y_next = _plus(y_next, j, SEMIRING)
            tl.store(out + n * K + states, y_next, mask=(states < K) & valid)
            y = tl.where(valid, y_next, y)
    else:
        # A block of BC states at a time, through the output itself.
        block = tl.arange(0, BC)
        for s in range(0, steps):
            n = _time(c, s, N, T, REVERSE)
            for k0 in range(0, BK, BC):
                columns = k0 + block
                m = _step_matrix(
                    trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn,
                    rows, columns[None, :], HAS_EMIT, EMIT_SUMMED, TRANSPOSE, WEIGHTED, SEMIRING,
                )  # fmt: skip
                terms = _times(y[:, None], m, SEMIRING)
                y_block = _reduce(terms, 0, SEMIRING)
                if ARGMAX:
                    best = tl.argmax(terms, 0, tie_break_left=True)
                    tl.store(argmax + n * K + columns, best, mask=columns < K)
                if HAS_INJ:
                    j = tl.load(inj + n * stride_jn + columns, mask=columns < K, other=ZERO)
                    y_block = _plus(y_block, j, SEMIRING)
                tl.store(out + n * K + columns, y_block, mask=columns < K)
            tl.debug_barrier()
            y = tl.load(out + n * K + states, mask=states < K, other=ZERO)


def _config(K: int) -> tuple[int, int, int, int]:
    """The padded K, the column and row tile sizes, and the warps.

    The warps for the register path were measured on an RTX 5060 Ti; the
    tiles of the blocked path hold about 4096 terms, BR rows by BK summed by
    BC columns, a size chosen to fit, not tuned.
    """
    BK = max(triton.next_power_of_2(K), 2)
    if BK <= _REGISTER_STATES:
        return BK, BK, BK, 1 if BK <= 4 else 2
    BC = max(1024 // BK, 1)
    return BK, BC, max(4096 // (BK * BC), 1), 4 if BK <= 32 else 8


def _strides(t: Tensor | None) -> tuple[int, int]:
    """The batch and time strides of a (B, N, ...) view; 0 where broadcast."""
    return (t.stride(0), t.stride(1)) if t is not None else (0, 0)


def _unit_stride(t: Tensor | None) -> Tensor | None:
    """t with a contiguous last dimension, which the kernels index directly."""
    return t if t is None or t.stride(-1) == 1 else t.contiguous()


def chain(
    y0: Tensor,
    trans: Tensor,
    log_emit: Tensor | None,
    inj: Tensor | None,
    semiring: str,
    reverse: bool = False,
    transpose: bool = False,
    emit_summed: bool = False,
    out: Tensor | None = None,
    weights: tuple[Tensor, Tensor] | None = None,
    argmax: Tensor | None = None,
) -> Tensor:
    """The messages y[t] = (y[t - 1] (x) A[t]) (+) j[t], t = 0, ..., N - 1, from y[-1] = y0.

    Args:
        y0: the starting vectors, (B, K).
        trans: (B, N, K, K), possibly with zero batch or time strides, and
            contiguous K x K matrices: log_trans, or explicit matrices for
            an unweighted linear chain. Or (2, B / 2, N, K, K) for two chains
            stacked in the batch, each half of it with its own matrices.
        log_emit: (B, N, K), added to each step's matrix, or None.
        inj: the injections j, (B, N, K), or None.
        semiring: "log", "max" or "linear".
        reverse: run the chain from time N - 1 down to 0; the message after
            the step at time n is still written at n.
        transpose: multiply by each matrix's transpose.
        emit_summed: add the emissions to the summed states' rows rather
            than to the output states' columns.
        out: where to write the messages, a (B, N, K) view whose steps are
            contiguous, such as a slice in time of a larger output, or None.
        weights: for a linear chain, the log weights (p, q), each (B, N, K),
            of A[t][r, c] = exp(M[t][r, c] + p[t][r] + q[t][c]) with r
            summed; None for explicit matrices.
        argmax: for a max-plus chain, a contiguous (B, N, K) int32 tensor
            that receives each message's maximizing previous state, the
            lowest-index one under ties; or None.

    Returns:
        The messages, (B, N, K), each written at its step's time.
    """
    halves = trans.dim() == 5
    B, N, K = (
        trans.shape[0] * trans.shape[1] if halves else trans.shape[0],
        trans.shape[-3],
        trans.shape[-1],
    )
    if out is None:
        out = y0.new_empty(B, N, K)
    if B == 0 or N == 0 or K == 0:
        return out
    linear = semiring == "linear"
    weighted = weights is not None
    assert linear or not weighted, "weights are for linear chains"
    assert weighted or not linear or log_emit is None, "explicit matrices take no emissions"
    assert argmax is None or semiring == "max", "argmax is for max-plus chains"
    T = _CHUNK
    C = triton.cdiv(N, T)
    p, q = weights if weighted else (None, None)
    log_emit, p, q, inj = (_unit_stride(t) for t in (log_emit, p, q, inj))
    pointers = [t if t is not None else y0 for t in (log_emit, p, q, inj)]
    # The matrices' half stride and per-half batch size, then each tensor's
    # batch and time strides.
    HB, stride_th = (trans.shape[1], trans.stride(0)) if halves else (B, 0)
    trans_strides = (trans.stride(1), trans.stride(2)) if halves else _strides(trans)
    strides = [HB, stride_th, *trans_strides]
    strides += [s for t in (log_emit, p, q, inj) for s in _strides(t)]
    BK, BC, BR, num_warps = _config(K)
    flags = dict(ZERO=0.0 if linear else _NEG_INF, T=T, HAS_EMIT=log_emit is not None)
    flags |= dict(EMIT_SUMMED=emit_summed)
    flags |= dict(HAS_INJ=inj is not None, WEIGHTED=weighted, REVERSE=reverse)
    flags |= dict(TRANSPOSE=transpose, SEMIRING=_SEMIRINGS[semiring], BK=BK, BC=BC)
    if C == 1:
        starts = y0.contiguous()
    else:
        totals = y0.new_empty(B, C, K, K)
        offsets = y0.new_empty(B, C, K) if inj is not None else None
        scratch = y0.new_empty(B * C, 2 * BK * BK + BK) if BC < BK else y0
        _chunk_totals_kernel[(B * C,)](
            trans, *pointers, totals, offsets if offsets is not None else y0, scratch, N, K, C,
            *strides, **flags, BR=BR, num_warps=num_warps,
        )  # fmt: skip
        # The totals are in chain order and already transposed and weighted:
        # their own chain runs forward on them as explicit matrices, with the
        # offsets as injections.
        ends = chain(y0, totals, None, offsets, semiring)
        starts = torch.cat([y0.unsqueeze(1), ends[:, :-1]], dim=1).contiguous()
    _chunk_sweep_kernel[(B * C,)](
        starts, trans, *pointers, out, argmax if argmax is not None else starts, N, K, C,
        *strides, out.stride(0), **flags, ARGMAX=argmax is not None, num_warps=num_warps,
    )  # fmt: skip
    return out


@triton.jit
def _trace_totals_kernel(
    maps_ptr, total_ptr, L, K, C, T: tl.constexpr, REVERSE: tl.constexpr, BK: tl.constexpr
):
    """Chunk c's composed map, total[k] = F[cT + T - 1](... F[cT](k)), in chain order."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, L - c * T)
    maps = maps_ptr + b * L * K
    states = tl.arange(0, BK)
    current = states
    for s in range(0, steps):
        n = _time(c, s, L, T, REVERSE)
        current = tl.load(maps + n * K + current, mask=states < K, other=0)
    tl.store(total_ptr + pid.to(tl.int64) * K + states, current, mask=states < K)


@triton.jit
def _trace_sweep_kernel(
    start_ptr, maps_ptr, out_ptr, L, K, C, T: tl.constexpr, REVERSE: tl.constexpr
):
    """x[t] = F[t](x[t - 1]) through chunk c from its start, written at time n."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, L - c * T)
    maps = maps_ptr + b * L * K
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
        maps: (B, L, K) int32, contiguous; maps[b, t, k] is the state that
            state k leads to at step t.
        reverse: run from time L - 1 down to 0; the state after the step at
            time n is still written at n.

    Returns:
        The states, (B, L) int32, each written at its step's time.
    """
    B, L, K = maps.shape
    out = maps.new_empty(B, L)
    if B == 0 or L == 0:
        return out
    T = _CHUNK
    C = triton.cdiv(L, T)
    if C == 1:
        starts = x0.contiguous()
    else:
        totals = maps.new_empty(B, C, K)
        BK = max(triton.next_power_of_2(K), 2)
        _trace_totals_kernel[(B * C,)](
            maps, totals, L, K, C, T=T, REVERSE=reverse, BK=BK, num_warps=1
        )
        ends = trace(x0, totals)
        starts = torch.cat([x0.unsqueeze(1), ends[:, :-1]], dim=1).contiguous()
    _trace_sweep_kernel[(B * C,)](starts, maps, out, L, K, C, T=T, REVERSE=reverse, num_warps=1)
    return out
