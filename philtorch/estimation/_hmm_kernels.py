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

* ``LOG``: (x) is the vector-matrix product with logsumexp in place of sums,
  (+) is logsumexp, and A[t] = M[t], log-probabilities;
* ``MAX``: the same with max in place of logsumexp;
* ``LINEAR``: the ordinary product and sum, with A[t] = exp(M[t][r, c] +
  p[t][r] + q[t][c]) for log weights p and q, or explicit matrices.

M[t] at time n is log_trans[n][i, j] + log_emit[n][j], built in the kernels
from ``log_trans``, which may be shared over the batch or time through zero
strides, so the B x N x K x K step matrices are never stored. With
``REVERSE``, the chain runs from the last time back, and with ``TRANSPOSE``
it uses the transpose, whose emissions are then on the summed index: so a
chain's derivative, a linear chain backwards, runs here too.

Up to ``_REGISTER_STATES`` states, a program holds the K x K x K terms of a
product in registers. Above, it computes them a block of columns at a time
and passes the running product between steps through a scratch buffer, so
any K fits.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

LOG, MAX, LINEAR = 0, 1, 2
_NEG_INF = float("-inf")
# Triton kernels can only read module constants made with tl.constexpr.
_KERNEL_NEG_INF = tl.constexpr(_NEG_INF)
# The chunk length: each kernel program takes T sequential steps.
_CHUNK = 64
# The largest K whose products a program holds whole in registers.
_REGISTER_STATES = 16


@triton.jit
def _reduce(x, axis: tl.constexpr, SEMIRING: tl.constexpr):
    """The semiring's sum along ``axis``; logsumexp is -inf where every term is."""
    if SEMIRING == 2:
        return tl.sum(x, axis)
    top = tl.max(x, axis)
    if SEMIRING == 1:
        return top
    shift = tl.where(top == _KERNEL_NEG_INF, 0.0, top)
    return shift + tl.log(tl.sum(tl.exp(x - tl.expand_dims(shift, axis)), axis))


@triton.jit
def _times(a, b, SEMIRING: tl.constexpr):
    if SEMIRING == 2:
        return a * b
    return a + b


@triton.jit
def _plus(a, b, SEMIRING: tl.constexpr):
    if SEMIRING == 2:
        return a + b
    top = tl.maximum(a, b)
    if SEMIRING == 1:
        return top
    shift = tl.where(top == _KERNEL_NEG_INF, 0.0, top)
    return shift + tl.log(tl.exp(a - shift) + tl.exp(b - shift))


@triton.jit
def _time(c, s, N, T: tl.constexpr, REVERSE: tl.constexpr):
    """The time of step s of chunk c, in 64 bits."""
    t = (c * T + s).to(tl.int64)
    return (N - 1 - t) if REVERSE else t


@triton.jit
def _step_matrix(
    trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn, rows, cols,
    HAS_EMIT: tl.constexpr, TRANSPOSE: tl.constexpr, WEIGHTED: tl.constexpr,
    SEMIRING: tl.constexpr,
):  # fmt: skip
    """Entries (rows, cols) of A at time n, from pointers offset to the batch item.

    The semiring's zero outside K x K.
    """
    mask = (rows < K) & (cols < K)
    # (i, j) of the stored matrix for entry (rows, cols); the emissions are
    # on the stored column, which the transpose makes the summed index.
    i, j = (cols, rows) if TRANSPOSE else (rows, cols)
    offsets = trans + n * stride_tn + i * K + j
    if SEMIRING == 2 and not WEIGHTED:
        return tl.load(offsets, mask=mask, other=0.0)
    m = tl.load(offsets, mask=mask, other=_KERNEL_NEG_INF)
    if HAS_EMIT:
        m += tl.load(emit + n * stride_en + j, mask=mask, other=0.0)
    if WEIGHTED:
        m += tl.load(p + n * stride_pn + rows, mask=rows < K, other=_KERNEL_NEG_INF)
        m += tl.load(q + n * stride_qn + cols, mask=cols < K, other=_KERNEL_NEG_INF)
        m = tl.exp(m)
    return m


@triton.jit
def _chunk_totals_kernel(
    trans_ptr, emit_ptr, p_ptr, q_ptr, inj_ptr, total_ptr, offset_ptr, scratch_ptr, N, K, C,
    stride_tb, stride_tn, stride_eb, stride_en, stride_pb, stride_pn, stride_qb, stride_qn,
    stride_jb, stride_jn, ZERO: tl.constexpr, T: tl.constexpr, HAS_EMIT: tl.constexpr,
    HAS_INJ: tl.constexpr, WEIGHTED: tl.constexpr, REVERSE: tl.constexpr,
    TRANSPOSE: tl.constexpr, SEMIRING: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
    BR: tl.constexpr,
):  # fmt: skip
    """Chunk c's map: total = A[cT] (x) ... (x) A[cT + T - 1], in chain order, and
    offset, the chain through those steps of the injections alone."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = pid % C
    steps = tl.minimum(T, N - c * T)
    trans, emit = trans_ptr + b * stride_tb, emit_ptr + b * stride_eb
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
            HAS_EMIT, TRANSPOSE, WEIGHTED, SEMIRING,
        )  # fmt: skip
        # A loop of constant length, with steps past the end masked, unrolls
        # and pipelines better for these small matrices.
        for s in range(1, T):
            valid = s < steps
            n = _time(c, tl.where(valid, s, 0), N, T, REVERSE)
            m = _step_matrix(
                trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn, rows, cols,
                HAS_EMIT, TRANSPOSE, WEIGHTED, SEMIRING,
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
                rows, columns, HAS_EMIT, TRANSPOSE, WEIGHTED, SEMIRING,
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
                    rows, columns[None, :], HAS_EMIT, TRANSPOSE, WEIGHTED, SEMIRING,
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
    start_ptr, trans_ptr, emit_ptr, p_ptr, q_ptr, inj_ptr, out_ptr, N, K, C,
    stride_tb, stride_tn, stride_eb, stride_en, stride_pb, stride_pn, stride_qb, stride_qn,
    stride_jb, stride_jn, stride_ob, ZERO: tl.constexpr, T: tl.constexpr,
    HAS_EMIT: tl.constexpr, HAS_INJ: tl.constexpr, WEIGHTED: tl.constexpr,
    REVERSE: tl.constexpr, TRANSPOSE: tl.constexpr, SEMIRING: tl.constexpr,
    BK: tl.constexpr, BC: tl.constexpr,
):  # fmt: skip
    """y[t] = (y[t - 1] (x) A[t]) (+) j[t] through chunk c from its start, written at time n."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = pid % C
    steps = tl.minimum(T, N - c * T)
    trans, emit = trans_ptr + b * stride_tb, emit_ptr + b * stride_eb
    p, q, inj = p_ptr + b * stride_pb, q_ptr + b * stride_qb, inj_ptr + b * stride_jb
    out = out_ptr + b * stride_ob
    states = tl.arange(0, BK)
    rows = states[:, None]
    y = tl.load(start_ptr + pid.to(tl.int64) * K + states, mask=states < K, other=ZERO)
    if BC == BK:
        for s in range(0, T):
            valid = s < steps
            n = _time(c, tl.where(valid, s, 0), N, T, REVERSE)
            m = _step_matrix(
                trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn,
                rows, states[None, :], HAS_EMIT, TRANSPOSE, WEIGHTED, SEMIRING,
            )  # fmt: skip
            # (y (x) m)[k] = sum over i of y[i] (x) m[i, k].
            y_next = _reduce(_times(y[:, None], m, SEMIRING), 0, SEMIRING)
            if HAS_INJ:
                j = tl.load(inj + n * stride_jn + states, mask=states < K, other=ZERO)
                y_next = _plus(y_next, j, SEMIRING)
            tl.store(out + n * K + states, y_next, mask=(states < K) & valid)
            y = tl.where(valid, y_next, y)
    else:
        for s in range(0, steps):
            n = _time(c, s, N, T, REVERSE)
            # A block of BC states at a time, through the output itself.
            block = tl.arange(0, BC)
            for k0 in range(0, BK, BC):
                columns = k0 + block
                m = _step_matrix(
                    trans, emit, p, q, n, K, stride_tn, stride_en, stride_pn, stride_qn,
                    rows, columns[None, :], HAS_EMIT, TRANSPOSE, WEIGHTED, SEMIRING,
                )  # fmt: skip
                y_block = _reduce(_times(y[:, None], m, SEMIRING), 0, SEMIRING)
                if HAS_INJ:
                    j = tl.load(inj + n * stride_jn + columns, mask=columns < K, other=ZERO)
                    y_block = _plus(y_block, j, SEMIRING)
                tl.store(out + n * K + columns, y_block, mask=columns < K)
            tl.debug_barrier()
            y = tl.load(out + n * K + states, mask=states < K, other=ZERO)


def _config(K: int) -> dict:
    """Blocks and warps by K (measured on an RTX 5060 Ti)."""
    BK = max(triton.next_power_of_2(K), 2)
    if BK <= _REGISTER_STATES:
        return dict(BK=BK, BC=BK, BR=BK, num_warps=1 if BK <= 4 else 2)
    # Tiles of about 4096 terms: BR rows by BK summed by BC columns.
    BC = max(1024 // BK, 1)
    return dict(BK=BK, BC=BC, BR=max(4096 // (BK * BC), 1), num_warps=4 if BK <= 32 else 8)


def _strides(t: Tensor | None) -> tuple[int, int]:
    """The batch and time strides of a (B, N, ...) view; 0 where broadcast."""
    return (t.stride(0), t.stride(1)) if t is not None else (0, 0)


def chain(
    y0: Tensor,
    trans: Tensor,
    log_emit: Tensor | None,
    inj: Tensor | None,
    semiring: int,
    reverse: bool = False,
    transpose: bool = False,
    out: Tensor | None = None,
    weights: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """The messages y[t] = (y[t - 1] (x) A[t]) (+) j[t], t = 0, ..., N - 1, from y[-1] = y0.

    Args:
        y0: the starting vectors, (B, K).
        trans: (B, N, K, K), possibly with zero batch or time strides, and
            contiguous K x K matrices: log_trans, or explicit matrices for
            an unweighted ``LINEAR`` chain.
        log_emit: (B, N, K) with a contiguous last dimension, added to each
            matrix's stored columns, or None.
        inj: the injections j, (B, N, K) with a contiguous last dimension,
            or None.
        semiring: ``LOG``, ``MAX`` or ``LINEAR``.
        reverse: run the chain from time N - 1 down to 0; the message after
            the step at time n is still written at n.
        transpose: multiply by each matrix's transpose.
        out: where to write the messages, a (B, N, K) view whose steps are
            contiguous, such as a slice in time of a larger output, or None.
        weights: for ``LINEAR``, the log weights (p, q), each (B, N, K) with
            a contiguous last dimension, of A[t][r, c] = exp(M[t][r, c] +
            p[t][r] + q[t][c]) with r summed; None for explicit matrices.

    Returns:
        The messages, (B, N, K), each written at its step's time.
    """
    B, N, K = trans.shape[0], trans.shape[1], trans.shape[-1]
    if out is None:
        out = y0.new_empty(B, N, K)
    if B == 0 or N == 0 or K == 0:
        return out
    weighted = weights is not None
    assert semiring == LINEAR or not weighted, "weights are for LINEAR chains"
    assert weighted or semiring != LINEAR or log_emit is None, "explicit matrices take no emissions"
    T = _CHUNK
    C = triton.cdiv(N, T)
    p, q = weights if weighted else (None, None)
    pointers = [t if t is not None else y0 for t in (log_emit, p, q, inj)]
    strides = [s for t in (trans, log_emit, p, q, inj) for s in _strides(t)]
    config = _config(K)
    flags = dict(ZERO=0.0 if semiring == LINEAR else _NEG_INF, T=T)
    flags |= dict(HAS_EMIT=log_emit is not None, HAS_INJ=inj is not None, WEIGHTED=weighted)
    flags |= dict(REVERSE=reverse, TRANSPOSE=transpose, SEMIRING=semiring, **config)
    if C == 1:
        starts = y0.contiguous()
    else:
        totals = y0.new_empty(B, C, K, K)
        offsets = y0.new_empty(B, C, K) if inj is not None else None
        BK = config["BK"]
        scratch = y0.new_empty(B * C, 2 * BK * BK + BK) if config["BC"] < BK else y0
        _chunk_totals_kernel[(B * C,)](
            trans, *pointers, totals, offsets if offsets is not None else y0, scratch, N, K, C,
            *strides, **flags,
        )  # fmt: skip
        # The totals are in chain order and already transposed and weighted:
        # their own chain runs forward on them as explicit matrices, with the
        # offsets as injections.
        ends = chain(y0, totals, None, offsets, semiring)
        starts = torch.cat([y0.unsqueeze(1), ends[:, :-1]], dim=1).contiguous()
    del flags["BR"]  # the sweep takes no row tiles
    _chunk_sweep_kernel[(B * C,)](
        starts, trans, *pointers, out, N, K, C, *strides, out.stride(0), **flags,
    )  # fmt: skip
    return out
