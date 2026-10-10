"""Triton kernels for the linear recurrence h[t] = A[t] h[t - 1] + x[t].

:func:`recurrence` computes it on CUDA for any state size M up to
``_SWEEP_STATES`` (half that for complex), as a chunked scan in row-vector
form, y[t] = y[t - 1] A[t]^T + x[t], in two levels, as the HMM's message
chains do (:mod:`philtorch.estimation._hmm_kernels`):

1. ``_totals_kernel`` composes each chunk of ``_CHUNK`` steps into one map
   y_end = y_start P + o, one program per chunk;
2. the chain of those maps gives each chunk's starting state, by the same
   method one level up;
3. ``_sweep_kernel`` carries each chunk's starting state through its steps.

With time-invariant A, every chunk's P is the same power of A^T, computed
once by repeated squaring (``_power_kernel``), so the totals only carry
the offsets o: O(M^2) work per step, as the sweep. Time-varying A costs
O(M^3) per step for the totals. With many batch items the batch alone fills
the GPU, and one chunk per item, the sweep alone, does less work
(:func:`_sweep_only`).

Complex inputs run as their real and imaginary parts (``view_as_real``),
multiplied out: four real products per complex one, as complex arithmetic
takes anyway. Half and bfloat16 inputs accumulate in float32. Autograd and
the choice of kernel are :mod:`philtorch._recur`'s.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

# Steps per chunk of the scan.
_CHUNK = 64
# The largest padded state sizes, a complex state counting twice: of a
# time-varying scan, which holds M^3 products in registers, and of an M x M
# matrix in registers, for the sweep and the time-invariant scan. Between
# the two, a time-varying recurrence runs the sweep alone.
_SCAN_STATES = 32
_SWEEP_STATES = 128


@triton.jit
def _matrix(a_ptr, base, n, stride_t, M, rows, cols, ACC: tl.constexpr,
            TRANSPOSE: tl.constexpr, COMPLEX: tl.constexpr):  # fmt: skip
    """Entries (rows, cols) of the step's matrix at time n, or of its transpose;
    real and imaginary parts, the real part twice for a real input."""
    mask = (rows < M) & (cols < M)
    i, j = (cols, rows) if TRANSPOSE else (rows, cols)
    offset = base + n * stride_t + i * M + j
    if COMPLEX:
        re = tl.load(a_ptr + 2 * offset, mask=mask, other=0.0).to(ACC)
        im = tl.load(a_ptr + 2 * offset + 1, mask=mask, other=0.0).to(ACC)
        return re, im
    re = tl.load(a_ptr + offset, mask=mask, other=0.0).to(ACC)
    return re, re


@triton.jit
def _vector(ptr, offset, states, M, ACC: tl.constexpr, COMPLEX: tl.constexpr):
    mask = states < M
    if COMPLEX:
        re = tl.load(ptr + 2 * (offset + states), mask=mask, other=0.0).to(ACC)
        im = tl.load(ptr + 2 * (offset + states) + 1, mask=mask, other=0.0).to(ACC)
        return re, im
    re = tl.load(ptr + offset + states, mask=mask, other=0.0).to(ACC)
    return re, re


@triton.jit
def _store_vector(ptr, offset, states, M, re, im, COMPLEX: tl.constexpr):
    dtype = ptr.dtype.element_ty
    mask = states < M
    if COMPLEX:
        tl.store(ptr + 2 * (offset + states), re.to(dtype), mask=mask)
        tl.store(ptr + 2 * (offset + states) + 1, im.to(dtype), mask=mask)
    else:
        tl.store(ptr + offset + states, re.to(dtype), mask=mask)


@triton.jit
def _times(yr, yi, mr, mi, COMPLEX: tl.constexpr):
    """(y A)[k] = sum over i of y[i] A[i, k], real and imaginary parts."""
    if COMPLEX:
        re = tl.sum(yr[:, None] * mr - yi[:, None] * mi, 0)
        im = tl.sum(yr[:, None] * mi + yi[:, None] * mr, 0)
        return re, im
    re = tl.sum(yr[:, None] * mr, 0)
    return re, re


@triton.jit
def _matmul(pr, pi, mr, mi, COMPLEX: tl.constexpr):
    """(P A)[i, k] = sum over j of P[i, j] A[j, k], real and imaginary parts."""
    if COMPLEX:
        re = tl.sum(pr[:, :, None] * mr[None, :, :] - pi[:, :, None] * mi[None, :, :], 1)
        im = tl.sum(pr[:, :, None] * mi[None, :, :] + pi[:, :, None] * mr[None, :, :], 1)
        return re, im
    re = tl.sum(pr[:, :, None] * mr[None, :, :], 1)
    return re, re


@triton.jit
def _power_kernel(
    a_ptr, out_ptr, M, SQUARINGS: tl.constexpr, TRANSPOSE: tl.constexpr,
    COMPLEX: tl.constexpr, BM: tl.constexpr,
):  # fmt: skip
    """Matrix b's (or its transpose's) 2^SQUARINGS-th power, one program per matrix."""
    b = tl.program_id(0).to(tl.int64)
    ACC: tl.constexpr = out_ptr.dtype.element_ty
    states = tl.arange(0, BM)
    rows = states[:, None]
    cols = states[None, :]
    pr, pi = _matrix(a_ptr, b * M * M, 0, 0, M, rows, cols, ACC, TRANSPOSE, COMPLEX)
    for _ in tl.static_range(SQUARINGS):
        # P P needs P's rows and columns in the product's two roles.
        qr, qi = _matmul(pr, pi, pr, pi, COMPLEX)
        pr = qr
        pi = qi
    out = b * M * M + rows * M + cols
    mask = (rows < M) & (cols < M)
    if COMPLEX:
        tl.store(out_ptr + 2 * out, pr, mask=mask)
        tl.store(out_ptr + 2 * out + 1, pi, mask=mask)
    else:
        tl.store(out_ptr + out, pr, mask=mask)


@triton.jit
def _totals_kernel(
    a_ptr, x_ptr, total_ptr, offset_ptr, N, M, C, stride_ab, stride_at,
    T: tl.constexpr, VARYING: tl.constexpr, TRANSPOSE: tl.constexpr,
    COMPLEX: tl.constexpr, BM: tl.constexpr,
):  # fmt: skip
    """Chunk c's map y_end = y_start P + o: o, and with time-varying A, P.

    With time-invariant A, P is a power of the one matrix, which the
    caller computes once, so only o is carried.
    """
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, N - c * T)
    ACC: tl.constexpr = offset_ptr.dtype.element_ty
    base = b * stride_ab
    states = tl.arange(0, BM)
    rows = states[:, None]
    cols = states[None, :]
    n = c * T
    x_base = b * N * M
    o_r, o_i = _vector(x_ptr, x_base + n * M, states, M, ACC, COMPLEX)
    mr, mi = _matrix(a_ptr, base, n, stride_at, M, rows, cols, ACC, TRANSPOSE, COMPLEX)
    pr = mr
    pi = mi
    for s in range(1, T):
        valid = s < steps
        n = c * T + tl.where(valid, s, 0)
        if VARYING:
            mr, mi = _matrix(a_ptr, base, n, stride_at, M, rows, cols, ACC, TRANSPOSE, COMPLEX)
            qr, qi = _matmul(pr, pi, mr, mi, COMPLEX)
            pr = tl.where(valid, qr, pr)
            if COMPLEX:
                pi = tl.where(valid, qi, pi)
        xr, xi = _vector(x_ptr, x_base + n * M, states, M, ACC, COMPLEX)
        nr, ni = _times(o_r, o_i, mr, mi, COMPLEX)
        o_r = tl.where(valid, nr + xr, o_r)
        if COMPLEX:
            o_i = tl.where(valid, ni + xi, o_i)
    _store_vector(offset_ptr, pid.to(tl.int64) * M, states, M, o_r, o_i, COMPLEX)
    if VARYING:
        out = pid.to(tl.int64) * M * M + rows * M + cols
        mask = (rows < M) & (cols < M)
        if COMPLEX:
            tl.store(total_ptr + 2 * out, pr, mask=mask)
            tl.store(total_ptr + 2 * out + 1, pi, mask=mask)
        else:
            tl.store(total_ptr + out, pr, mask=mask)


@triton.jit
def _sweep_kernel(
    start_ptr, a_ptr, x_ptr, out_ptr, N, M, C, chunk, stride_ab, stride_at,
    VARYING: tl.constexpr, TRANSPOSE: tl.constexpr, COMPLEX: tl.constexpr,
    BM: tl.constexpr,
):  # fmt: skip
    """y[t] = y[t - 1] A[t]^T + x[t] through chunk c of ``chunk`` steps, from its start."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(chunk, N - c * chunk)
    ACC: tl.constexpr = start_ptr.dtype.element_ty
    base = b * stride_ab
    states = tl.arange(0, BM)
    rows = states[:, None]
    cols = states[None, :]
    yr, yi = _vector(start_ptr, pid.to(tl.int64) * M, states, M, ACC, COMPLEX)
    x_base = b * N * M
    # Time-invariant A is loaded once.
    mr, mi = _matrix(a_ptr, base, 0, 0, M, rows, cols, ACC, TRANSPOSE, COMPLEX)
    for s in range(0, steps):
        n = c * chunk + s
        if VARYING:
            mr, mi = _matrix(a_ptr, base, n, stride_at, M, rows, cols, ACC, TRANSPOSE, COMPLEX)
        xr, xi = _vector(x_ptr, x_base + n * M, states, M, ACC, COMPLEX)
        nr, ni = _times(yr, yi, mr, mi, COMPLEX)
        yr = nr + xr
        if COMPLEX:
            yi = ni + xi
        _store_vector(out_ptr, x_base + n * M, states, M, yr, yi, COMPLEX)


def _real(t: Tensor) -> Tensor:
    return torch.view_as_real(t) if t.is_complex() else t


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float64, torch.complex128):
        return dtype
    return torch.complex64 if dtype.is_complex else torch.float32


def _padded_states(M: int) -> int:
    return max(triton.next_power_of_2(M), 2)


def _num_warps(BM: int) -> int:
    """Measured on an RTX 5060 Ti: one warp up to 16 states, more costs more."""
    return 1 if BM <= 16 else 2 if BM <= 64 else 4


def _sweep_only(B: int, BM: int, complex_: bool, varying: bool) -> bool:
    """Whether one chunk per batch item beats the scan, a complex state counting twice.

    Measured on an RTX 5060 Ti: once the batch fills the GPU, the scan's
    extra work dominates, at B M = 2048 for time-varying A, whose scan
    multiplies matrices, and 8192 for time-invariant A, whose doesn't. Above
    ``_SCAN_STATES``, a time-varying scan doesn't fit in registers.
    """
    size = B * BM * (2 if complex_ else 1)
    if varying:
        return size >= 2048 or BM * (2 if complex_ else 1) > _SCAN_STATES
    return size >= 8192


def _power(A: Tensor, n: int, transpose: bool, acc: torch.dtype) -> Tensor:
    """(A^T)^n with ``transpose``, else A^n, of each of A's (k, M, M) matrices, n a power of 2."""
    k, M, _ = A.shape
    BM = _padded_states(M)
    if BM * (2 if A.is_complex() else 1) > _SCAN_STATES:
        # Too large for the products in registers: squarings in PyTorch, once per call.
        power = (A.mT if transpose else A).to(acc)
        for _ in range(n.bit_length() - 1):
            power = power @ power
        return power.contiguous()
    out = A.new_empty(k, M, M, dtype=acc)
    _power_kernel[(k,)](
        _real(A.contiguous()), _real(out), M, SQUARINGS=n.bit_length() - 1, TRANSPOSE=transpose,
        COMPLEX=A.is_complex(), BM=BM, num_warps=_num_warps(BM),
    )  # fmt: skip
    return out


def _chain(A: Tensor, stride_ab: int, stride_at: int, zi: Tensor, x: Tensor, transpose: bool,
           chunk: int) -> Tensor:  # fmt: skip
    """The row-vector recurrence y[t] = y[t - 1] A[t] + x[t] (A[t]^T with ``transpose``).

    x is a contiguous (B, N, M), zi a (B, M); A[t] of batch item b starts at
    ``b * stride_ab + t * stride_at`` in A's contiguous storage of M x M
    matrices, with 0 strides for what is shared, so that time-invariant A
    holds one matrix or one per batch item.
    """
    B, N, M = x.shape
    out = torch.empty_like(x)
    acc = _accumulation_dtype(x.dtype)
    BM = _padded_states(M)
    flags = dict(
        VARYING=stride_at != 0, TRANSPOSE=transpose, COMPLEX=x.is_complex(), BM=BM,
        num_warps=_num_warps(BM),
    )  # fmt: skip
    C = triton.cdiv(N, chunk)
    if C == 1:
        starts = zi.to(acc).contiguous()
    else:
        offsets = x.new_empty(B, C, M, dtype=acc)
        if stride_at == 0:
            # Each chunk's map is the same power of the matrix, one per batch
            # item or one for all: A holds just those.
            totals = _power(A.reshape(-1, M, M), chunk, transpose, acc)
            totals_strides = (stride_ab, 0)
        else:
            totals = x.new_empty(B, C, M, M, dtype=acc)
            totals_strides = (C * M * M, M * M)
        _totals_kernel[(B * C,)](
            _real(A), _real(x), _real(totals), _real(offsets), N, M, C, stride_ab, stride_at,
            T=chunk, **flags,
        )  # fmt: skip
        # The chunks' maps are in row-vector form already: their own chain.
        ends = _chain(totals, *totals_strides, zi.to(acc), offsets, False,
                      _CHUNK if C > _CHUNK else C)  # fmt: skip
        starts = torch.cat([zi.to(acc).unsqueeze(1), ends[:, :-1]], 1).contiguous()
    _sweep_kernel[(B * C,)](
        _real(starts), _real(A), _real(x), _real(out), N, M, C, chunk, stride_ab, stride_at,
        **flags,
    )  # fmt: skip
    return out


def fits(x: Tensor, M: int) -> bool:
    """Whether the kernels take state size M in x's dtype."""
    return _padded_states(M) * (2 if x.is_complex() else 1) <= _SWEEP_STATES


def recurrence(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    """h[t] = A[t] h[t - 1] + x[t]: A (B or 1, T or 1, M, M), zi (B, M), x (B, T, M)."""
    B, T, M = x.shape
    Ba, Ta = A.shape[:2]
    stride_ab = Ta * M * M if Ba > 1 else 0
    stride_at = M * M if Ta > 1 else 0
    chunk = T if _sweep_only(B, _padded_states(M), x.is_complex(), Ta > 1) else _CHUNK
    return _chain(A.contiguous(), stride_ab, stride_at, zi, x.contiguous(), True, chunk)
