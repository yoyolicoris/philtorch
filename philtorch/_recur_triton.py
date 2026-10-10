"""Triton kernels for the linear recurrence h[t] = A[t] h[t - 1] + x[t].

:func:`recurrence` computes it on CUDA for any state size M up to
``MAX_STATES`` (half that for complex), in row-vector form, y[t] = y[t - 1]
A[t]^T + x[t], as one single-pass scan with decoupled look-back
(``_scan_kernel``). Each program claims the next chunk of a batch item from
an atomic counter, so every chunk it waits on belongs to a program already
running, and

1. composes its chunk into the map y_end = y_start P + o and publishes it;
2. looks back over its predecessors' flags to the latest that has published
   its end state, and from it applies the maps in between, y <- y P + o, to
   get its own start;
3. publishes its own end state, y_start P + o;
4. carries its start through its steps.

With time-invariant A, every full chunk's P is the same power of A^T,
computed once by repeated squaring (``_power_kernel``), so a chunk's map
costs O(M^2) per step, as the steps themselves; time-varying A costs O(M^3)
per step for P, in registers up to ``_SCAN_STATES``, and above a row at a
time against A[t]^T in registers, the running product alternating between
two scratch buffers. M is a compile-time constant: with a run-time M that
is not a multiple of 16, the loads of A lose their vectorization, up to
4.5 times slower. With many batch items the batch alone fills the GPU, and
one chunk per item, the steps alone, does less work (:func:`_chunk`).

Complex inputs run as their real and imaginary parts (``view_as_real``),
multiplied out: four real products per complex one, as complex arithmetic
takes anyway. Half and bfloat16 inputs accumulate in float32. Autograd and
the choice of kernel are :mod:`philtorch._recur`'s.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

# The largest padded state sizes, a complex state counting twice: of a
# time-varying chunk's map whose M^3 products fit in registers, and of an
# M x M matrix in registers, the kernels' limit. Between the two, the map's
# running product lives in scratch memory, a row at a time in registers.
_SCAN_STATES = 32
MAX_STATES = 128


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
def _store_matrix(ptr, offset, rows, cols, M, re, im, COMPLEX: tl.constexpr):
    dtype = ptr.dtype.element_ty
    mask = (rows < M) & (cols < M)
    out = offset + rows * M + cols
    if COMPLEX:
        tl.store(ptr + 2 * out, re.to(dtype), mask=mask)
        tl.store(ptr + 2 * out + 1, im.to(dtype), mask=mask)
    else:
        tl.store(ptr + out, re.to(dtype), mask=mask)


@triton.jit
def _load_tile(ptr, rows, cols, BM: tl.constexpr, COMPLEX: tl.constexpr):
    """Entries (rows, cols) of a padded BM x BM scratch matrix, its imaginary part after."""
    offset = rows * BM + cols
    re = tl.load(ptr + offset)
    if COMPLEX:
        return re, tl.load(ptr + BM * BM + offset)
    return re, re


@triton.jit
def _store_tile(ptr, rows, cols, BM: tl.constexpr, re, im, COMPLEX: tl.constexpr):
    offset = rows * BM + cols
    tl.store(ptr + offset, re)
    if COMPLEX:
        tl.store(ptr + BM * BM + offset, im)


@triton.jit
def _power_kernel(
    a_ptr, out_ptr, M: tl.constexpr, SQUARINGS: tl.constexpr, COMPLEX: tl.constexpr,
    BM: tl.constexpr,
):  # fmt: skip
    """Matrix b's transpose's 2^SQUARINGS-th power, one program per matrix."""
    b = tl.program_id(0).to(tl.int64)
    ACC: tl.constexpr = out_ptr.dtype.element_ty
    states = tl.arange(0, BM)
    rows = states[:, None]
    cols = states[None, :]
    pr, pi = _matrix(a_ptr, b * M * M, 0, 0, M, rows, cols, ACC, True, COMPLEX)
    for _ in tl.static_range(SQUARINGS):
        # P P needs P's rows and columns in the product's two roles.
        qr, qi = _matmul(pr, pi, pr, pi, COMPLEX)
        pr = qr
        pi = qi
    _store_matrix(out_ptr, b * M * M, rows, cols, M, pr, pi, COMPLEX)


@triton.jit
def _scan_kernel(
    a_ptr, x_ptr, zi_ptr, power_ptr, out_ptr, offset_ptr, map_ptr, state_ptr, flag_ptr,
    counter_ptr, scratch_ptr, N, C, chunk, stride_ab, stride_at, stride_pb, M: tl.constexpr,
    SCAN: tl.constexpr, VARYING: tl.constexpr, SCRATCH: tl.constexpr, COMPLEX: tl.constexpr,
    BM: tl.constexpr,
):  # fmt: skip
    """The steps of one chunk of ``chunk`` steps of one batch item; see the module docstring.

    Without ``SCAN``, a batch item is one chunk, and runs its steps alone.
    With ``SCRATCH``, a time-varying chunk's P alternates between two scratch
    buffers, computed a row at a time. A chunk's flag
    is 1 once its map (o, and P for time-varying A) is published, and 2 once
    its end state is.
    """
    tile = tl.atomic_add(counter_ptr, 1).to(tl.int64)
    b = tile // C
    c = tile % C
    steps = tl.minimum(chunk, N - c * chunk)
    ACC: tl.constexpr = state_ptr.dtype.element_ty
    base = b * stride_ab
    states = tl.arange(0, BM)
    rows = states[:, None]
    cols = states[None, :]
    x_base = b * N * M
    first = c * chunk
    index = b * C + c
    if not VARYING:
        # Time-invariant A, transposed into row-vector form, is loaded once;
        # time-varying A, where each step reads it, as loaded once outside its
        # loops, would need a conversion between layouts as large as itself.
        mr, mi = _matrix(a_ptr, base, 0, 0, M, rows, cols, ACC, True, COMPLEX)
    yr, yi = _vector(zi_ptr, b * M, states, M, ACC, COMPLEX)
    if SCAN:
        # Time-invariant A's chunks share P, for the look-back and the end
        # state; a time-varying chunk composes its own, in registers or in
        # scratch memory. A scratch chunk's P stays in its second buffer, the
        # last of an even number of steps, where its successors read it too.
        PLANE: tl.constexpr = (2 if COMPLEX else 1) * BM * BM
        scratch = scratch_ptr + index * 2 * PLANE
        if VARYING and not SCRATCH:
            pr, pi = _matrix(a_ptr, base, first, stride_at, M, rows, cols, ACC, True, COMPLEX)
        last = c == C - 1
        # 1. The chunk's map: o from a zero start, and P. The last chunk's is
        # never read, but composing it spaces out its look-back, which would
        # otherwise spin on predecessors still composing theirs: skipping it
        # made scans up to 15% slower.
        o_r, o_i = _vector(x_ptr, x_base + first * M, states, M, ACC, COMPLEX)
        if SCRATCH:
            # P starts as the first matrix, copied a row at a time: whole,
            # its transposed load and the store would need a conversion
            # between layouts as large as itself.
            for row in range(0, M):
                fr, fi = _matrix(a_ptr, base, first, stride_at, M, row, states, ACC, True, COMPLEX)
                _store_tile(scratch, row, states, BM, fr, fi, COMPLEX)
            tl.debug_barrier()
        for s in range(1, steps):
            n = first + s
            if VARYING:
                mr, mi = _matrix(a_ptr, base, n, stride_at, M, rows, cols, ACC, True, COMPLEX)
                if SCRATCH:
                    # P A^T one row of P at a time, against A^T in registers.
                    current = scratch + ((s - 1) % 2) * PLANE
                    following = scratch + (s % 2) * PLANE
                    for row in range(0, M):
                        ar, ai = _load_tile(current, row, states, BM, COMPLEX)
                        qr, qi = _times(ar, ai, mr, mi, COMPLEX)
                        _store_tile(following, row, states, BM, qr, qi, COMPLEX)
                    tl.debug_barrier()
                else:
                    pr, pi = _matmul(pr, pi, mr, mi, COMPLEX)
            xr, xi = _vector(x_ptr, x_base + n * M, states, M, ACC, COMPLEX)
            nr, ni = _times(o_r, o_i, mr, mi, COMPLEX)
            o_r = nr + xr
            if COMPLEX:
                o_i = ni + xi
        if c > 0 and not last:
            _store_vector(offset_ptr, index * M, states, M, o_r, o_i, COMPLEX)
            if VARYING and not SCRATCH:
                _store_matrix(map_ptr, index * M * M, rows, cols, M, pr, pi, COMPLEX)
            tl.debug_barrier()
            tl.atomic_xchg(flag_ptr + index, 1, sem="release")
        if not VARYING:
            pr, pi = _matrix(power_ptr, b * stride_pb, 0, 0, M, rows, cols, ACC, False, COMPLEX)
        if c > 0:
            # 2. Back over the flags to the latest predecessor with its end
            # state, then forward from it through the maps in between.
            j = c - 1
            flag = tl.atomic_add(flag_ptr + b * C + j, 0, sem="acquire")
            while flag != 2:
                j = tl.where(flag == 1, j - 1, j)
                flag = tl.atomic_add(flag_ptr + b * C + j, 0, sem="acquire")
            # Every thread reads what the acquire made visible.
            tl.debug_barrier()
            yr, yi = _vector(state_ptr, (b * C + j) * M, states, M, ACC, COMPLEX)
            for i in range(j + 1, c):
                ar, ai = _vector(offset_ptr, (b * C + i) * M, states, M, ACC, COMPLEX)
                if SCRATCH:
                    their = scratch_ptr + (b * C + i) * 2 * PLANE + PLANE
                    qr, qi = _load_tile(their, rows, cols, BM, COMPLEX)
                elif VARYING:
                    qr, qi = _matrix(
                        map_ptr, (b * C + i) * M * M, 0, 0, M, rows, cols, ACC, False, COMPLEX
                    )
                else:
                    qr = pr
                    qi = pi
                ur, ui = _times(yr, yi, qr, qi, COMPLEX)
                yr = ur + ar
                if COMPLEX:
                    yi = ui + ai
        if not last:
            # 3. The chunk's end state.
            if SCRATCH:
                own_r, own_i = _load_tile(scratch + PLANE, rows, cols, BM, COMPLEX)
                er, ei = _times(yr, yi, own_r, own_i, COMPLEX)
            else:
                er, ei = _times(yr, yi, pr, pi, COMPLEX)
            _store_vector(state_ptr, index * M, states, M, er + o_r, ei + o_i, COMPLEX)
            tl.debug_barrier()
            tl.atomic_xchg(flag_ptr + index, 2, sem="release")
    # 4. The chunk's steps.
    for s in range(0, steps):
        n = first + s
        if VARYING:
            mr, mi = _matrix(a_ptr, base, n, stride_at, M, rows, cols, ACC, True, COMPLEX)
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
    """Measured on an RTX 5060 Ti: one warp up to 32 states, complex too; more costs more."""
    return 1 if BM <= 32 else 2 if BM <= 64 else 4


def _chunk(B: int, T: int, BM: int, complex_: bool, varying: bool) -> int:
    """Steps per chunk: T, one chunk per batch item, once the batch alone fills the
    GPU and the scan's maps would only add work; else a power of 2, for the
    time-invariant maps' power. A complex state counts twice.

    Measured on an RTX 5060 Ti. One chunk from B M = 4096 for time-varying A,
    whose maps multiply matrices, and from B = 1024 or B M = 16384 for
    time-invariant A, whose don't; with the maps' products through scratch
    memory, from B M = 512, and always at M = 128, where they cost more than
    the parallelism gains. Otherwise longer chunks mean fewer look-backs,
    shorter ones more programs, and the matrices' size moves the balance.
    """
    states = BM * (2 if complex_ else 1)
    if varying:
        if states > _SCAN_STATES:
            single = BM > 64 or B * states > 512
        else:
            single = B * states >= 4096
        return T if single else 128 if states <= 16 else 64
    if B >= 1024 or B * states >= 16384:
        return T
    return 256 if states <= 16 else 128


def _power(A: Tensor, n: int, acc: torch.dtype) -> Tensor:
    """(A^T)^n of each of A's (k, M, M) matrices, n a power of 2, in dtype ``acc``."""
    k, M, _ = A.shape
    BM = _padded_states(M)
    if BM * (2 if A.is_complex() else 1) > _SCAN_STATES:
        # Too large for the products in registers: squarings in PyTorch, once per call.
        power = A.mT.to(acc)
        for _ in range(n.bit_length() - 1):
            power = power @ power
        return power.contiguous()
    out = A.new_empty(k, M, M, dtype=acc)
    _power_kernel[(k,)](
        _real(A.contiguous()), _real(out), M=M, SQUARINGS=n.bit_length() - 1,
        COMPLEX=A.is_complex(), BM=BM, num_warps=_num_warps(BM),
    )  # fmt: skip
    return out


def fits(x: Tensor, M: int) -> bool:
    """Whether the kernels take state size M in x's dtype."""
    return _padded_states(M) * (2 if x.is_complex() else 1) <= MAX_STATES


def recurrence(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    """h[t] = A[t] h[t - 1] + x[t]: A (B or 1, T or 1, M, M), zi (B, M), x (B, T, M)."""
    B, T, M = x.shape
    Ba, Ta = A.shape[:2]
    varying = Ta > 1
    complex_ = x.is_complex()
    acc = _accumulation_dtype(x.dtype)
    BM = _padded_states(M)
    A, zi, x = A.contiguous(), zi.contiguous(), x.contiguous()
    stride_ab = Ta * M * M if Ba > 1 else 0
    stride_at = M * M if varying else 0
    chunk = _chunk(B, T, BM, complex_, varying)
    C = triton.cdiv(T, chunk)
    out = torch.empty_like(x)
    # Each chunk's end state, map and flag; unused pointers take a placeholder.
    states = x.new_empty(B * C, M, dtype=acc)
    offsets = x.new_empty(B * C, M, dtype=acc) if C > 1 else states
    planes = 2 if complex_ else 1
    in_scratch = varying and BM * planes > _SCAN_STATES
    maps = x.new_empty(B * C, M, M, dtype=acc) if C > 1 and varying and not in_scratch else states
    power, stride_pb = states, 0
    if C > 1 and not varying:
        power = _power(A.reshape(-1, M, M), chunk, acc)
        stride_pb = M * M if Ba > 1 else 0
    # Time-varying maps too large for registers: two scratch buffers per
    # chunk, the second holding the chunk's P, as the chunks' even length
    # leaves it there.
    scratch = states
    if C > 1 and in_scratch:
        scratch = x.new_empty(B * C, 2 * planes * BM * BM, dtype=acc.to_real())
    # The flags, then the count of chunks claimed.
    flags = torch.zeros(B * C + 1, dtype=torch.int32, device=x.device)
    _scan_kernel[(B * C,)](
        _real(A), _real(x), _real(zi), _real(power), _real(out), _real(offsets), _real(maps),
        _real(states), flags, flags[B * C :], _real(scratch), T, C, chunk, stride_ab,
        stride_at, stride_pb, M=M, SCAN=C > 1, VARYING=varying, SCRATCH=in_scratch,
        COMPLEX=complex_, BM=BM, num_warps=_num_warps(BM),
    )  # fmt: skip
    return out
