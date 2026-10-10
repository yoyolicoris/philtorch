r"""The linear recurrence h[t] = A[t] h[t - 1] + x[t], one operator over every kernel.

:func:`recurrence` computes, for a batch of inputs x of shape (B, T, M) and
initial states zi of shape (B, M),

    h[t] = A[t] h[t - 1] + x[t],   t = 0, ..., T - 1,   h[-1] = zi,

with A of shape (B or 1, T or 1, M, M): shared over the batch, or over time
(time-invariant), or both. :class:`Recurrence` defines its derivatives once,
whatever kernel runs: the gradient is the same recurrence backward in time
over A[t]^H, and the tangent the same recurrence forward, so both are
differentiable again, to any order.

The forward pass picks a kernel from the input (:func:`_forward`):

* CPU: the C++ kernels, ``lti_recur`` (M = 1) and ``lti_recurN`` for
  time-invariant A, ``recurN`` otherwise;
* CUDA: CUB scans for M <= 2 (``lti_recur``, ``lti_recur2``, ``scan``,
  ``recur2``), the ParaRNN kernels for real time-varying M = 2 and 3 within
  their limits, and the Triton kernels below for the rest;
* MPS: the Metal ``lti_recur`` for time-invariant M = 1 in float32.

Anything else raises: there is no fallback to PyTorch ops.

The Triton kernels compute the recurrence as a chunked scan in row-vector
form, y[t] = y[t - 1] A[t]^T + x[t], in two levels, as the HMM's message
chains do (:mod:`philtorch.estimation._hmm_kernels`):

1. ``_totals_kernel`` composes each chunk of ``_CHUNK`` steps into one map
   y_end = y_start P + o, one program per chunk;
2. the chain of those maps gives each chunk's starting state, by the same
   method one level up;
3. ``_sweep_kernel`` carries each chunk's starting state through its steps.

With time-invariant A, every chunk's P is the same power of A^T, computed
once, so the totals only carry the offsets o: O(M^2) work per step, as the
sweep. Time-varying A costs O(M^3) per step for the totals. With many
batch items the batch alone fills the GPU, and one chunk per item, the
sweep alone, does less work (:func:`_sweep_only`).

Complex inputs run as their real and imaginary parts (``view_as_real``),
multiplied out: four real products per complex one, as complex arithmetic
takes anyway. Half and bfloat16 inputs accumulate in float32.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton comes with PyTorch's CUDA builds for Linux
    triton = None

# Steps per chunk of the Triton scan.
_CHUNK = 64
# The largest padded state sizes, a complex state counting twice: of a
# time-varying scan, which holds M^3 products in registers, and of an M x M
# matrix in registers, for the sweep and the time-invariant scan. Between
# the two, a time-varying recurrence runs the sweep alone; above, the loop.
_SCAN_STATES = 32
_SWEEP_STATES = 128

if triton is not None:

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
        # Too large for the products in registers; rare, and once per call.
        A = A.to(acc)
        return torch.linalg.matrix_power(A.mT if transpose else A, n).contiguous()
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


@torch.library.custom_op("philtorch::recur_triton", mutates_args=())
def recur_triton(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    """The recurrence by the Triton kernels: A (B or 1, T or 1, M, M), zi (B, M), x (B, T, M)."""
    B, T, M = x.shape
    Ba, Ta = A.shape[:2]
    A = A.contiguous()
    stride_ab = Ta * M * M if Ba > 1 else 0
    stride_at = M * M if Ta > 1 else 0
    chunk = T if _sweep_only(B, _padded_states(M), x.is_complex(), Ta > 1) else _CHUNK
    return _chain(A, stride_ab, stride_at, zi, x.contiguous(), True, chunk)


@recur_triton.register_fake
def _(A, zi, x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _triton_applies(x: Tensor, M: int) -> bool:
    if triton is None or not x.is_cuda:
        return False
    return _padded_states(M) * (2 if x.is_complex() else 1) <= _SWEEP_STATES


# Launch limits of the vendored ParaRNN kernels (third_party/pararnn/csrc):
# the batch runs along grid dimension y, which CUDA caps at 65535, and a
# sequence is split into blocks of 1024 threads that each take `chunk`
# steps, combined by one block of at most 1024 threads. The chunk sizes are
# dtype2chunkSizeBlockDiag in helpers.h: 1 for 3 x 3 float64, 2 otherwise.
_PARARNN_MAX_BATCH = 65535
# The dtypes of the ``scan`` kernel, the vendored torchlpc's.
_SCAN_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)


def _pararnn_applies(x: Tensor, M: int) -> bool:
    chunk = 1 if M == 3 and x.dtype == torch.float64 else 2
    return (
        x.is_cuda
        and x.dtype in (torch.float32, torch.float64)
        and M in (2, 3)
        and x.size(0) <= _PARARNN_MAX_BATCH
        and x.size(1) + 1 <= 1024 * 1024 * chunk
    )


def _forward(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    """The recurrence by the fastest kernel for the input; see the module docstring."""
    B, T, M = x.shape
    Ba, Ta = A.shape[:2]
    if T == 0 or B == 0:
        return torch.empty_like(x)
    ops = torch.ops.philtorch
    time_invariant = Ta == 1
    # The native kernels' layouts: one matrix (M, M) or one per batch item (B, M, M)
    # for time-invariant A; (T, M, M) shared over the batch or (B, T, M, M) otherwise.
    A_lti = A[0, 0] if Ba == 1 else A[:, 0]
    A_lpv = A[0] if Ba == 1 else A
    device = x.device.type
    if device == "cpu":
        if time_invariant:
            if M == 1:
                return ops.lti_recur(A_lti[..., 0, 0], zi[:, 0], x[..., 0]).unsqueeze(-1)
            return ops.lti_recurN(A_lti.contiguous(), zi, x.contiguous())
        return ops.recurN(A_lpv.contiguous(), zi, x.contiguous())
    if device == "cuda":
        if time_invariant and M == 1:
            return ops.lti_recur(A_lti[..., 0, 0], zi[:, 0], x[..., 0]).unsqueeze(-1)
        if time_invariant and M == 2:
            return ops.lti_recur2(A_lti.contiguous(), zi, x.contiguous())
        if not time_invariant and M == 1 and x.dtype in _SCAN_DTYPES:
            decay = A[..., 0, 0].expand(B, T)
            return ops.scan(x[..., 0].contiguous(), decay.contiguous(), zi[:, 0]).unsqueeze(-1)
        if not time_invariant and _pararnn_applies(x, M):
            jac = F.pad(-A.expand(B, T, M, M), (0, 0, 0, 0, 1, 0))
            rhs = torch.cat([zi.unsqueeze(1), x], 1)
            reduce = (
                torch.ops.parallel_reduce_cuda.parallel_reduce_block_diag_3x3_cuda
                if M == 3
                else torch.ops.parallel_reduce_cuda.parallel_reduce_block_diag_2x2_cuda
            )
            return reduce(jac.contiguous(), rhs)[:, 1:]
        if not time_invariant and M == 2:
            return ops.recur2(A_lpv.contiguous(), zi, x.contiguous())
        if _triton_applies(x, M):
            return recur_triton(A, zi, x)
    if device == "mps" and time_invariant and M == 1 and x.dtype == torch.float32:
        return ops.lti_recur(A_lti[..., 0, 0], zi[:, 0], x[..., 0]).unsqueeze(-1)
    kind = "time-invariant" if time_invariant else "time-varying"
    reason = ""
    if device == "cuda":
        reason = (
            ": Triton is not installed"
            if triton is None
            else f"; the Triton kernels take M up to {_SWEEP_STATES} ({_SWEEP_STATES // 2} complex)"
        )
    raise NotImplementedError(
        f"no recurrence kernel for {kind} A with M = {M} in {x.dtype} on {device}{reason}"
    )


def _sum_outer(lam: Tensor, h_prev: Tensor, Ba: int, Ta: int) -> Tensor:
    """sum of lam[b, t] h_prev[b, t]^H over the batch items and steps that share A[b, t]."""
    h_prev = h_prev.conj()
    if Ba == 1 and Ta == 1:
        return torch.einsum("btm,btn->mn", lam, h_prev)[None, None]
    if Ta == 1:
        return torch.einsum("btm,btn->bmn", lam, h_prev)[:, None]
    if Ba == 1:
        return torch.einsum("btm,btn->tmn", lam, h_prev)[None]
    return torch.einsum("btm,btn->btmn", lam, h_prev)


class Recurrence(torch.autograd.Function):
    """h[t] = A[t] h[t - 1] + x[t], differentiable to any order in both modes.

    With g the gradient of h, the adjoints lam[t] = g[t] + A[t + 1]^H lam[t + 1]
    are the same recurrence backward in time, so x's gradient is lam, zi's
    A[0]^H lam[0], and A[t]'s lam[t] h[t - 1]^H, summed over what shares it.
    """

    @staticmethod
    def forward(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
        return _forward(A, zi, x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, zi, _ = inputs
        ctx.save_for_backward(A, zi, output)
        ctx.save_for_forward(A, zi, output)

    @staticmethod
    def backward(ctx, grad):
        A, zi, h = ctx.saved_tensors
        Ba, Ta = A.shape[:2]
        A_H = A.mT.conj()
        # Step s of the reversed recurrence takes A[T - s]^H; the first step's
        # matrix multiplies a zero state, so any will do.
        reversed_A = A_H if Ta == 1 else A_H.roll(-1, 1).flip(1)
        lam = Recurrence.apply(reversed_A, torch.zeros_like(zi), grad.flip(1)).flip(1)
        grad_A = grad_zi = None
        if ctx.needs_input_grad[0]:
            h_prev = torch.cat([zi.unsqueeze(1), h[:, :-1]], 1)
            grad_A = _sum_outer(lam, h_prev, Ba, Ta)
        if ctx.needs_input_grad[1]:
            grad_zi = (A_H[:, 0] @ lam[:, 0].unsqueeze(-1)).squeeze(-1)
        return grad_A, grad_zi, lam

    @staticmethod
    def jvp(ctx, dA, dzi, dx):
        A, zi, h = ctx.saved_tensors
        tangent = dx if dx is not None else torch.zeros_like(h)
        if dA is not None:
            h_prev = torch.cat([zi.unsqueeze(1), h[:, :-1]], 1)
            tangent = tangent + (dA @ h_prev.unsqueeze(-1)).squeeze(-1)
        return Recurrence.apply(A, dzi if dzi is not None else torch.zeros_like(zi), tangent)


def recurrence(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    """h[t] = A[t] h[t - 1] + x[t], t = 0, ..., T - 1, from h[-1] = zi.

    Args:
        A: (B or 1, T or 1, M, M): per batch item or shared, per step or
            time-invariant.
        zi: the initial states, (B, M).
        x: the inputs, (B, T, M).

    Returns:
        The states h, (B, T, M).
    """
    B, T, M = x.shape
    if A.dim() != 4 or A.shape[0] not in (1, B) or A.shape[1] not in (1, T) or A.shape[2:] != (
        M, M
    ):  # fmt: skip
        raise ValueError(
            f"A must be (1 or {B}, 1 or {T}, {M}, {M}) for x of shape {tuple(x.shape)}, "
            f"got {tuple(A.shape)}"
        )
    if zi.shape != (B, M):
        raise ValueError(f"zi must be ({B}, {M}), got {tuple(zi.shape)}")
    if not A.dtype == zi.dtype == x.dtype or not A.device == zi.device == x.device:
        raise ValueError(
            "A, zi and x must share a dtype and a device, got "
            f"{[(t.dtype, str(t.device)) for t in (A, zi, x)]}"
        )
    return Recurrence.apply(A, zi, x)
