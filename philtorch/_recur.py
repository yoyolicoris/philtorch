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
* CUDA: CUB scans for M <= 2 (``lti_recur``, ``lti_recur2``, ``scan``, and
  ``recur2`` for real time-varying M = 2), the ParaRNN kernels for real
  time-varying M = 2 and 3 within their limits, and the Triton kernels of
  :mod:`philtorch._recur_triton` for the rest, which they beat;
* MPS: the Metal ``lti_recur`` for time-invariant M = 1 in float32.

Anything else raises: there is no fallback to PyTorch ops.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from . import _recur_triton
except ImportError:  # pragma: no cover - Triton comes with PyTorch's CUDA builds for Linux
    _recur_triton = None


@torch.library.custom_op("philtorch::recur_triton", mutates_args=())
def recur_triton(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    """The recurrence by the Triton kernels: A (B or 1, T or 1, M, M), zi (B, M), x (B, T, M)."""
    return _recur_triton.recurrence(A, zi, x)


@recur_triton.register_fake
def _(A, zi, x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _triton_applies(x: Tensor, M: int) -> bool:
    return _recur_triton is not None and x.is_cuda and _recur_triton.fits(x, M)


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
        if not time_invariant and M == 2 and not x.is_complex():
            # Complex inputs run faster in the Triton kernels.
            return ops.recur2(A_lpv.contiguous(), zi, x.contiguous())
        if _triton_applies(x, M):
            return recur_triton(A, zi, x)
    if device == "mps" and time_invariant and M == 1 and x.dtype == torch.float32:
        return ops.lti_recur(A_lti[..., 0, 0], zi[:, 0], x[..., 0]).unsqueeze(-1)
    kind = "time-invariant" if time_invariant else "time-varying"
    reason = ""
    if device == "cuda":
        if _recur_triton is None:
            reason = ": Triton is not installed"
        else:
            most = _recur_triton.MAX_STATES
            reason = f"; the Triton kernels take M up to {most} ({most // 2} complex)"
    raise NotImplementedError(
        f"no recurrence kernel for {kind} A with M = {M} in {x.dtype} on {device}{reason}"
    )


def _sum_outer(lam: Tensor, h_prev: Tensor, Ba: int, Ta: int) -> Tensor:
    """sum of lam[b, t] h_prev[b, t]^H over the batch items and steps that share A[b, t]."""
    h_prev = h_prev.conj()
    if Ba == 1 and Ta == 1:
        return (lam.flatten(0, 1).mT @ h_prev.flatten(0, 1))[None, None]
    if Ta == 1:
        return (lam.mT @ h_prev).unsqueeze(1)
    if Ba == 1:
        return (lam.permute(1, 2, 0) @ h_prev.transpose(0, 1)).unsqueeze(0)
    return lam.unsqueeze(-1) * h_prev.unsqueeze(-2)


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
        A_H = A.mH
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
