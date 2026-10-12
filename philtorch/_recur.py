r"""The linear recurrence h[t] = A[t] h[t - 1] + x[t], one operator over every kernel.

:func:`recurrence` computes, for a batch of inputs x of shape (B, T, M) and
initial states zi of shape (B, M),

    h[t] = A[t] h[t - 1] + x[t],   t = 0, ..., T - 1,   h[-1] = zi,

with A of shape (B or 1, T or 1, M, M): shared over the batch, or over time
(time-invariant), or both. :class:`Recurrence` defines its derivatives once,
whatever kernel runs: the gradient is the same recurrence backward in time
over A[t]^H, and the tangent the same recurrence forward, so both are
differentiable again, to any order. Dynamo can't trace an autograd.Function
with a jvp, so compiled code calls the kernels' ops directly instead. They
have gradients registered, so it compiles to one graph, with reverse mode but
no forward mode.

The forward pass runs the first kernel of ``_KERNELS`` that takes the input:

* CPU: the C++ kernels, ``lti_recur`` (M = 1) and ``lti_recurN`` for
  time-invariant A, ``scan`` (M = 1) and ``recurN`` otherwise;
* CUDA: CUB scans for M <= 2 (``lti_recur``, ``lti_recur2``, ``scan``, and
  ``recur2`` for real time-varying M = 2), the ParaRNN kernels for real
  time-varying M = 2 and 3 within their limits, and the Triton kernels of
  :mod:`philtorch._recur_kernels` for the rest, which they beat;
* MPS: the Metal ``lti_recur`` for time-invariant M = 1 in float32.

Every kernel but ``scan``, ParaRNN and the Metal ``lti_recur`` takes int64
too, so integer recurrences such as the Fibonacci numbers run exactly. Anything else
raises: there is no fallback to PyTorch ops.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ._ssm import to_lpv, to_lti

try:
    from . import _recur_kernels
except ImportError:  # pragma: no cover - Triton comes with PyTorch's CUDA builds for Linux
    _recur_kernels = None


@torch.library.custom_op("philtorch::recur_triton", mutates_args=())
def recur_triton(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    """The recurrence by the Triton kernels: A (B or 1, T or 1, M, M), zi (B, M), x (B, T, M)."""
    return _recur_kernels.recurrence(A, zi, x)


@recur_triton.register_fake
def _(A, zi, x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


# Each kernel is a condition on the input, (A, x), and a launch, (A, zi, x),
# listed in _KERNELS in order of preference. The native kernels take A as one
# matrix (M, M) or one per batch item (B, M, M) when it is time-invariant,
# and as (T, M, M) shared over the batch or (B, T, M, M) otherwise.


def _time_invariant(A: Tensor) -> bool:
    return A.size(1) == 1


def _floating_or_int64(x: Tensor) -> bool:
    """The dtypes of the kernels of order M: integers only as int64."""
    return x.is_floating_point() or x.is_complex() or x.dtype == torch.int64


def _lti_recur_applies(A: Tensor, x: Tensor) -> bool:
    device = x.device.type
    return (
        _time_invariant(A)
        and x.size(-1) == 1
        and (
            device in ("cpu", "cuda")
            and _floating_or_int64(x)
            or device == "mps"
            and x.dtype == torch.float32
        )
    )


def _lti_recur(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    A = to_lti(A)[..., 0, 0]
    return torch.ops.philtorch.lti_recur(A, zi[:, 0], x[..., 0]).unsqueeze(-1)


def _lti_recurN_applies(A: Tensor, x: Tensor) -> bool:
    return x.is_cpu and _time_invariant(A) and _floating_or_int64(x)


def _lti_recurN(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    return torch.ops.philtorch.lti_recurN(to_lti(A).contiguous(), zi, x.contiguous())


def _recurN_applies(A: Tensor, x: Tensor) -> bool:
    return x.is_cpu and not _time_invariant(A) and _floating_or_int64(x)


def _recurN(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    return torch.ops.philtorch.recurN(to_lpv(A).contiguous(), zi, x.contiguous())


def _lti_recur2_applies(A: Tensor, x: Tensor) -> bool:
    return x.is_cuda and _time_invariant(A) and x.size(-1) == 2 and _floating_or_int64(x)


def _lti_recur2(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    return torch.ops.philtorch.lti_recur2(to_lti(A).contiguous(), zi, x.contiguous())


# The dtypes of the ``scan`` kernel, the vendored torchlpc's.
_SCAN_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)


def _scan_applies(A: Tensor, x: Tensor) -> bool:
    # On CPU too: 3-7 times faster than recurN at M = 1.
    return (
        x.device.type in ("cpu", "cuda")
        and not _time_invariant(A)
        and x.size(-1) == 1
        and x.dtype in _SCAN_DTYPES
    )


def _scan(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    B, T, _ = x.shape
    decay = A[..., 0, 0].expand(B, T).contiguous()
    return torch.ops.philtorch.scan(x[..., 0].contiguous(), decay, zi[:, 0]).unsqueeze(-1)


# Launch limits of the vendored ParaRNN kernels (third_party/pararnn/csrc):
# the batch runs along grid dimension y, which CUDA caps at 65535, and a
# sequence is split into blocks of 1024 threads that each take `chunk`
# steps, combined by one block of at most 1024 threads. The chunk sizes are
# dtype2chunkSizeBlockDiag in helpers.h: 1 for 3 x 3 float64, 2 otherwise.
_PARARNN_MAX_BATCH = 65535


def _pararnn_applies(A: Tensor, x: Tensor) -> bool:
    # ParaRNN takes half and bfloat16 too, but accumulates in them; the
    # Triton kernels accumulate those in float32.
    M = x.size(-1)
    chunk = 1 if M == 3 and x.dtype == torch.float64 else 2
    return (
        x.is_cuda
        and not _time_invariant(A)
        and x.dtype in (torch.float32, torch.float64)
        and M in (2, 3)
        and x.size(0) <= _PARARNN_MAX_BATCH
        and x.size(1) + 1 <= 1024 * 1024 * chunk
    )


def _pararnn(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    B, T, M = x.shape
    jac = F.pad(-A.expand(B, T, M, M), (0, 0, 0, 0, 1, 0)).contiguous()
    rhs = torch.cat([zi.unsqueeze(1), x], 1)
    ops = torch.ops.parallel_reduce_cuda
    reduce = (
        ops.parallel_reduce_block_diag_3x3_cuda
        if M == 3
        else ops.parallel_reduce_block_diag_2x2_cuda
    )
    return reduce(jac, rhs)[:, 1:]


def _recur2_applies(A: Tensor, x: Tensor) -> bool:
    # Complex inputs run faster in the Triton kernels.
    return (
        x.is_cuda
        and not _time_invariant(A)
        and x.size(-1) == 2
        and not x.is_complex()
        and _floating_or_int64(x)
    )


def _recur2(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    return torch.ops.philtorch.recur2(to_lpv(A).contiguous(), zi, x.contiguous())


def _triton_applies(A: Tensor, x: Tensor) -> bool:
    return _recur_kernels is not None and x.is_cuda and _recur_kernels.fits(x, x.size(-1))


_KERNELS = (
    (_lti_recur_applies, _lti_recur),
    (_lti_recurN_applies, _lti_recurN),
    (_scan_applies, _scan),
    (_recurN_applies, _recurN),
    (_lti_recur2_applies, _lti_recur2),
    (_pararnn_applies, _pararnn),
    (_recur2_applies, _recur2),
    (_triton_applies, recur_triton),
)


def _forward(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    """The recurrence by the first kernel of _KERNELS that takes the input."""
    if x.size(0) == 0 or x.size(1) == 0:
        # Built from empty slices of every input, rather than as a new tensor,
        # so that compiled code, which runs no Recurrence, keeps them in the
        # graph and gives them zero gradients.
        return x + zi[:, :0].sum(-1)[:, None, None] + A[..., :0].sum()
    for applies, run in _KERNELS:
        if applies(A, x):
            return run(A, zi, x)
    kind = "time-invariant" if _time_invariant(A) else "time-varying"
    reason = ""
    if x.is_cuda:
        if _recur_kernels is None:
            reason = ": Triton is not installed"
        elif not _floating_or_int64(x):
            reason = "; the Triton kernels take floating-point, complex and int64 dtypes"
        else:
            most = _recur_kernels.MAX_STATES
            reason = f"; the Triton kernels take M up to {most} ({most // 2} complex)"
    raise NotImplementedError(
        f"no recurrence kernel for {kind} A with M = {x.size(-1)} in {x.dtype} "
        f"on {x.device.type}{reason}"
    )


# Rows per block of the long contractions in _sum_outer. As one matmul over
# B * T rows, cuBLAS takes its slow large-K path; blocks summed after ran
# 2-4 times faster, and add up in a shallower tree.
_CONTRACT_BLOCK = 1024


def _contract(lam: Tensor, h: Tensor) -> Tensor:
    """sum over k of lam[p, k] h[p, k]^T: (P, K, M) twice, to (P, M, M)."""
    P, K, M = lam.shape
    if K <= _CONTRACT_BLOCK:
        return lam.mT @ h
    blocks = -(-K // _CONTRACT_BLOCK)
    pad = blocks * _CONTRACT_BLOCK - K
    if pad:
        lam, h = F.pad(lam, (0, 0, 0, pad)), F.pad(h, (0, 0, 0, pad))
    lam, h = (t.reshape(P * blocks, _CONTRACT_BLOCK, M) for t in (lam, h))
    return (lam.mT @ h).reshape(P, blocks, M, M).sum(1)


def _sum_outer(lam: Tensor, h_prev_conj: Tensor, Ba: int, Ta: int) -> Tensor:
    """sum of lam[b, t] h_prev[b, t]^H over the batch items and steps that share A[b, t],
    from h_prev's conjugate."""
    h = h_prev_conj
    if lam.size(-1) == 1:
        # Scalar states: the outer products are products, summed elementwise.
        # As matmuls of 1 x 1 matrices, they cost more to dispatch than to run.
        products = (lam * h).unsqueeze(-1)
        shared = [dim for dim, size in ((0, Ba), (1, Ta)) if size == 1]
        return products.sum(shared, keepdim=True) if shared else products
    if Ba == 1 and Ta == 1:
        return _contract(lam.flatten(0, 1)[None], h.flatten(0, 1)[None])[None]
    if Ta == 1:
        return _contract(lam, h).unsqueeze(1)
    if Ba == 1:
        return (lam.permute(1, 2, 0) @ h.transpose(0, 1)).unsqueeze(0)
    return lam.unsqueeze(-1) * h.unsqueeze(-2)


def _previous(zi: Tensor, h: Tensor) -> Tensor:
    """The states each step starts from, h[t - 1]: zi, then h but its last.

    The concatenation copies, so conjugate views come out resolved.
    """
    return torch.cat([zi.unsqueeze(1), h[:, :-1]], 1)


def _gradients(run, A, zi, h, grad, needs_A, needs_zi):
    """The gradients of A, zi and x from h's, ``grad``; ``run`` runs the recurrence.

    With g the gradient of h, the adjoints lam[t] = g[t] + A[t + 1]^H lam[t + 1]
    are the same recurrence backward in time, so x's gradient is lam, zi's
    A[0]^H lam[0], and A[t]'s lam[t] h[t - 1]^H, summed over what shares it.
    """
    Ba, Ta = A.shape[:2]
    if grad.size(1) == 0:
        return (
            torch.zeros_like(A) if needs_A else None,
            torch.zeros_like(zi) if needs_zi else None,
            grad,
        )
    A_H = A.mH
    if Ta > 1:
        # Step s of the reversed recurrence takes A[T - s]^H; the first
        # step's matrix multiplies a zero state, so A[0]^H will do. One
        # gather reverses them, already contiguous.
        A_H = A_H.index_select(1, torch.arange(Ta, 0, -1, device=A.device) % Ta)
    lam = run(A_H, torch.zeros_like(zi), grad.flip(1)).flip(1)
    grad_A = grad_zi = None
    if needs_A:
        grad_A = _sum_outer(lam, _previous(zi.conj(), h.conj()), Ba, Ta)
    if needs_zi:
        # A[0]^H lam[0], summed elementwise: as a batched matmul of small
        # matrices, PyTorch routes it through a Triton kernel whose launch
        # costs more than the product.
        grad_zi = (A[:, 0].conj() * lam[:, 0].unsqueeze(-1)).sum(-2)
    return grad_A, grad_zi, lam


class Recurrence(torch.autograd.Function):
    """h[t] = A[t] h[t - 1] + x[t], differentiable to any order in both modes, in eager.

    The gradients come from :func:`_gradients`, and the tangent is the same
    recurrence forward on the tangents' inputs. Both call this function
    again, so they are differentiable in turn. Under :func:`torch.func.vmap`,
    the vmapped dimension joins the batch.
    """

    @staticmethod
    def forward(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
        return _forward(A, zi, x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, zi, _ = inputs
        ctx.save_for_backward(A, zi, output)
        ctx.save_for_forward(A, zi, output)
        # Inputs without a tangent reach the jvp as None, not zeros, so that it
        # skips their terms, such as dA[t] h[t - 1] over every step; so does
        # an output without a gradient reach backward.
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad):
        if grad is None:  # no gradient reached h; see setup_context
            return None, None, None
        A, zi, h = ctx.saved_tensors
        return _gradients(Recurrence.apply, A, zi, h, grad, *ctx.needs_input_grad[:2])

    @staticmethod
    def vmap(info, in_dims, A, zi, x):
        # The kernels take no vmapped dimension: fold it into the batch, (V, B)
        # to V * B. A shared over both stays shared; otherwise each of the V * B
        # sequences gets its own matrices.
        V = info.batch_size
        A_dim, zi_dim, x_dim = in_dims

        def front(t, dim):
            return t.expand(V, *t.shape) if dim is None else t.movedim(dim, 0)

        zi, x = front(zi, zi_dim), front(x, x_dim)
        B = x.size(1)
        if A_dim is not None or A.size(0) > 1:
            A = front(A, A_dim)
            A = A.expand(V, B, *A.shape[2:]).reshape(V * B, *A.shape[2:])
        h = Recurrence.apply(A, zi.reshape(V * B, -1), x.reshape(V * B, *x.shape[2:]))
        return h.reshape(V, B, *h.shape[1:]), 0

    @staticmethod
    def jvp(ctx, dA, dzi, dx):
        A, zi, h = ctx.saved_tensors
        if h.size(1) == 0:
            return torch.zeros_like(h)
        # The tangent's inputs: dx, and dA[t] h[t - 1]; for time-invariant A
        # a matmul over each sequence's steps at once.
        tangent = dx
        if dA is not None:
            h_prev = _previous(zi, h)
            if dA.size(1) == 1:
                term = h_prev @ dA[:, 0].mT
            else:
                term = (dA @ h_prev.unsqueeze(-1)).squeeze(-1)
            tangent = term if tangent is None else tangent + term
        if tangent is None:
            tangent = torch.zeros_like(h)
        return Recurrence.apply(A, dzi if dzi is not None else torch.zeros_like(zi), tangent)


# Compiled code runs the kernels' ops directly (see recurrence), so the
# Triton op carries the same gradients; the native ops' are registered in
# philtorch/_ops.py.
def _recur_triton_setup_context(ctx, inputs, output):
    A, zi, _ = inputs
    ctx.save_for_backward(A, zi, output)


def _recur_triton_backward(ctx, grad):
    A, zi, h = ctx.saved_tensors
    return _gradients(_forward, A, zi, h, grad, *ctx.needs_input_grad[:2])


recur_triton.register_autograd(_recur_triton_backward, setup_context=_recur_triton_setup_context)


def recurrence(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    """h[t] = A[t] h[t - 1] + x[t], t = 0, ..., T - 1, from h[-1] = zi.

    Args:
        A: (B or 1, T or 1, M, M): per batch item or shared, per step or
            time-invariant.
        zi: the initial states, (B, M).
        x: the inputs, (B, T, M).

    Returns:
        The states h, (B, T, M).

    Raises:
        ValueError: if the shapes disagree, or the tensors' dtypes or devices.
        NotImplementedError: if no kernel takes the input; see the module
            docstring.
    """
    if x.dim() != 3:
        raise ValueError(f"x must be (B, T, M), got {tuple(x.shape)}")
    B, T, M = x.shape
    if (
        A.dim() != 4
        or A.shape[0] not in (1, B)
        or A.shape[1] not in (1, T)
        or A.shape[2:] != (M, M)
    ):
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
    if torch.compiler.is_compiling():
        # Dynamo can't trace an autograd.Function with a jvp: compiled code
        # runs the kernels' ops, which carry their own gradients, without
        # forward mode.
        return _forward(A, zi, x)
    return Recurrence.apply(A, zi, x)
