"""Fake and autograd implementations of the native ops in :mod:`philtorch._C`.

The fakes give :func:`torch.compile` the ops' output shapes. The gradients
are what compiled code differentiates: there :mod:`philtorch._recur` calls the
ops directly, outside its autograd function. Every recurrence op maps its
layout to that module's, (B or 1, T or 1, M, M) matrices, whose ``_gradients``
defines the gradients once; torchlpc's ``lpc``, an all-pole filter, keeps its
own.
"""

import functools
from typing import Any

import torch
import torch.nn.functional as F

from . import _ssm


def _recurrence_gradients(A, zi, h, grad, needs_A, needs_zi):
    # Imported here, so that importing philtorch doesn't load Triton.
    from ._recur import _forward, _gradients

    return _gradients(_forward, A, zi, h, grad, needs_A, needs_zi)


def _setup_context(ctx: Any, inputs: list[Any], output: Any) -> Any:
    A, zi, _ = inputs
    ctx.save_for_backward(A, zi, output)


def _recurN_backward(ctx: Any, grad: torch.Tensor):
    """recur2's and recurN's: A of shape (T, M, M), shared, or (B, T, M, M)."""
    A, zi, y = ctx.saved_tensors
    grad_A, grad_zi, grad_x = _recurrence_gradients(
        _ssm.from_lpv(A), zi, y, grad, *ctx.needs_input_grad[:2]
    )
    if grad_A is not None:
        grad_A = grad_A.reshape(A.shape)
    return grad_A, grad_zi, grad_x


def _lti_recurN_backward(ctx: Any, grad: torch.Tensor):
    """lti_recur2's and lti_recurN's: A of shape (M, M), shared, or (B, M, M)."""
    A, zi, y = ctx.saved_tensors
    grad_A, grad_zi, grad_x = _recurrence_gradients(
        _ssm.from_lti(A), zi, y, grad, *ctx.needs_input_grad[:2]
    )
    if grad_A is not None:
        grad_A = grad_A.reshape(A.shape)
    return grad_A, grad_zi, grad_x


def _lti_recur_backward(ctx: Any, grad: torch.Tensor):
    """lti_recur's: the scalar recurrence, a of shape (), (1,) or (B,), zi (B,)
    and x (B, T)."""
    a, zi, y = ctx.saved_tensors
    grad_a, grad_zi, grad_x = _recurrence_gradients(
        a.reshape(-1, 1, 1, 1),
        zi[:, None],
        y[..., None],
        grad[..., None],
        *ctx.needs_input_grad[:2],
    )
    if grad_a is not None:
        grad_a = grad_a.reshape(a.shape)
    if grad_zi is not None:
        grad_zi = grad_zi[:, 0]
    return grad_a, grad_zi, grad_x[..., 0]


def _scan_setup_context(ctx, inputs, output):
    _, decay, init = inputs
    ctx.save_for_backward(decay, init, output)


def _scan_backward(ctx, grad: torch.Tensor):
    """scan's: h[t] = decay[t] h[t - 1] + impulse[t], each of shape (B, T),
    from init (B,)."""
    decay, init, out = ctx.saved_tensors
    needs_impulse, needs_decay, needs_init = ctx.needs_input_grad
    grad_decay, grad_init, grad_impulse = _recurrence_gradients(
        decay[..., None, None],
        init[:, None],
        out[..., None],
        grad[..., None],
        needs_decay,
        needs_init,
    )
    return (
        grad_impulse[..., 0] if needs_impulse else None,
        grad_decay[..., 0, 0] if needs_decay else None,
        grad_init[:, 0] if needs_init else None,
    )


def _lpc_setup_context(ctx, inputs, output):
    _, A, zi = inputs
    ctx.save_for_backward(A, zi, output)


def _lpc_backward(ctx, grad_y):
    A, zi, y = ctx.saved_tensors
    B, T, order = A.shape
    flipped_A = A.flip(2)
    padded_flipped_A = F.pad(flipped_A.transpose(1, 2), (0, order + 1))
    shifted_A = (
        padded_flipped_A.reshape(B, T + order + 1, order)[:, :-1, :]
        .reshape(B, order, T + order)
        .transpose(1, 2)
        .flip(2)
    )
    if not ctx.needs_input_grad[2]:
        shifted_A = shifted_A[:, order:, :]
        padded_grad_y = grad_y
    else:
        padded_grad_y = F.pad(grad_y.unsqueeze(1), (order, 0)).squeeze(1)
    flipped_grad_x = torch.ops.philtorch.lpc(
        padded_grad_y.flip(1),
        shifted_A.flip(1).conj_physical(),
        torch.zeros_like(zi),
    )
    grad_zi = flipped_grad_x[:, -order:] if ctx.needs_input_grad[2] else None
    if ctx.needs_input_grad[2]:
        flipped_grad_x = flipped_grad_x[:, :-order]
    grad_x = flipped_grad_x.flip(1) if ctx.needs_input_grad[0] else None
    grad_A = None
    if ctx.needs_input_grad[1]:
        valid_y = y[:, :-1]
        padded_y = torch.cat([zi.flip(1), valid_y], dim=1)
        unfolded_y = padded_y.unfold(1, order, 1).flip(2)
        grad_A = unfolded_y.conj_physical() * -flipped_grad_x.flip(1).unsqueeze(2)
    return grad_x, grad_A, grad_zi


def _pararnn_setup_context(ctx, inputs, output):
    jac, rhs = inputs
    ctx.save_for_backward(jac, rhs, output)


def _pararnn_backward(ctx: Any, grad: torch.Tensor):
    """ParaRNN's, whose solution h[t] = -jac[t] h[t - 1] + rhs[t] starts from
    h[0] = rhs[0]: A is -jac[1:], zi rhs[0] and x rhs[1:]; jac[0] is unused."""
    jac, rhs, output = ctx.saved_tensors
    needs_jac, needs_rhs = ctx.needs_input_grad
    grad_A, grad_zi, grad_x = _recurrence_gradients(
        -jac[:, 1:], rhs[:, 0], output[:, 1:], grad[:, 1:], needs_jac, needs_rhs
    )
    grad_jac = grad_rhs = None
    if needs_jac:
        grad_jac = torch.cat([torch.zeros_like(jac[:, :1]), -grad_A], 1)
    if needs_rhs:
        # h[0] is rhs[0] itself, and zi of the steps after.
        grad_rhs = torch.cat([(grad[:, 0] + grad_zi).unsqueeze(1), grad_x], 1)
    return grad_jac, grad_rhs


# The native kernels always return contiguous outputs, so the fakes must too;
# `empty_like` would copy a permuted input's strides.


def _recurrence_fake(order: int | None, time_invariant: bool):
    """The fake of a matrix recurrence op of this order, or any with ``None``.

    Its A is (M, M) or (B, M, M) if ``time_invariant``, else (T, M, M) or
    (B, T, M, M); zi is (B, M) and x (B, T, M).
    """
    dims = (2, 3) if time_invariant else (3, 4)

    def fake(A, zi, x):
        torch._check(x.ndim == 3, lambda: "x must be 3D.")
        torch._check(zi.ndim == 2, lambda: "zi must be 2D.")
        M = x.shape[2]
        if order is not None:
            torch._check(M == order, lambda: f"x must have last dimension of size {order}.")
        torch._check(A.shape[-1] == A.shape[-2] == M, lambda: "A must be square, of x's size.")
        torch._check(A.ndim in dims, lambda: f"A must be {dims[0]}D or {dims[1]}D.")
        torch._check(zi.shape[1] == M, lambda: "zi must have x's last dimension.")
        if not time_invariant:
            torch._check(
                x.shape[1] == A.shape[-3],
                lambda: "x's second dimension must match A's last-3 dimension.",
            )
        torch._check(x.shape[0] == zi.shape[0], lambda: "x and zi must have the same batch size.")
        if A.ndim == dims[1]:
            torch._check(
                A.shape[0] == x.shape[0],
                lambda: f"If A is {dims[1]}D, its first dimension must match x's batch size.",
            )
        return x.new_empty(x.shape)

    return fake


@torch.library.register_fake("philtorch::lti_recur")
def _(A, zi, x):
    torch._check(A.ndim <= 1, "A must be 1D or scalar.")
    torch._check(zi.ndim == 1, "zi must be 1D.")
    torch._check(x.ndim == 2, "x must be 2D.")
    torch._check(x.shape[1] > 0, lambda: "x must contain at least one time step.")
    torch._check(x.shape[0] == zi.shape[0], "x and zi must have the same batch size.")
    if A.ndim == 1 and A.shape[0] != 1:
        torch._check(
            A.shape[0] == x.shape[0],
            "If A is 1D, its length must match x's batch size.",
        )
    return x.new_empty(x.shape)


@torch.library.register_fake("philtorch::scan")
def _scan_fake(impulse, decay, init):
    torch._check(impulse.ndim == 2, lambda: "impulse must be 2D.")
    torch._check(decay.shape == impulse.shape, lambda: "decay must match impulse's shape.")
    torch._check(init.ndim == 1, lambda: "init must be 1D.")
    torch._check(
        init.shape[0] == impulse.shape[0],
        lambda: "init and impulse must have the same batch size.",
    )
    return impulse.new_empty(impulse.shape)


@torch.library.register_fake("philtorch::lpc")
def _lpc_fake(x, A, zi):
    torch._check(x.ndim == 2, lambda: "x must be 2D.")
    torch._check(A.ndim == 3, lambda: "A must be 3D.")
    torch._check(zi.ndim == 2, lambda: "zi must be 2D.")
    torch._check(A.shape[:2] == x.shape, lambda: "A's leading dimensions must match x.")
    torch._check(A.shape[2] == zi.shape[1], lambda: "A and zi must have the same order.")
    torch._check(x.shape[0] == zi.shape[0], lambda: "x and zi must have the same batch size.")
    return x.new_empty(x.shape)


def _pararnn_fake(block_size):
    def fake(jac, rhs):
        torch._check(rhs.ndim == 3, lambda: "rhs must be 3D.")
        torch._check(jac.ndim == 4, lambda: "jac must be 4D.")
        torch._check(
            rhs.shape[2] == block_size,
            lambda: f"rhs's last dimension must be {block_size}.",
        )
        torch._check(
            jac.shape[2:] == (block_size, block_size),
            lambda: f"jac's blocks must be {block_size}x{block_size}.",
        )
        torch._check(
            jac.shape[:2] == rhs.shape[:2],
            lambda: "jac and rhs must have the same batch and time dimensions.",
        )
        return rhs.new_empty(rhs.shape)

    return fake


def _resample_fake(x, table, q, out_len):
    torch._check(x.ndim == 2, lambda: f"x must be 2D (B, T), got {x.ndim}D")
    return x.new_empty((x.shape[0], out_len))


def _resample_setup_context(ctx, inputs, output):
    x, table, q, _ = inputs
    # x is read by the table's gradient only.
    ctx.save_for_backward(x if ctx.needs_input_grad[1] else None, table)
    ctx.q, ctx.in_len, ctx.out_len = q, x.shape[1], output.shape[1]


def _tap_columns(p, q, taps, device):
    """The column of tap k of phase i in the dense kernel: ``(i q) // p + k``."""
    start = torch.arange(p, device=device) * q // p
    return start.unsqueeze(1) + torch.arange(taps, device=device)


def _resample_table_gradient(x, grad, table, q):
    """The gradient in ``table`` of ``philtorch::resample``, from x and grad.

    It is the weight gradient of the CPU path's strided convolution, with the
    dense kernel of width ``((p - 1) q) // p + taps``, gathered at the tap
    columns: ``sum_{b, f} grad[b, f p + i] x[b, q f + (i q) // p - K + k]`` for
    the entry (i, k) of the table, with ``taps = 2 K + 2``.
    """
    p, taps = table.shape
    K = (taps - 2) // 2
    B, out_len = grad.shape
    frames = -(-out_len // p)
    width = ((p - 1) * q) // p + taps
    # The op accumulates in float32 for the 16-bit dtypes.
    acc = torch.promote_types(x.dtype, torch.float32)
    g = F.pad(grad.to(acc), (0, frames * p - out_len)).reshape(B, frames, p).mT
    # The right pad crops, or adds zeros, to the length the frames read.
    right = q * (frames - 1) + width - K - x.size(1)
    xp = F.pad(x.to(acc), (K, right)).unsqueeze(1)
    dense = torch.nn.grad.conv1d_weight(xp, (p, 1, width), g, stride=q).squeeze(1)
    columns = _tap_columns(p, q, taps, table.device)
    return dense.gather(1, columns).to(table.dtype)


def _transpose_index(p, q, taps, device):
    """The index of :func:`_transposed_table` into the flattened table."""
    K = (taps - 2) // 2
    K_t = -(-K * p // q)
    j = torch.arange(q, device=device).unsqueeze(1)
    m = j * p // q - K_t + torch.arange(2 * K_t + 2, device=device)
    phase = m % p
    column = j - (q * (m // p) + phase * q // p - K)
    # Entries off the taps of their row point at the zero appended to the table.
    inside = (column >= 0) & (column < taps)
    return torch.where(inside, phase * taps + column, p * taps)


@functools.lru_cache(maxsize=16)
def _cached_transpose_index(p, q, taps, device):
    # Built outside inference mode, so that a later backward can save it.
    with torch.inference_mode(False):
        return _transpose_index(p, q, taps, device)


def _transposed_table(table, q):
    """The table of the transposed resampler, of stride p and q phases.

    The operator is ``M[m, n] = table[m mod p, n - first(m)]``, zero outside the
    taps, with ``first(m) = q (m // p) + (m mod p) q // p - K``. Its transpose
    has the same entries, in the layout of ``table`` with the roles of p and q
    swapped: ``K' = ceil(K p / q)`` and ``first'_j = (j p) // q - K'``. The
    entries are gathered from ``table`` (and zeros), so they are differentiable
    in it. Eager calls cache the index, whose dozen small ops would otherwise
    dominate a backward pass at small batches. Traced code (torch.compile,
    opcheck) builds it in the graph: its table is a fake or functional tensor,
    and so would be the cached index.
    """
    p, taps = table.shape
    index_of = _cached_transpose_index if type(table) is torch.Tensor else _transpose_index
    index = index_of(p, q, taps, table.device)
    return torch.cat([table.flatten(), table.new_zeros(1)])[index]


def _resample_gradient(run, ctx, grad):
    """The gradients of ``philtorch::resample``, with ``run`` the op to call.

    ``run`` is the autograd function's ``apply`` in eager mode and the op in
    compiled code. The gradient in x is the op with the transposed table, p as
    the stride and x's length as the output length; the table is a function of
    ``table``, so the derivatives of any order follow. The op is linear in the
    table as well: the gradient in ``table`` is :func:`_resample_table_gradient`.
    """
    x, table = ctx.saved_tensors
    needs_x, needs_table = ctx.needs_input_grad[:2]
    grad_x = grad_table = None
    if needs_x:
        grad_x = run(grad, _transposed_table(table, ctx.q), table.size(0), ctx.in_len)
    if needs_table:
        grad_table = _resample_table_gradient(x, grad, table, ctx.q)
    return grad_x, grad_table, None, None


for name, order, time_invariant in [
    ("recur2", 2, False),
    ("recurN", None, False),
    ("lti_recur2", 2, True),
    ("lti_recurN", None, True),
]:
    torch.library.register_fake(f"philtorch::{name}", _recurrence_fake(order, time_invariant))
    torch.library.register_autograd(
        f"philtorch::{name}",
        _lti_recurN_backward if time_invariant else _recurN_backward,
        setup_context=_setup_context,
    )
torch.library.register_autograd(
    "philtorch::lti_recur", _lti_recur_backward, setup_context=_setup_context
)
torch.library.register_autograd(
    "philtorch::scan", _scan_backward, setup_context=_scan_setup_context
)
torch.library.register_autograd("philtorch::lpc", _lpc_backward, setup_context=_lpc_setup_context)


for block_size in (2, 3):
    op = f"parallel_reduce_block_diag_{block_size}x{block_size}_cuda"
    if hasattr(torch.ops.parallel_reduce_cuda, op):  # pragma: no cover - CUDA-only schema
        torch.library.register_fake(f"parallel_reduce_cuda::{op}", _pararnn_fake(block_size))
        torch.library.register_autograd(
            f"parallel_reduce_cuda::{op}", _pararnn_backward, setup_context=_pararnn_setup_context
        )

if hasattr(torch.ops.philtorch, "resample"):  # pragma: no cover - CUDA-only op
    torch.library.register_fake("philtorch::resample", _resample_fake)
    torch.library.register_autograd(
        "philtorch::resample",
        lambda ctx, grad: _resample_gradient(torch.ops.philtorch.resample, ctx, grad),
        setup_context=_resample_setup_context,
    )
