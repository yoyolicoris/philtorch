"""Differentiable, time-domain linear digital filters for PyTorch.

Time-invariant filters live in :mod:`philtorch.lti`, time-varying (linear
parameter-varying) filters in :mod:`philtorch.lpv`, state estimation in
:mod:`philtorch.estimation`, and sequence alignment in :mod:`philtorch.align`.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ._build_check import check_torch_build, installed_torch_requirement

try:
    from ._version import __version__ as __version__  # type: ignore
except ImportError:
    __version__ = Path(__file__).parent.joinpath("VERSION.txt").read_text().strip()

# Fail clearly on a wheel installed next to a torch it wasn't built for, before
# loading the extension would fail with an undefined-symbol error.
check_torch_build(__version__, torch.__version__, torch.version.cuda, installed_torch_requirement())

from . import _C  # noqa: E402, F401

# The native recurrences' registered gradients, which compiled code uses: there
# philtorch._recur calls the ops directly. Each maps its layout to that
# module's, (B or 1, T or 1, M, M) matrices, whose _gradients defines them once.


def _recurrence_gradients(A, zi, h, grad, needs_A, needs_zi):
    # Imported here, so that importing philtorch doesn't load Triton.
    from ._recur import _forward, _gradients

    return _gradients(_forward, A, zi, h, grad, needs_A, needs_zi)


def _recurN_backward(ctx: Any, grad: torch.Tensor):
    """recur2's and recurN's: A of shape (T, M, M), shared, or (B, T, M, M)."""
    A, zi, y = ctx.saved_tensors
    shared = A.dim() == 3
    grad_A, grad_zi, grad_x = _recurrence_gradients(
        A[None] if shared else A, zi, y, grad, *ctx.needs_input_grad[:2]
    )
    if grad_A is not None and shared:
        grad_A = grad_A[0]
    return grad_A, grad_zi, grad_x


def _lti_recurN_backward(ctx: Any, grad: torch.Tensor):
    """lti_recur2's and lti_recurN's: A of shape (M, M), shared, or (B, M, M)."""
    A, zi, y = ctx.saved_tensors
    grad_A, grad_zi, grad_x = _recurrence_gradients(
        A[None, None] if A.dim() == 2 else A[:, None], zi, y, grad, *ctx.needs_input_grad[:2]
    )
    if grad_A is not None:
        grad_A = grad_A.reshape(A.shape)
    return grad_A, grad_zi, grad_x


def _lti_recur_backward(ctx: Any, grad: torch.Tensor):
    """lti_recur's: the scalar recurrence, a of shape (), (1,) or (B,), zi (B,), x (B, T)."""
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


def _setup_context(ctx: Any, inputs: list[Any], output: Any) -> Any:
    A, zi, _ = inputs
    y = output
    ctx.save_for_backward(A, zi, y)


@torch.library.register_fake("philtorch::recur2")
def _(A, zi, x):
    torch._check(A.shape[-1] == A.shape[-2] == 2, "A must be square.")
    torch._check(A.ndim in (3, 4), "A must be 3D or 4D.")
    torch._check(zi.shape[1] == 2, "zi must have last dimension of size 2.")
    torch._check(x.shape[2] == 2, "x must have last dimension of size 2.")
    torch._check(
        x.shape[1] == A.shape[-3],
        "x's second dimension must match A's last-3 dimension.",
    )
    torch._check(x.shape[0] == zi.shape[0], "x and zi must have the same batch size.")
    if A.ndim == 4:
        torch._check(
            A.shape[0] == x.shape[0],
            "If A is 4D, its first dimension must match x's batch size.",
        )
    return torch.empty_like(x)


@torch.library.register_fake("philtorch::recurN")
def _(A, zi, x):
    torch._check(A.shape[-1] == A.shape[-2] == x.shape[2], "A must be square.")
    torch._check(A.ndim in (3, 4), "A must be 3D or 4D.")
    torch._check(zi.shape[1] == x.shape[2], "zi must have last dimension of size 2.")
    torch._check(
        x.shape[1] == A.shape[-3],
        "x's second dimension must match A's last-3 dimension.",
    )
    torch._check(x.shape[0] == zi.shape[0], "x and zi must have the same batch size.")
    if A.ndim == 4:
        torch._check(
            A.shape[0] == x.shape[0],
            "If A is 4D, its first dimension must match x's batch size.",
        )
    return torch.empty_like(x)


@torch.library.register_fake("philtorch::lti_recur2")
def _(A, zi, x):
    torch._check(A.shape[-1] == A.shape[-2] == 2, "A must be square.")
    torch._check(A.ndim in (2, 3), "A must be 2D or 3D.")
    torch._check(zi.shape[1] == 2, "zi must have last dimension of size 2.")
    torch._check(x.shape[2] == 2, "x must have last dimension of size 2.")
    torch._check(x.shape[0] == zi.shape[0], "x and zi must have the same batch size.")
    if A.ndim == 3:
        torch._check(
            A.shape[0] == x.shape[0],
            "If A is 3D, its first dimension must match x's batch size.",
        )
    return torch.empty_like(x)


@torch.library.register_fake("philtorch::lti_recurN")
def _(A, zi, x):
    torch._check(A.shape[-1] == A.shape[-2] == x.shape[2], "A must be square.")
    torch._check(A.ndim in (2, 3), "A must be 2D or 3D.")
    torch._check(zi.shape[1] == x.shape[2], "zi must have last dimension of size 2.")
    torch._check(x.shape[0] == zi.shape[0], "x and zi must have the same batch size.")
    if A.ndim == 3:
        torch._check(
            A.shape[0] == x.shape[0],
            "If A is 3D, its first dimension must match x's batch size.",
        )
    return torch.empty_like(x)


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
    return torch.empty_like(x)


# The vendored scan, LPC, and ParaRNN kernels always return contiguous outputs,
# so their fakes must too; `empty_like` would copy a permuted input's strides.
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


def _scan_setup_context(ctx, inputs, output):
    _, decay, init = inputs
    ctx.save_for_backward(decay, init, output)


def _scan_backward(ctx, grad_out):
    decay, init, out = ctx.saved_tensors
    n_dims = decay.size(0)
    padded_decay = F.pad(decay.unsqueeze(1), (0, 1)).squeeze(1)
    if ctx.needs_input_grad[2]:
        padded_grad = F.pad(grad_out.unsqueeze(1), (1, 0)).squeeze(1)
    else:
        padded_grad, padded_decay = grad_out, padded_decay[:, 1:]
    flipped = torch.ops.philtorch.scan(
        padded_grad.flip(1),
        padded_decay.flip(1).conj_physical(),
        padded_grad.new_zeros(n_dims),
    )
    grad_init = flipped[:, -1] if ctx.needs_input_grad[2] else None
    if ctx.needs_input_grad[2]:
        flipped = flipped[:, :-1]
    grad_impulse = flipped.flip(1) if ctx.needs_input_grad[0] else None
    if ctx.needs_input_grad[1]:
        grad_decay = torch.cat(
            [init.unsqueeze(1), out[:, :-1]], dim=1
        ).conj_physical() * flipped.flip(1)
    else:
        grad_decay = None
    return grad_impulse, grad_decay, grad_init


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


torch.library.register_autograd(
    "philtorch::scan", _scan_backward, setup_context=_scan_setup_context
)
torch.library.register_autograd("philtorch::lpc", _lpc_backward, setup_context=_lpc_setup_context)


if hasattr(  # pragma: no cover - CUDA-only schema
    torch.ops.parallel_reduce_cuda, "parallel_reduce_block_diag_2x2_cuda"
):
    torch.library.register_fake(
        "parallel_reduce_cuda::parallel_reduce_block_diag_2x2_cuda",
        _pararnn_fake(2),
    )

    torch.library.register_autograd(
        "parallel_reduce_cuda::parallel_reduce_block_diag_2x2_cuda",
        _pararnn_backward,
        setup_context=_pararnn_setup_context,
    )


if hasattr(  # pragma: no cover - CUDA-only schema
    torch.ops.parallel_reduce_cuda, "parallel_reduce_block_diag_3x3_cuda"
):
    torch.library.register_fake(
        "parallel_reduce_cuda::parallel_reduce_block_diag_3x3_cuda",
        _pararnn_fake(3),
    )

    torch.library.register_autograd(
        "parallel_reduce_cuda::parallel_reduce_block_diag_3x3_cuda",
        _pararnn_backward,
        setup_context=_pararnn_setup_context,
    )


torch.library.register_autograd(
    "philtorch::recur2",
    _recurN_backward,
    setup_context=_setup_context,
)
torch.library.register_autograd(
    "philtorch::recurN",
    _recurN_backward,
    setup_context=_setup_context,
)
torch.library.register_autograd(
    "philtorch::lti_recur2",
    _lti_recurN_backward,
    setup_context=_setup_context,
)
torch.library.register_autograd(
    "philtorch::lti_recurN",
    _lti_recurN_backward,
    setup_context=_setup_context,
)
torch.library.register_autograd(
    "philtorch::lti_recur", _lti_recur_backward, setup_context=_setup_context
)
