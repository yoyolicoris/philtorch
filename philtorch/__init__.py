from pathlib import Path
import warnings
import torch
import torch.nn.functional as F
from typing import Any, Optional

from . import _C  # noqa: F401

try:
    from ._helion import (
        lti_recursion_loop,
        lti_shared_A_recursion_loop,
        lpv_recursion_loop,
        lpv_shared_A_recursion_loop,
    )
except ImportError:
    HELION_LOADED = False
    warnings.warn(
        "Helion kernels not loaded. Please ensure Helion is installed and compatible with your PyTorch version."
    )
else:
    HELION_LOADED = True


try:
    from ._version import __version__ as __version__  # type: ignore
except ImportError:
    __version__ = Path(__file__).parent.joinpath("VERSION.txt").read_text().strip()


def _recurN_backward(f):
    def closure(
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        A, zi, y = ctx.saved_tensors
        grad_x = grad_A = grad_zi = None

        AmT = A.mT.conj_physical()
        AmT_rolled = torch.roll(AmT, shifts=-1, dims=-3)

        runner = f if A.shape[-1] != 2 else torch.ops.philtorch.recur2

        flipped_grad_x = runner(
            AmT_rolled.flip(-3),
            torch.zeros_like(zi),
            grad_y.flip(1),
        )

        if ctx.needs_input_grad[1]:
            grad_zi = (AmT[..., 0, :, :] @ flipped_grad_x[:, -1, :, None]).squeeze(-1)

        if ctx.needs_input_grad[2]:
            grad_x = flipped_grad_x.flip(1)

        if ctx.needs_input_grad[0]:
            valid_y = y[:, :-1]
            padded_y = torch.cat([zi.unsqueeze(1), valid_y], dim=1)

            if A.dim() == 3:
                grad_A = flipped_grad_x.flip(1).permute(
                    1, 2, 0
                ) @ padded_y.conj_physical().transpose(0, 1)
            else:
                grad_A = padded_y.conj_physical().unsqueeze(-2) * flipped_grad_x.flip(
                    1
                ).unsqueeze(-1)

        return grad_A, grad_zi, grad_x

    return closure


def _lti_recurN_backward(f):
    def closure(
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        A, zi, y = ctx.saved_tensors
        grad_x = grad_A = grad_zi = None

        AmT = A.mT.conj_physical()

        runner = (
            # torch.ops.philtorch.lti_recurN
            f
            if A.shape[-1] != 2
            else torch.ops.philtorch.lti_recur2
        )

        flipped_grad_x = runner(AmT, torch.zeros_like(zi), grad_y.flip(1))

        if ctx.needs_input_grad[1]:
            grad_zi = (AmT @ flipped_grad_x[:, -1, :, None]).squeeze(-1)

        if ctx.needs_input_grad[2]:
            grad_x = flipped_grad_x.flip(1)

        if ctx.needs_input_grad[0]:
            valid_y = y[:, :-1]
            padded_y = torch.cat([zi.unsqueeze(1), valid_y], dim=1)
            if A.dim() == 2:
                grad_A = flipped_grad_x.flip(1).flatten(
                    0, 1
                ).T @ padded_y.conj_physical().flatten(0, 1)
            else:
                grad_A = flipped_grad_x.flip(1).mT @ padded_y.conj_physical()

        return grad_A, grad_zi, grad_x

    return closure


def _lti_recur_backward(
    ctx: Any, grad_out: torch.Tensor
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    a, init, out = ctx.saved_tensors
    grad_a = grad_x = grad_init = None

    bp_init = grad_out[:, -1]
    flipped_grad_x = torch.cat(
        [
            bp_init.unsqueeze(1),
            torch.ops.philtorch.lti_recur(
                a.conj_physical(),
                bp_init,
                grad_out[:, :-1].flip(1),
            ),
        ],
        dim=1,
    )

    if ctx.needs_input_grad[1]:
        grad_init = flipped_grad_x[:, -1] * a.conj_physical()

    if ctx.needs_input_grad[2]:
        grad_x = flipped_grad_x.flip(1)

    if ctx.needs_input_grad[0]:
        valid_out = out[:, :-1]
        padded_out = torch.cat([init.unsqueeze(1), valid_out], dim=1)
        if a.dim() == 1:
            grad_a = torch.linalg.vecdot(padded_out, flipped_grad_x.flip(1))
        else:
            grad_a = padded_out.flatten().conj() @ flipped_grad_x.flip(1).flatten()

    return grad_a, grad_init, grad_x


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


@torch.library.register_fake("philtorch::scan")
def _scan_fake(impulse, decay, init):
    torch._check(impulse.ndim == 2, "impulse must be 2D.")
    torch._check(decay.shape == impulse.shape, "decay must match impulse's shape.")
    torch._check(init.ndim == 1, "init must be 1D.")
    torch._check(
        init.shape[0] == impulse.shape[0],
        "init and impulse must have the same batch size.",
    )
    return torch.empty_like(impulse)


@torch.library.register_fake("philtorch::lpc")
def _lpc_fake(x, A, zi):
    torch._check(x.ndim == 2, "x must be 2D.")
    torch._check(A.ndim == 3, "A must be 3D.")
    torch._check(zi.ndim == 2, "zi must be 2D.")
    torch._check(A.shape[:2] == x.shape, "A's leading dimensions must match x.")
    torch._check(A.shape[2] == zi.shape[1], "A and zi must have the same order.")
    torch._check(x.shape[0] == zi.shape[0], "x and zi must have the same batch size.")
    return torch.empty_like(x)


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


def _pararnn_backward(runner):
    def closure(ctx, grad_output):
        jac, rhs, output = ctx.saved_tensors
        A = -jac[:, 1:]
        zi = rhs[:, 0]
        y = output[:, 1:]
        grad_y = grad_output[:, 1:]

        AmT = A.mT.conj_physical()
        AmT_rolled = torch.roll(AmT, shifts=-1, dims=-3)
        reverse_A = AmT_rolled.flip(-3)
        reverse_jac = F.pad(-reverse_A, (0, 0, 0, 0, 1, 0))
        reverse_rhs = torch.cat(
            [torch.zeros_like(zi).unsqueeze(1), grad_y.flip(1)], dim=1
        )
        flipped_grad_x = runner(reverse_jac, reverse_rhs)[:, 1:]
        grad_zi = (AmT[..., 0, :, :] @ flipped_grad_x[:, -1, :, None]).squeeze(-1)
        grad_x = flipped_grad_x.flip(1)

        padded_y = torch.cat([zi.unsqueeze(1), y[:, :-1]], dim=1)
        grad_A = padded_y.conj_physical().unsqueeze(-2) * grad_x.unsqueeze(-1)
        grad_jac = torch.cat([torch.zeros_like(jac[:, :1]), -grad_A], dim=1)
        grad_rhs = torch.cat(
            [(grad_output[:, 0] + grad_zi).unsqueeze(1), grad_x], dim=1
        )
        return (
            grad_jac if ctx.needs_input_grad[0] else None,
            grad_rhs if ctx.needs_input_grad[1] else None,
        )

    return closure


torch.library.register_autograd(
    "philtorch::scan", _scan_backward, setup_context=_scan_setup_context
)
torch.library.register_autograd(
    "philtorch::lpc", _lpc_backward, setup_context=_lpc_setup_context
)


if hasattr(  # pragma: no cover - CUDA-only schema
    torch.ops.parallel_reduce_cuda, "parallel_reduce_block_diag_2x2_cuda"
):

    @torch.library.register_fake(
        "parallel_reduce_cuda::parallel_reduce_block_diag_2x2_cuda"
    )
    def _parallel_reduce_block_diag_2x2_fake(jac, rhs):
        return torch.empty_like(rhs)

    torch.library.register_autograd(
        "parallel_reduce_cuda::parallel_reduce_block_diag_2x2_cuda",
        _pararnn_backward(
            torch.ops.parallel_reduce_cuda.parallel_reduce_block_diag_2x2_cuda
        ),
        setup_context=_pararnn_setup_context,
    )


if hasattr(  # pragma: no cover - CUDA-only schema
    torch.ops.parallel_reduce_cuda, "parallel_reduce_block_diag_3x3_cuda"
):

    @torch.library.register_fake(
        "parallel_reduce_cuda::parallel_reduce_block_diag_3x3_cuda"
    )
    def _parallel_reduce_block_diag_3x3_fake(jac, rhs):
        return torch.empty_like(rhs)

    torch.library.register_autograd(
        "parallel_reduce_cuda::parallel_reduce_block_diag_3x3_cuda",
        _pararnn_backward(
            torch.ops.parallel_reduce_cuda.parallel_reduce_block_diag_3x3_cuda
        ),
        setup_context=_pararnn_setup_context,
    )


torch.library.register_autograd(
    "philtorch::recur2",
    _recurN_backward(torch.ops.philtorch.recur2),
    setup_context=_setup_context,
)
torch.library.register_autograd(
    "philtorch::recurN",
    _recurN_backward(torch.ops.philtorch.recurN),
    setup_context=_setup_context,
)
torch.library.register_autograd(
    "philtorch::lti_recur2",
    _lti_recurN_backward(torch.ops.philtorch.lti_recur2),
    setup_context=_setup_context,
)
torch.library.register_autograd(
    "philtorch::lti_recurN",
    _lti_recurN_backward(torch.ops.philtorch.lti_recurN),
    setup_context=_setup_context,
)
torch.library.register_autograd(
    "philtorch::lti_recur", _lti_recur_backward, setup_context=_setup_context
)


if HELION_LOADED:

    @torch.library.custom_op("philtorch::hl_lti_recurN", mutates_args=())
    def hl_lti_recurN(
        A: torch.Tensor, zi: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        if A.ndim == 2:
            return lti_shared_A_recursion_loop(A, zi, x)
        return lti_recursion_loop(A, zi, x)

    @torch.library.custom_op("philtorch::hl_recurN", mutates_args=())
    def hl_recurN(A: torch.Tensor, zi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if A.ndim == 3:
            return lpv_shared_A_recursion_loop(A, zi, x)
        return lpv_recursion_loop(A, zi, x)

    @hl_lti_recurN.register_fake
    def _(A, zi, x):
        torch._check(A.shape[-1] == A.shape[-2] == x.shape[2], "A must be square.")
        torch._check(A.ndim in (2, 3), "A must be 2D or 3D.")
        torch._check(
            zi.shape[1] == x.shape[2], "zi must have last dimension of size 2."
        )
        torch._check(
            x.shape[0] == zi.shape[0], "x and zi must have the same batch size."
        )
        if A.ndim == 3:
            torch._check(
                A.shape[0] == x.shape[0],
                "If A is 3D, its first dimension must match x's batch size.",
            )
        # return torch.empty_like(x)
        return x.new_empty(x.shape[0], x.shape[1] + 1, x.shape[2])[:, 1:]

    @hl_recurN.register_fake
    def _(A, zi, x):
        torch._check(A.shape[-1] == A.shape[-2] == x.shape[2], "A must be square.")
        torch._check(A.ndim in (3, 4), "A must be 3D or 4D.")
        torch._check(
            zi.shape[1] == x.shape[2], "zi must have last dimension of size 2."
        )
        torch._check(
            x.shape[1] == A.shape[-3],
            "x's second dimension must match A's last-3 dimension.",
        )
        torch._check(
            x.shape[0] == zi.shape[0], "x and zi must have the same batch size."
        )
        if A.ndim == 4:
            torch._check(
                A.shape[0] == x.shape[0],
                "If A is 4D, its first dimension must match x's batch size.",
            )
        # return torch.empty_like(x)
        return x.new_empty(x.shape[0], x.shape[1] + 1, x.shape[2])[:, 1:]

    torch.library.register_autograd(
        "philtorch::hl_lti_recurN",
        _lti_recurN_backward(hl_lti_recurN),
        setup_context=_setup_context,
    )
    torch.library.register_autograd(
        "philtorch::hl_recurN",
        _recurN_backward(hl_recurN),
        setup_context=_setup_context,
    )
