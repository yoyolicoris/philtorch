from pathlib import Path
import warnings
import torch
from typing import Any, Optional

try:
    from . import _C

    EXTENSION_LOADED = True
except ImportError:
    EXTENSION_LOADED = False
    warnings.warn("Custom extension not loaded.")


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


__version__ = Path(__file__).parent.joinpath("VERSION.txt").read_text()


def _recurN_backward(
    ctx: Any, grad_y: torch.Tensor
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    A, zi, y = ctx.saved_tensors
    grad_x = grad_A = grad_zi = None

    AmT = A.mT.conj_physical()
    AmT_rolled = torch.roll(AmT, shifts=-1, dims=-3)

    runner = (
        torch.ops.philtorch.recurN if A.shape[-1] != 2 else torch.ops.philtorch.recur2
    )

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


def _lti_recurN_backward(
    ctx: Any, grad_y: torch.Tensor
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    A, zi, y = ctx.saved_tensors
    grad_x = grad_A = grad_zi = None

    AmT = A.mT.conj_physical()

    runner = (
        torch.ops.philtorch.lti_recurN
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


if EXTENSION_LOADED:

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
        torch._check(
            x.shape[0] == zi.shape[0], "x and zi must have the same batch size."
        )
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
        return torch.empty_like(x)

    @torch.library.register_fake("philtorch::lti_recur2")
    def _(A, zi, x):
        torch._check(A.shape[-1] == A.shape[-2] == 2, "A must be square.")
        torch._check(A.ndim in (2, 3), "A must be 2D or 3D.")
        torch._check(zi.shape[1] == 2, "zi must have last dimension of size 2.")
        torch._check(x.shape[2] == 2, "x must have last dimension of size 2.")
        torch._check(
            x.shape[0] == zi.shape[0], "x and zi must have the same batch size."
        )
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
        return torch.empty_like(x)

    @torch.library.register_fake("philtorch::lti_recur")
    def _(A, zi, x):
        torch._check(A.ndim <= 1, "A must be 1D or scalar.")
        torch._check(
            zi.shape[0] == x.shape[0], "x and zi must have the same batch size."
        )
        if A.ndim == 1 and A.shape[0] != 1:
            torch._check(
                A.shape[0] == x.shape[0],
                "If A is 1D, its length must match x's batch size.",
            )
        return torch.empty_like(x)

    torch.library.register_autograd(
        "philtorch::recur2", _recurN_backward, setup_context=_setup_context
    )
    torch.library.register_autograd(
        "philtorch::recurN", _recurN_backward, setup_context=_setup_context
    )
    torch.library.register_autograd(
        "philtorch::lti_recur2", _lti_recurN_backward, setup_context=_setup_context
    )
    torch.library.register_autograd(
        "philtorch::lti_recurN", _lti_recurN_backward, setup_context=_setup_context
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
        return torch.empty_like(x)

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
        return torch.empty_like(x)

    torch.library.register_autograd(
        "philtorch::hl_lti_recurN", _lti_recurN_backward, setup_context=_setup_context
    )
    torch.library.register_autograd(
        "philtorch::hl_recurN", _recurN_backward, setup_context=_setup_context
    )
