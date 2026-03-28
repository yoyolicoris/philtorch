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

__version__ = Path(__file__).parent.joinpath("VERSION.txt").read_text()


@torch.library.register_fake("philtorch::recur2")
def _(A, zi, x):
    torch._check(A.shape[-1] == A.shape[-2] == 2, "A must be square.")
    torch._check(A.ndim in (3, 4), "A must be 3D or 4D.")
    torch._check(zi.shape[-1] == 2, "zi must have last dimension of size 2.")
    torch._check(x.shape[-1] == 2, "x must have last dimension of size 2.")
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
    torch._check(A.shape[-1] == A.shape[-2] == x.shape[-1], "A must be square.")
    torch._check(A.ndim in (3, 4), "A must be 3D or 4D.")
    torch._check(zi.shape[-1] == x.shape[-1], "zi must have last dimension of size 2.")
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


def _recurN_setup_context(ctx: Any, inputs: list[Any], output: Any) -> Any:
    A, zi, _ = inputs
    y = output
    ctx.save_for_backward(A, zi, y)


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


torch.library.register_autograd(
    "philtorch::recur2", _recurN_backward, setup_context=_recurN_setup_context
)
torch.library.register_autograd(
    "philtorch::recurN", _recurN_backward, setup_context=_recurN_setup_context
)
