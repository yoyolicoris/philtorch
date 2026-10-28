"""Lightweight vendor shim for torchlpc ops."""

from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd import Function

from . import _C  # noqa: F401


def _lpc_gradients(run, A, zi, y, grad_y, needs_input_grad):
    """The gradients of x, A and zi from ``grad_y``; ``run`` is lpc: the op or
    ``AllPole.apply``.

    The adjoint of an all-pole filter is the same filter backward in time,
    with A's coefficients shifted to the steps they act on.
    """
    B, T, order = A.shape
    flipped_A = A.flip(2)
    padded_flipped_A = F.pad(flipped_A.transpose(1, 2), (0, order + 1))
    shifted_A = (
        padded_flipped_A.reshape(B, T + order + 1, order)[:, :-1, :]
        .reshape(B, order, T + order)
        .transpose(1, 2)
        .flip(2)
    )
    if not needs_input_grad[2]:
        shifted_A = shifted_A[:, order:, :]
        padded_grad_y = grad_y
    else:
        padded_grad_y = F.pad(grad_y.unsqueeze(1), (order, 0)).squeeze(1)
    flipped_grad_x = run(
        padded_grad_y.flip(1),
        shifted_A.flip(1).conj_physical(),
        torch.zeros_like(zi),
    )
    grad_zi = flipped_grad_x[:, -order:] if needs_input_grad[2] else None
    if needs_input_grad[2]:
        flipped_grad_x = flipped_grad_x[:, :-order]
    grad_x = flipped_grad_x.flip(1) if needs_input_grad[0] else None
    grad_A = None
    if needs_input_grad[1]:
        valid_y = y[:, :-1]
        padded_y = torch.cat([zi.flip(1), valid_y], dim=1)
        unfolded_y = padded_y.unfold(1, order, 1).flip(2)
        grad_A = unfolded_y.conj_physical() * -flipped_grad_x.flip(1).unsqueeze(2)
    return grad_x, grad_A, grad_zi


class AllPole(Function):
    @staticmethod
    def forward(x: torch.Tensor, A: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
        return torch.ops.philtorch.lpc(x, A, zi)

    @staticmethod
    def setup_context(ctx: Any, inputs: list[Any], output: Any) -> Any:
        _, A, zi = inputs
        y = output
        ctx.save_for_backward(A, zi, y)
        ctx.save_for_forward(A, zi, y)

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor):  # type: ignore[override]
        return _lpc_gradients(AllPole.apply, *ctx.saved_tensors, grad_y, ctx.needs_input_grad)

    @staticmethod
    def jvp(
        ctx: Any, grad_x: torch.Tensor, grad_A: torch.Tensor, grad_zi: torch.Tensor
    ) -> torch.Tensor:
        A, zi, y = ctx.saved_tensors
        order = A.shape[2]
        fwd_zi = grad_zi if grad_zi is not None else torch.zeros_like(zi)
        fwd_x = grad_x if grad_x is not None else torch.zeros_like(y)
        if grad_A is not None:
            unfolded_y = torch.cat([zi.flip(1), y[:, :-1]], dim=1).unfold(1, order, 1).flip(2)
            fwd_x = fwd_x - torch.sum(unfolded_y * grad_A, dim=2)
        return AllPole.apply(fwd_x, A, fwd_zi)

    @staticmethod
    def vmap(info, in_dims, *args):
        def maybe_expand_bdim_at_front(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        x, A, zi = tuple(
            map(
                lambda x: x.reshape(-1, *x.shape[2:]),
                map(maybe_expand_bdim_at_front, args, in_dims),
            )
        )
        y = AllPole.apply(x, A, zi)
        return y.reshape(info.batch_size, -1, *y.shape[1:]), 0


def lpc(x: torch.Tensor, A: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return torch.ops.philtorch.lpc(x, A, zi)
    return AllPole.apply(x, A, zi)


__all__ = ["AllPole", "lpc"]
