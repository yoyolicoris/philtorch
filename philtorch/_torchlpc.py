"""Lightweight vendor shim for torchlpc ops."""

from typing import Any, List

import torch
import torch.nn.functional as F
from torch.autograd import Function

from . import _C  # noqa: F401


class AllPole(Function):
    @staticmethod
    def forward(x: torch.Tensor, A: torch.Tensor, zi: torch.Tensor) -> torch.Tensor:
        return torch.ops.philtorch.lpc(x, A, zi)

    @staticmethod
    def setup_context(ctx: Any, inputs: List[Any], output: Any) -> Any:
        _, A, zi = inputs
        y = output
        ctx.save_for_backward(A, zi, y)
        ctx.save_for_forward(A, zi, y)

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor):  # type: ignore[override]
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
        flipped_grad_x = AllPole.apply(
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

    @staticmethod
    def jvp(
        ctx: Any, grad_x: torch.Tensor, grad_A: torch.Tensor, grad_zi: torch.Tensor
    ) -> torch.Tensor:
        A, zi, y = ctx.saved_tensors
        order = A.shape[2]
        fwd_zi = grad_zi if grad_zi is not None else torch.zeros_like(zi)
        fwd_x = grad_x if grad_x is not None else torch.zeros_like(y)
        if grad_A is not None:
            unfolded_y = (
                torch.cat([zi.flip(1), y[:, :-1]], dim=1).unfold(1, order, 1).flip(2)
            )
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


class ScanRecurrence(Function):
    @staticmethod
    def forward(
        impulse: torch.Tensor, decay: torch.Tensor, init: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.philtorch.scan(impulse, decay, init)

    @staticmethod
    def setup_context(ctx: Any, inputs: List[Any], output: Any) -> Any:
        impulse, decay, init = inputs
        ctx.save_for_backward(decay, init, output)
        ctx.save_for_forward(decay, init, output)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        decay, init, out = ctx.saved_tensors
        n_dims = decay.size(0)
        padded_decay = F.pad(decay.unsqueeze(1), (0, 1)).squeeze(1)
        if ctx.needs_input_grad[2]:
            padded_grad = F.pad(grad_out.unsqueeze(1), (1, 0)).squeeze(1)
        else:
            padded_grad, padded_decay = grad_out, padded_decay[:, 1:]
        flipped = ScanRecurrence.apply(
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

    @staticmethod
    def jvp(
        ctx: Any,
        grad_impulse: torch.Tensor,
        grad_decay: torch.Tensor,
        grad_init: torch.Tensor,
    ) -> torch.Tensor:
        decay, init, out = ctx.saved_tensors
        fwd_init = grad_init if grad_init is not None else torch.zeros_like(init)
        fwd_imp = grad_impulse if grad_impulse is not None else torch.zeros_like(out)
        if grad_decay is not None:
            fwd_imp = (
                fwd_imp
                + torch.cat([init.unsqueeze(1), out[:, :-1]], dim=1) * grad_decay
            )
        return ScanRecurrence.apply(fwd_imp, decay, fwd_init)

    @staticmethod
    def vmap(info, in_dims, *args):
        def maybe_expand_bdim_at_front(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        impulse, decay, init = tuple(
            map(
                lambda t: t.reshape(-1, *t.shape[2:]) if t.dim() > 1 else t.reshape(-1),
                map(maybe_expand_bdim_at_front, args, in_dims),
            )
        )
        return (
            ScanRecurrence.apply(impulse, decay, init).reshape(
                info.batch_size, -1, *impulse.shape[1:]
            ),
            0,
        )


__all__ = ["AllPole", "ScanRecurrence"]
