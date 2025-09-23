import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import functional as F
from typing import Optional, Tuple, Any, List


class Recurrence(Function):
    # mostly copied from torchlpc
    @staticmethod
    def forward(a: Tensor, init: Tensor, x: Tensor) -> Tensor:
        return torch.ops.philtorch.lti_recur(a, init, x)

    @staticmethod
    def setup_context(ctx: Any, inputs: List[Any], output: Any) -> Any:
        a, init, _ = inputs
        ctx.save_for_backward(a, init, output)
        ctx.save_for_forward(a, init, output)

    @staticmethod
    def backward(
        ctx: Any, grad_out: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        a, init, out = ctx.saved_tensors
        grad_a = grad_x = grad_init = None

        bp_init = grad_out[:, -1]
        flipped_grad_x = Recurrence.apply(
            a.conj_physical(),
            bp_init,
            grad_out[:, :-1].flip(1),
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

    @staticmethod
    def jvp(
        ctx: Any,
        grad_a: Optional[Tensor],
        grad_init: Optional[Tensor],
        grad_x: Optional[Tensor],
    ) -> Tensor:
        a, init, out = ctx.saved_tensors

        fwd_init = grad_init if grad_init is not None else torch.zeros_like(init)
        fwd_x = grad_x if grad_x is not None else torch.zeros_like(out)

        if grad_a is not None:
            concat_out = torch.cat([init.unsqueeze(1), out[:, :-1]], dim=1)
            fwd_a = concat_out * grad_a.view(-1, 1)
            fwd_x = fwd_x + fwd_a

        return Recurrence.apply(a, fwd_init, fwd_x)


def _scalar_recursion_loop(
    a: Tensor,
    init: Tensor,
    x: Tensor,
) -> Tensor:
    results = []
    h = init
    for xn in x.unbind(dim=1):
        h = a * h + xn
        results.append(h)
    return torch.stack(results, dim=1)


def linear_recurrence(
    a: Tensor, init: Tensor, x: Tensor, *, unroll_factor: int = 1
) -> Tensor:
    assert x.dim() == 2, f"Input x must be 2D, got {x.shape}"
    assert a.dim() in (0, 1), f"State matrix a must be 1D or 0D, got {a.shape}"
    if a.dim() == 1 and a.size(0) > 1 and a.size(0) != x.size(0):
        raise ValueError(
            f"State matrix a must be 1D with the same batch size as x, got a: {a.size(0)}, x: {x.size(0)}"
        )

    if a.dim() == 0:
        a = a.expand(1)

    batch_size, N = x.size(0), x.size(1)

    assert init.dim() in (
        0,
        1,
    ), f"Initial state init must be 1D or 0D, got {init.shape}"
    if init.dim() == 1 and init.size(0) > 1 and init.size(0) != batch_size:
        raise ValueError(
            f"Initial state init must be 1D with the same batch size as x, got init: {init.size(0)}, x: {batch_size}"
        )

    if init.dim() == 0:
        init = init.expand(1)

    if unroll_factor < 1:
        raise ValueError("Unroll factor must be >= 1")
    else:
        block_size = unroll_factor

    # boundary condition
    if block_size == 1 or block_size >= N:
        return _scalar_recursion_loop(a, init, x)

    remainder = N % block_size
    if remainder != 0:
        x = F.pad(x, (0, block_size - remainder))
        N = x.size(1)  # Update T after padding

    unrolled_x = x.unflatten(1, (-1, block_size))

    a_powers = torch.cumprod(
        (
            a.expand(block_size)
            if a.numel() == 1
            else a.unsqueeze(1).expand(-1, block_size)
        ),
        dim=-1,
    )
    a_powered = a_powers[..., -1]
    a_powers_plus_I = F.pad(a_powers[..., :-1].flip(-1), (0, 1), value=1.0)

    z = (
        unrolled_x @ a_powers_plus_I
        if a_powers_plus_I.dim() == 1
        else torch.linalg.vecdot(unrolled_x.conj(), a_powers_plus_I.unsqueeze(1))
    )

    initials = torch.cat(
        [
            init.unsqueeze(1).broadcast_to(batch_size, 1),
            linear_recurrence(a_powered, init, z, unroll_factor=unroll_factor),
        ],
        dim=1,
    )

    # prepare the augmented matrix and input for all the remaining steps
    aug_x = torch.cat([initials[:, :-1, None], unrolled_x[..., :-1]], dim=2)
    aug_A = (
        F.pad(a_powers_plus_I, (0, block_size - 2), value=0.0)
        .unfold(-1, block_size, 1)
        .flip(-2)
    )

    output = aug_x @ aug_A.mT

    # concat the first M - 1 outputs with the last one
    output = torch.cat([output, initials[:, 1:, None]], dim=2).flatten(1, 2)
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output
