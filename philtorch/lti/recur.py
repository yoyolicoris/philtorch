"""First-order linear recurrences with time-invariant coefficients."""

from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import functional as F


class LTIRecurrence(Function):
    r"""Autograd function for :math:`h[n] = a\, h[n - 1] + x[n]`.

    The forward pass runs the native ``lti_recur`` kernel. The backward pass
    runs the same recurrence backward in time with the conjugated
    coefficient, and the JVP runs it forward on the tangents, so both
    reverse- and forward-mode differentiation work.
    """

    @staticmethod
    def forward(a: Tensor, init: Tensor, x: Tensor) -> Tensor:
        return torch.ops.philtorch.lti_recur(a, init, x)

    @staticmethod
    def setup_context(ctx: Any, inputs: list[Any], output: Any) -> Any:
        a, init, _ = inputs
        ctx.save_for_backward(a, init, output)
        ctx.save_for_forward(a, init, output)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        a, init, out = ctx.saved_tensors
        grad_a = grad_x = grad_init = None

        bp_init = grad_out[:, -1]
        flipped_grad_x = torch.cat(
            [
                bp_init.unsqueeze(1),
                LTIRecurrence.apply(
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

    @staticmethod
    def jvp(
        ctx: Any,
        grad_a: Tensor | None,
        grad_init: Tensor | None,
        grad_x: Tensor | None,
    ) -> Tensor:
        a, init, out = ctx.saved_tensors

        fwd_init = grad_init if grad_init is not None else torch.zeros_like(init)
        fwd_x = grad_x if grad_x is not None else torch.zeros_like(out)

        if grad_a is not None:
            concat_out = torch.cat([init.unsqueeze(1), out[:, :-1]], dim=1)
            fwd_a = concat_out * grad_a.view(-1, 1)
            fwd_x = fwd_x + fwd_a

        return LTIRecurrence.apply(a, fwd_init, fwd_x)


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


def linear_recurrence(a: Tensor, init: Tensor, x: Tensor, *, unroll_factor: int = 1) -> Tensor:
    r"""Compute batched first-order linear recurrences.

    This computes

    .. math::
        h[n] = a\, h[n - 1] + x[n], \quad n = 0, \dots, N - 1,

    starting from :math:`h[-1] = \text{init}`.

    Args:
        a (Tensor): the coefficient, of shape :math:`()` to share it or
            :math:`(B)` for one per signal.
        init (Tensor): the initial value :math:`h[-1]`, of shape :math:`()` or
            :math:`(B)`.
        x (Tensor): input sequences, of shape :math:`(B, N)`.
        unroll_factor (int, optional): ``1`` runs the native kernel. A larger
            value runs a block-unrolled PyTorch recursion with blocks of this
            length, which parallelizes over blocks; see the README for
            guidance. Default: ``1``.

    Returns:
        Tensor: :math:`h[0], \dots, h[N - 1]`, of shape :math:`(B, N)`.

    Raises:
        ValueError: if :attr:`unroll_factor` is less than 1, or :attr:`a` or
            :attr:`init` has a batch size other than that of :attr:`x`.

    Example::

        >>> from philtorch.lti import linear_recurrence
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        >>> linear_recurrence(torch.tensor(0.5), torch.tensor(0.0), x)
        tensor([[1.0000, 0.5000, 0.2500, 0.1250]])
    """
    if unroll_factor == 1:
        return LTIRecurrence.apply(a.broadcast_to(x.shape[0]), init.broadcast_to(x.shape[0]), x)

    assert x.dim() == 2, f"Input x must be 2D, got {x.shape}"
    assert a.dim() in (0, 1), f"State matrix a must be 1D or 0D, got {a.shape}"
    if a.dim() == 1 and a.size(0) > 1 and a.size(0) != x.size(0):
        raise ValueError(
            f"State matrix a must be 1D with the same batch size as x, "
            f"got a: {a.size(0)}, x: {x.size(0)}"
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
            f"Initial state init must be 1D with the same batch size as x, "
            f"got init: {init.size(0)}, x: {batch_size}"
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
        (a.expand(block_size) if a.numel() == 1 else a.unsqueeze(1).expand(-1, block_size)),
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
        F.pad(a_powers_plus_I, (0, block_size - 2), value=0.0).unfold(-1, block_size, 1).flip(-2)
    )

    output = aug_x @ aug_A.mT

    # concat the first M - 1 outputs with the last one
    output = torch.cat([output, initials[:, 1:, None]], dim=2).flatten(1, 2)
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output
