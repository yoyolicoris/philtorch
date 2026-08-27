import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Optional

from .._torchlpc import ScanRecurrence


def _scalar_recursion_loop(
    a: Tensor,
    init: Tensor,
    x: Tensor,
) -> Tensor:
    results = []
    h = init
    for an, xn in zip(a.unbind(dim=-1), x.unbind(dim=-1)):
        h = torch.addcmul(xn, h, an)
        results.append(h)
    return torch.stack(results, dim=-1)


def linear_recurrence(
    a: Tensor, init: Tensor, x: Tensor, *, unroll_factor: int = 1
) -> Tensor:
    """A pure-Python implementation of a linear recurrence with time-varying
    coefficients.

    Implements elementwise recurrence h[t] = a[t] * h[t-1] + x[t] where `a`
    is a time-varying coefficient sequence. The function supports block
    unrolling to accelerate long recurrences.

    Args:
        a (Tensor): Coefficients with shape (N,) or (B, N).
        init (Tensor): Initial state (scalar or vector matching batch dims).
        x (Tensor): Input of shape (B, N).
        unroll_factor (int): Unroll factor for blocked processing. Must be >=1.
            1 = use native scan kernel when available; >1 = pure-PyTorch unrolled.

    Returns:
        Tensor: Output sequence of shape (B, N).
    """
    if unroll_factor < 1:
        raise ValueError(f"Unroll factor must be >= 1, got {unroll_factor}")

    # Validate x early
    assert x.dim() == 2, f"Input x must be 2D, got {x.shape}"
    batch_size, N = x.size(0), x.size(1)

    # Normalize a: allow (N,) -> (B,N) broadcast
    if a.dim() == 1:
        if a.size(0) != N:
            raise ValueError(
                f"State matrix a must be 1D with the same length as x, got a: {a.size(0)}, x: {x.size(1)}"
            )
        a = a.expand(batch_size, -1)
    assert a.dim() == 2, f"State matrix a must be 1D or 2D, got {a.shape}"
    assert a.shape == x.shape, f"a and x must have same shape after normalization, got a: {a.shape}, x: {x.shape}"

    # Normalize init: allow scalar 0-d -> (B,), (1,) -> (B,)
    if init.dim() == 0:
        init = init.expand(batch_size)
    elif init.dim() == 1:
        if init.size(0) == 1:
            init = init.expand(batch_size)
        elif init.size(0) != batch_size:
            raise ValueError(
                f"Initial state init must be 1D with the same batch size as x, got init: {init.size(0)}, x: {batch_size}"
            )
    else:
        raise AssertionError(f"Initial state init must be 1D or 0D, got {init.shape}")

    # Only use native scan when unroll_factor == 1 (respects explicit >1 request)
    if unroll_factor == 1:
        # ScanRecurrence expects (impulse, decay, init) = (x, a, init)
        return ScanRecurrence.apply(x, a, init)  # type: ignore[arg-type]

    assert x.dim() == 2, f"Input x must be 2D, got {x.shape}"
    assert a.dim() in (1, 2), f"State matrix a must be 1D or 2D, got {a.shape}"
    if a.dim() == 1 and a.size(0) != x.size(1):
        raise ValueError(
            f"State matrix a must be 1D with the same length as x, got a: {a.size(0)}, x: {x.size(1)}"
        )

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
        a = F.pad(a, (0, block_size - remainder))
        N = x.size(1)  # Update N after padding

    unrolled_x = x.unflatten(1, (-1, block_size))
    unrolled_a = a.unflatten(-1, (-1, block_size))

    a_powers = torch.cumprod(unrolled_a[..., 1:].flip(-1), dim=-1).flip(-1)
    a_powered = a_powers[..., 0] * unrolled_a[..., 0]
    a_powers_plus_I = F.pad(a_powers, (0, 1), value=1.0)

    z = torch.linalg.vecdot(unrolled_x.conj(), a_powers_plus_I)

    initials = torch.cat(
        [
            init.unsqueeze(1).broadcast_to(batch_size, 1),
            linear_recurrence(a_powered, init, z, unroll_factor=unroll_factor),
        ],
        dim=1,
    )

    output = _scalar_recursion_loop(
        unrolled_a[..., :-1], initials[:, :-1], unrolled_x[..., :-1]
    )

    # concat the first M - 1 outputs with the last one
    output = torch.cat([output, initials[:, 1:, None]], dim=2).flatten(1, 2)
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output
