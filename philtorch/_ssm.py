"""The state-space recursion of :mod:`philtorch.lti` and :mod:`philtorch.lpv`.

:func:`recursion` computes h[n + 1] = A[n] h[n] + x[n] from h[0] = zi and
returns h[1], ..., h[N], with A of shape (B or 1, T or 1, M, M) as
:func:`philtorch._recur.recurrence` takes it: per signal or shared, per step or
time-invariant. The modules validate their own layouts of A and turn them
into this one; :func:`states` is the part of their ``state_space`` between
the inputs' mapping, B x, and the outputs', C h + D x.

``unroll_factor`` picks the algorithm:

* 1: the operator's kernels, raising NotImplementedError where none takes
  the input;
* N or more: the loop;
* in between: a block-unrolled recursion in PyTorch ops. Time-invariant A
  composes its powers once; time-varying A composes cumulative products per
  block. Each runs on its module's own layout of A, which it was written for.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .mat import matrices_cumdot, matrix_power_accumulate


def from_lti(A: Tensor) -> Tensor:
    """(M, M) or (B, M, M) time-invariant matrices, as (1 or B, 1, M, M)."""
    return A[None, None] if A.dim() == 2 else A[:, None]


def from_lpv(A: Tensor) -> Tensor:
    """(T, M, M) or (B, T, M, M) time-varying matrices, as (1 or B, T, M, M)."""
    return A[None] if A.dim() == 3 else A


def to_lti(A: Tensor) -> Tensor:
    """(1 or B, 1, M, M) matrices as time-invariant ones, (M, M) or (B, M, M)."""
    return A[0, 0] if A.size(0) == 1 else A[:, 0]


def to_lpv(A: Tensor) -> Tensor:
    """(1 or B, T, M, M) matrices as time-varying ones, (T, M, M) or (B, T, M, M)."""
    return A[0] if A.size(0) == 1 else A


def _first_state(x: Tensor, M: int) -> Tensor:
    """A 2-D input (B, T) feeding the first of M states, as (B, T, M)."""
    return F.pad(x.unsqueeze(-1), (0, M - 1)) if x.dim() == 2 else x


def _loop(A: Tensor, zi: Tensor, x: Tensor, out_idx: int | None = None) -> Tensor:
    """The recursion by a loop over time steps; see :func:`recursion`."""
    Ba, Ta, M, _ = A.shape
    x = _first_state(x, M)
    if x.size(1) == 0:
        # From the inputs' empty slices, so that it takes their promoted dtype
        # and carries their gradients, all zero.
        h = x + zi[:, None][:, :0] + A[:, :0].sum(-1)
        return h if out_idx is None else h[..., out_idx]
    AT = A.mT
    results = []
    h = zi if Ba == 1 else zi.unsqueeze(1)
    for n, xn in enumerate(x.unbind(1)):
        AnT = AT[:, n if Ta > 1 else 0]
        if Ba == 1:
            h = torch.addmm(xn, h, AnT[0])
            results.append(h if out_idx is None else h[:, out_idx])
        else:
            h = torch.baddbmm(xn.unsqueeze(1), h, AnT)
            results.append(h[:, 0] if out_idx is None else h[:, 0, out_idx])
    return torch.stack(results, dim=1)


# Appended to the operator's error where no kernel takes the input.
_NO_KERNEL_HINT = (
    "; pass unroll_factor > 1 to run PyTorch ops instead, on any device: a "
    "block-unrolled recursion, or, from the sequence length on, the sequential loop, "
    "in the dtypes PyTorch's matrix products take there (on CUDA, no integers)"
)


def _recurrence(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    """:func:`philtorch._recur.recurrence`, its error for want of a kernel saying
    how to run PyTorch ops instead."""
    # Imported here, so that importing philtorch doesn't load Triton.
    from ._recur import recurrence

    try:
        return recurrence(A, zi, x)
    except NotImplementedError as error:
        raise NotImplementedError(f"{error}{_NO_KERNEL_HINT}") from error


def _kernels(A: Tensor, zi: Tensor, x: Tensor, out_idx: int | None = None) -> Tensor:
    """The recursion by :func:`philtorch._recur.recurrence`'s kernels; see
    :func:`recursion`."""
    h = _recurrence(A, zi, _first_state(x, A.size(-1)))
    return h if out_idx is None else h[..., out_idx]


def _unrolled_time_invariant(
    A: Tensor, zi: Tensor, x: Tensor, block_size: int, out_idx: int | None
) -> Tensor:
    """The block-unrolled recursion for A of shape (M, M) or (B, M, M)."""
    N = x.size(1)
    M = A.size(-1)
    remainder = N % block_size
    if remainder != 0:
        x = F.pad(x, (0, 0) * (x.dim() - 2) + (0, block_size - remainder))
        N = x.size(1)  # Update T after padding

    unrolled_x = x.unflatten(1, (-1, block_size)).flatten(2, -1)

    A_powers = matrix_power_accumulate(A, block_size)
    A_powered = A_powers[..., -1, :, :]
    A_powers_plus_I = torch.cat(
        [
            A_powers[..., :-1, :, :].flip(-3),
            torch.eye(M, device=A.device, dtype=A.dtype).broadcast_to(A.shape).unsqueeze(-3),
        ],
        dim=-3,
    )

    mat1 = (
        A_powers_plus_I.transpose(-2, -1).flatten(-3, -2)
        if x.dim() == 3
        else A_powers_plus_I[..., 0]  # assume x -> x * [1, 0, 0, ...] in the 2D case
    )
    z = unrolled_x @ mat1

    initials = torch.cat(
        [
            zi.unsqueeze(1),
            recursion(from_lti(A_powered), zi, z, unroll_factor=block_size),
        ],
        dim=1,
    )

    # prepare the augmented matrix and input for all the remaining steps
    aug_x = torch.cat([initials[:, :-1], unrolled_x[..., : -(1 if x.dim() == 2 else M)]], dim=2)

    if out_idx is None:
        mat2 = A_powers[..., :-1, :, :].flatten(-3, -2)
        if x.dim() == 3:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, :, :],
                        A_powers_plus_I.new_zeros(
                            A_powers_plus_I.shape[:-3] + (block_size - 2, M, M)
                        ),
                    ],
                    dim=-3,
                )
                .unfold(-3, block_size - 1, 1)
                .transpose(-2, -1)
                .flip(-4)
                .flatten(-4, -3)
                .flatten(-2, -1)
            )
        else:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, :, 0],
                        A_powers_plus_I.new_zeros(A_powers_plus_I.shape[:-3] + (block_size - 2, M)),
                    ],
                    dim=-2,
                )
                .unfold(-2, block_size - 1, 1)
                .flip(-3)
                .flatten(-3, -2)
            )
    else:
        mat2 = A_powers[..., :-1, out_idx, :]
        if x.dim() == 3:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, out_idx, :],
                        A_powers_plus_I.new_zeros(A_powers_plus_I.shape[:-3] + (block_size - 2, M)),
                    ],
                    dim=-2,
                )
                .unfold(-2, block_size - 1, 1)
                .transpose(-2, -1)
                .flip(-3)
                .flatten(-2, -1)
            )
        else:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, out_idx, 0],
                        A_powers_plus_I.new_zeros(A_powers_plus_I.shape[:-3] + (block_size - 2,)),
                    ],
                    dim=-1,
                )
                .unfold(-1, block_size - 1, 1)
                .flip(-2)
            )

    aug_A = torch.cat([mat2.mT, mat3.mT], dim=-2)
    output = aug_x @ aug_A

    # concat the first M - 1 outputs with the last one
    if out_idx is None:
        output = torch.cat([output, initials[:, 1:, :]], dim=2).unflatten(2, (-1, M)).flatten(1, 2)
    else:
        output = torch.cat([output, initials[:, 1:, out_idx, None]], dim=2).flatten(1, 2)
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output


def _unrolled_time_varying(
    A: Tensor, zi: Tensor, x: Tensor, block_size: int, out_idx: int | None
) -> Tensor:
    """The block-unrolled recursion for A of shape (N, M, M) or (B, N, M, M)."""
    batch_size, N = x.size(0), x.size(1)
    M = A.size(-1)
    remainder = N % block_size
    if remainder != 0:
        x = F.pad(x, (0, 0) * (x.dim() - 2) + (0, block_size - remainder))
        A = F.pad(A, (0,) * 4 + (0, block_size - remainder))  # pad
        N = x.size(1)  # Update T after padding

    unrolled_x = x.unflatten(1, (-1, block_size))
    unrolled_x_flatten = unrolled_x.flatten(2, -1)
    unrolled_A = A.unflatten(-3, (-1, block_size))

    A_cums = matrices_cumdot(unrolled_A[..., 1:, :, :].flip(-3)).flip(-3)
    A_last_cum = A_cums[..., 0, :, :] @ unrolled_A[..., 0, :, :]
    A_cums_plus_I = torch.cat(
        [
            A_cums,
            torch.eye(M, device=A.device, dtype=A.dtype).broadcast_to(
                A_cums.shape[:-3] + (1, M, M)
            ),
        ],
        dim=-3,
    )
    mat1 = (
        A_cums_plus_I.mT.flatten(-3, -2)
        if x.dim() == 3
        else A_cums_plus_I[..., 0]  # assume x -> x * [1, 0, 0, ...] in the 2D case
    )
    z = torch.squeeze(unrolled_x_flatten.unsqueeze(-2) @ mat1, -2)

    initials = torch.cat(
        [
            zi.unsqueeze(1),
            recursion(from_lpv(A_last_cum), zi, z, unroll_factor=block_size),
        ],
        dim=1,
    )

    # The blocks' steps after the first, each block a sequence of its own.
    output = _loop(
        (
            unrolled_A[:, :, :-1].flatten(0, 1)
            if unrolled_A.dim() == 5
            else unrolled_A[:, :-1].repeat(batch_size, 1, 1, 1)
        ),
        initials[:, :-1].flatten(0, 1),
        unrolled_x[:, :, :-1].flatten(0, 1),
        out_idx=out_idx,
    ).unflatten(0, (batch_size, unrolled_x.size(1)))

    # concat the first M - 1 outputs with the last one
    if out_idx is None:
        output = torch.cat([output, initials[:, 1:, None, :]], dim=2).flatten(1, 2)
    else:
        output = torch.cat([output, initials[:, 1:, out_idx, None]], dim=2).flatten(1, 2)
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output


def recursion(
    A: Tensor, zi: Tensor, x: Tensor, *, unroll_factor: int = 1, out_idx: int | None = None
) -> Tensor:
    """h[1], ..., h[N] of h[n + 1] = A[n] h[n] + x[n], h[0] = zi.

    Args:
        A: (B or 1, T or 1, M, M).
        zi: (B, M).
        x: (B, N, M), or (B, N) to feed the first state only.
        unroll_factor: the algorithm; see the module docstring.
        out_idx: return only this state.

    Returns:
        (B, N, M), or (B, N) with ``out_idx``.

    Raises:
        ValueError: if ``unroll_factor`` is less than 1.
        NotImplementedError: with ``unroll_factor=1``, if no kernel takes the
            input.
    """
    if unroll_factor < 1:
        raise ValueError("Unroll factor must be >= 1")
    if unroll_factor == 1:
        return _kernels(A, zi, x, out_idx=out_idx)
    if unroll_factor >= x.size(1):
        return _loop(A, zi, x, out_idx=out_idx)
    if A.size(1) == 1:
        return _unrolled_time_invariant(to_lti(A), zi, x, unroll_factor, out_idx)
    return _unrolled_time_varying(to_lpv(A), zi, x, unroll_factor, out_idx)


def states(
    run: Callable[..., Tensor], zi: Tensor, *, out_idx: int | None, final_state: bool
) -> tuple[Tensor, Tensor | None]:
    """The states state_space reads its outputs from, h[0], ..., h[N - 1], and the
    final state h[N] when ``final_state``.

    Args:
        run: the module's state_space_recursion with its A, zi and inputs bound,
            taking ``out_idx``; it validates them in the module's layout.
        zi: the initial state h[0], (B, M).
        out_idx: return only this state.
        final_state: whether to return h[N]; otherwise ``None``.

    Returns:
        h[0], ..., h[N - 1], of shape (B, N, M), or (B, N) with ``out_idx``,
        and h[N], of shape (B, M), or ``None``; with N = 0, no states and zi.
    """
    if final_state or out_idx is None:
        h = run(out_idx=None)
        N = h.size(1)
        zf = (h[:, -1, :] if N else zi) if final_state else None
        if out_idx is None:
            return torch.cat([zi.unsqueeze(1), h[:, :-1]], dim=1)[:, :N], zf
        return torch.cat([zi[:, None, out_idx], h[:, :-1, out_idx]], dim=1)[:, :N], zf
    h = run(out_idx=out_idx)
    return torch.cat([zi[:, None, out_idx], h[:, :-1]], dim=1)[:, : h.size(1)], None


def outputs(Ch: Tensor, Dx: Tensor | None) -> Tensor:
    """state_space's outputs, C h + D x, each (B, N) for one output or (B, N, P) for P.

    Raises:
        ValueError: if D x gives a different number of outputs from C h, the
            states, or the state ``out_idx`` picks.
    """
    if Dx is None:
        return Ch
    if Ch.shape[2:] != Dx.shape[2:]:

        def count(t):
            return 1 if t.dim() == 2 else t.size(-1)

        raise ValueError(
            f"D gives {count(Dx)} outputs, but C, out_idx or the states give {count(Ch)}"
        )
    return Ch + Dx


# The scalar recursion, h[n + 1] = a[n] h[n] + x[n], of lti and lpv's
# linear_recurrence: a of shape (1 or B, 1 or N), the scalar case of A's
# layout, init of shape (1 or B) and x of shape (B, N).


def _scalar_loop(a: Tensor, init: Tensor, x: Tensor) -> Tensor:
    """The scalar recursion along the last axis, by a loop over its steps; a, init
    and x broadcast, and a time-invariant a is expanded over the steps."""
    if x.size(-1) == 0:
        # As in _loop: from the inputs' empty slices.
        return x + a[..., :0] + init.unsqueeze(-1)[..., :0]
    a = a.expand(*a.shape[:-1], x.size(-1))
    results = []
    h = init
    for an, xn in zip(a.unbind(dim=-1), x.unbind(dim=-1), strict=True):
        h = torch.addcmul(xn, h, an)
        results.append(h)
    return torch.stack(results, dim=-1)


def _scalar_kernels(a: Tensor, init: Tensor, x: Tensor) -> Tensor:
    """The scalar recursion by :func:`philtorch._recur.recurrence`'s kernels."""
    init = init.broadcast_to(x.size(0))
    return _recurrence(a[..., None, None], init[:, None], x[..., None])[..., 0]


def _scalar_unrolled_time_invariant(a: Tensor, init: Tensor, x: Tensor, block_size: int) -> Tensor:
    """The block-unrolled scalar recursion for a of shape (1) or (B)."""
    batch_size, N = x.size(0), x.size(1)
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
            scalar_recursion(a_powered.reshape(-1, 1), init, z, unroll_factor=block_size),
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


def _scalar_unrolled_time_varying(a: Tensor, init: Tensor, x: Tensor, block_size: int) -> Tensor:
    """The block-unrolled scalar recursion for a of shape (N) or (B, N)."""
    batch_size, N = x.size(0), x.size(1)
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
            scalar_recursion(
                a_powered[None] if a_powered.dim() == 1 else a_powered,
                init,
                z,
                unroll_factor=block_size,
            ),
        ],
        dim=1,
    )

    # The blocks' steps after the first, along their last axis.
    output = _scalar_loop(unrolled_a[..., :-1], initials[:, :-1], unrolled_x[..., :-1])

    # concat the first M - 1 outputs with the last one
    output = torch.cat([output, initials[:, 1:, None]], dim=2).flatten(1, 2)
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output


def scalar_recursion(a: Tensor, init: Tensor, x: Tensor, *, unroll_factor: int = 1) -> Tensor:
    """h[1], ..., h[N] of h[n + 1] = a[n] h[n] + x[n], h[0] = init.

    Args:
        a: (1 or B, 1 or N).
        init: (1 or B).
        x: (B, N).
        unroll_factor: the algorithm, as for :func:`recursion`.

    Returns:
        (B, N).

    Raises:
        ValueError: if ``unroll_factor`` is less than 1.
        NotImplementedError: with ``unroll_factor=1``, if no kernel takes the
            input.
    """
    if unroll_factor < 1:
        raise ValueError("Unroll factor must be >= 1")
    if unroll_factor == 1:
        return _scalar_kernels(a, init, x)
    if unroll_factor >= x.size(1):
        return _scalar_loop(a, init, x)
    if a.size(1) == 1:
        return _scalar_unrolled_time_invariant(a[:, 0], init, x, unroll_factor)
    return _scalar_unrolled_time_varying(a[0] if a.size(0) == 1 else a, init, x, unroll_factor)
