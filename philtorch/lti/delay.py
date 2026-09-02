from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor

from .ssm import _ssm_C_D


def _delay_state_input(B: Optional[Tensor], x: Tensor, batch_size: int, M: int):
    features = -1 if x.dim() == 2 else x.size(-1)

    if B is None:
        if x.dim() == 2:
            return torch.cat(
                [x.unsqueeze(-1), x.new_zeros(batch_size, x.size(1), M - 1)],
                dim=-1,
            )
        assert (
            features == M
        ), f"Last dimension of x must match the number of delays when B is None, got x: {features}, delays: {M}"
        return x

    match B.shape:
        case (BM,) if BM == M:
            assert (
                x.dim() == 2
            ), f"Input signal x must be 2D when B is of shape {M,}, got {x.shape}"
            return x.unsqueeze(-1) * B
        case (B_batch, BM) if B_batch == batch_size and BM == M:
            assert (
                x.dim() == 2
            ), f"Input signal x must be 2D when B is of shape {batch_size, M}, got {x.shape}"
            return x.unsqueeze(-1) * B.unsqueeze(1)
        case (BM, F) if BM == M and F == features:
            return x @ B.mT
        case (B_batch, BM, F) if B_batch == batch_size and BM == M and F == features:
            return x @ B.mT
        case _:
            raise ValueError(
                f"Input matrix B must be of shape ({M},), ({batch_size, M}), ({M, features}), or ({batch_size, M, features}), got {B.shape}"
            )


def delay_state_space(
    A: Tensor,
    x: Tensor,
    delays: Sequence[int],
    B: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    D: Optional[Tensor] = None,
    zi: Optional[Sequence[Tensor]] = None,
    block_size: Optional[int] = None,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Compute a structured state-space model with explicit delay lines.

    For delay lengths ``m_i``, this evaluates

        r_i[n] = s_i[n - m_i]
        s[n] = A @ r[n] + B @ x[n]
        y[n] = C @ r[n] + D @ x[n]

    Each delay state is an output-first queue, so ``zi[i][..., 0]`` is the
    next value emitted by delay line ``i``. The sequence is processed in
    blocks no longer than the shortest delay. All queue updates are
    out-of-place, which keeps the implementation compatible with autograd.

    Args:
        A (Tensor): Feedback matrix with shape ``(M, M)`` or ``(B, M, M)``.
        x (Tensor): Input sequence with shape ``(B, N)`` or ``(B, N, F)``.
        delays (Sequence[int]): Positive delay lengths for the ``M`` lines.
        B (Tensor, optional): Input matrix using the same shape conventions as
            :func:`state_space`. If omitted, scalar input enters the first
            delay line, or vector input must have ``M`` features.
        C (Tensor, optional): Output matrix using the same shape conventions as
            :func:`state_space`. If omitted, all delay outputs are returned.
        D (Tensor, optional): Direct matrix using the same shape conventions as
            :func:`state_space`.
        zi (Sequence[Tensor], optional): One initial queue per delay line. Each
            queue has shape ``(m_i,)`` or ``(B, m_i)``. Missing states are zero.
        block_size (int, optional): Processing block length. It must be no
            greater than ``min(delays)`` and defaults to that value.

    Returns:
        tuple: ``(y, zf)`` where ``zf`` is a tuple containing one final queue
        per delay line.
    """
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"
    assert A.dim() in (2, 3), f"State matrix A must be 2D or 3D, got {A.shape}"
    assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"

    delays = tuple(delays)
    if not delays:
        raise ValueError("delays must contain at least one delay line")
    if any(
        not isinstance(delay, int) or isinstance(delay, bool) or delay < 1
        for delay in delays
    ):
        raise ValueError("Every delay must be a positive integer")

    batch_size, samples, *_ = x.shape
    M = len(delays)
    assert (
        A.size(-1) == M
    ), f"Last dimension of A must match the number of delays, got A: {A.size(-1)}, delays: {M}"
    if A.dim() == 3:
        assert (
            A.size(0) == batch_size
        ), f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {batch_size}"

    if block_size is None:
        block_size = min(delays)
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise ValueError("block_size must be an integer")
    if block_size < 1 or block_size > min(delays):
        raise ValueError("block_size must satisfy 1 <= block_size <= min(delays)")

    Bx = _delay_state_input(B, x, batch_size, M)

    if zi is None:
        states = tuple(x.new_zeros(batch_size, delay) for delay in delays)
    else:
        if isinstance(zi, Tensor) or len(zi) != M:
            raise ValueError(f"zi must contain one state for each of the {M} delays")
        expanded_states = []
        for index, (state, delay) in enumerate(zip(zi, delays)):
            if not isinstance(state, Tensor):
                raise TypeError(f"zi[{index}] must be a Tensor")
            assert state.dim() in (
                1,
                2,
            ), f"Initial delay state zi[{index}] must be 1D or 2D, got {state.shape}"
            assert (
                state.size(-1) == delay
            ), f"Last dimension of zi[{index}] must match delay {delay}, got {state.size(-1)}"
            if state.dim() == 1:
                state = state.unsqueeze(0).expand(batch_size, -1)
            else:
                assert (
                    state.size(0) == batch_size
                ), f"Batch size of zi[{index}] must match batch size of x, got zi: {state.size(0)}, x: {batch_size}"
            expanded_states.append(state)
        states = tuple(expanded_states)

    if samples == 0:
        delay_outputs = x.new_empty(batch_size, 0, M)
        return _ssm_C_D(delay_outputs, x, C, D, batch_size, M), states

    outputs = []
    for start in range(0, samples, block_size):
        length = min(block_size, samples - start)
        delay_outputs = torch.stack([state[:, :length] for state in states], dim=-1)
        delay_inputs = delay_outputs @ A.mT + Bx[:, start : start + length]
        outputs.append(
            _ssm_C_D(
                delay_outputs,
                x[:, start : start + length],
                C,
                D,
                batch_size,
                M,
            )
        )
        states = tuple(
            torch.cat([state[:, length:], delay_inputs[:, :, index]], dim=-1)
            for index, state in enumerate(states)
        )

    return torch.cat(outputs, dim=1), states
