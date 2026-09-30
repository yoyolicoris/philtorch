import operator
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from torch import Tensor

from .ssm import _ssm_B, _ssm_C_D

_OUTPUT_CHUNK = 4096


def _as_int(value: Any, message: str) -> int:
    """Convert Python, NumPy, or single-element integer tensor scalars to ``int``."""
    if isinstance(value, bool) or (
        isinstance(value, Tensor) and value.dtype == torch.bool
    ):
        raise ValueError(message)
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(message) from None


def _read_delay_line(
    initial: Tensor,
    blocks: list[Optional[Tensor]],
    block_size: int,
    delay: int,
    line: int,
    start: int,
    stop: int,
) -> Tensor:
    """Read positions ``[start, stop)`` of ``initial ++ s_line``.

    Blocks have shape ``(B, M, length)`` and every block except the last has
    ``block_size`` samples, so position ``p >= delay`` lives at
    ``divmod(p - delay, block_size)``.
    """
    parts = []
    if start < delay:
        parts.append(initial[:, start : min(stop, delay)])
        start = delay
    while start < stop:
        block, offset = divmod(start - delay, block_size)
        length = min(stop - start, block_size - offset)
        parts.append(blocks[block][:, line, offset : offset + length])
        start += length
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def delay_state_space(
    A: Tensor,
    x: Tensor,
    delays: Union[int, Sequence[int], Tensor],
    B: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    D: Optional[Tensor] = None,
    zi: Optional[Sequence[Tensor]] = None,
    block_size: Optional[int] = None,
    out_idx: Optional[int] = None,
):
    """Compute a structured state-space model with explicit delay lines.

    For delay lengths ``m_i``, this evaluates

        r_i[n] = s_i[n - m_i]
        s[n] = A @ r[n] + B @ x[n]
        y[n] = C @ r[n] + D @ x[n]

    With all delays equal to one this is :func:`state_space` with ``h = r``;
    ``B``, ``C``, ``D``, ``out_idx`` and the return value follow the same
    conventions. Each delay state is an output-first queue, so
    ``zi[i][..., 0]`` is the next value emitted by delay line ``i``. The
    sequence is processed in blocks no longer than the shortest delay, so each
    block only reads values written by earlier blocks.

    Args:
        A (Tensor): Feedback matrix with shape ``(M, M)`` or ``(B, M, M)``.
        x (Tensor): Input sequence with shape ``(B, N)`` or ``(B, N, F)``.
        delays (Sequence[int] or Tensor): Positive integer delay lengths for
            the ``M`` lines. Python, NumPy, and integer tensor values are
            accepted; a scalar gives a single delay line.
        B (Tensor, optional): Input matrix with the same shapes as in
            :func:`state_space`. If omitted, scalar input enters the first
            delay line, or vector input must have ``M`` features.
        C (Tensor, optional): Output matrix with the same shapes as in
            :func:`state_space`. If omitted, all delay outputs are returned.
        D (Tensor, optional): Direct matrix with the same shapes as in
            :func:`state_space`.
        zi (Sequence[Tensor], optional): One initial queue per delay line. Each
            queue has shape ``(m_i,)`` or ``(B, m_i)``. Zero when omitted.
        block_size (int, optional): Processing block length. It must be no
            greater than ``min(delays)`` and defaults to that value.
        out_idx (int, optional): If provided, return only this delay line's
            output per timestep. Cannot be combined with ``C``.

    Returns:
        Tensor or 2-tuple ``(y, zf)`` when ``zi`` is provided, where ``zf``
        is a tuple containing one final queue per delay line.
    """
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"
    assert A.dim() in (2, 3), f"State matrix A must be 2D or 3D, got {A.shape}"
    assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
    if not (C is None or out_idx is None):
        raise ValueError(
            "C and out_idx cannot be used together. Use either C or out_idx."
        )

    # A scalar (int, NumPy scalar, or 0-d array/tensor) is a single delay line.
    if isinstance(delays, int) or getattr(delays, "ndim", None) == 0:
        delays = (delays,)
    delays = tuple(
        _as_int(delay, "Every delay must be a positive integer") for delay in delays
    )
    if not delays:
        raise ValueError("delays must contain at least one delay line")
    if any(delay < 1 for delay in delays):
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
    block_size = _as_int(block_size, "block_size must be an integer")
    if block_size < 1 or block_size > min(delays):
        raise ValueError("block_size must satisfy 1 <= block_size <= min(delays)")

    if B is None and x.dim() == 3:
        assert (
            x.size(-1) == M
        ), f"Last dimension of x must match the number of delays when B is None, got x: {x.size(-1)}, delays: {M}"
    Bx = _ssm_B(B, x, batch_size, M)
    if Bx.dim() == 2:
        Bx = torch.cat([Bx.unsqueeze(-1), Bx.new_zeros(batch_size, samples, M - 1)], -1)

    return_zf = zi is not None
    if zi is None:
        initial = tuple(x.new_zeros(batch_size, delay) for delay in delays)
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
        initial = tuple(expanded_states)

    def read(line: int, start: int, stop: int) -> Tensor:
        return _read_delay_line(
            initial[line], blocks, block_size, delays[line], line, start, stop
        )

    blocks: list[Optional[Tensor]] = []
    history = max(delays)
    outputs, pending, pending_start = [], [], 0
    for start in range(0, samples, block_size):
        stop = min(start + block_size, samples)
        r = torch.stack([read(line, start, stop) for line in range(M)], dim=-1)
        blocks.append((r @ A.mT + Bx[:, start:stop]).mT.contiguous())
        # Release blocks older than the longest delay so their memory is reused.
        expired = (start - history) // block_size
        if expired > 0:
            blocks[expired - 1] = None
        pending.append(r if out_idx is None else r[..., out_idx])
        # Map delay outputs to y in chunks: per block is slow for short blocks,
        # and all at once materialises a (B, N, M) tensor.
        if stop - pending_start >= _OUTPUT_CHUNK or stop == samples:
            h = pending[0] if len(pending) == 1 else torch.cat(pending, dim=1)
            outputs.append(_ssm_C_D(h, x[:, pending_start:stop], C, D, batch_size, M))
            pending, pending_start = [], stop

    if outputs:
        y = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
    else:
        h = x.new_empty(batch_size, 0, M)
        if out_idx is not None:
            h = h[..., out_idx]
        y = _ssm_C_D(h, x, C, D, batch_size, M)

    if return_zf:
        zf = tuple(read(line, samples, samples + delays[line]) for line in range(M))
        return y, zf
    return y
