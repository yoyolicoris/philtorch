"""State-space models with delay lines."""

import operator
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..mat import companion
from .ssm import _ssm_B, _ssm_C_D, state_space

_OUTPUT_CHUNK = 4096


def _as_int(value: Any, message: str) -> int:
    """Convert a Python, NumPy or one-element integer tensor scalar to ``int``."""
    # Booleans are integers to operator.index, so reject them first: Python
    # bools, bool tensors, and NumPy bools (dtype kind "b"), which NumPy
    # before 2.3 still converts to 0 or 1, with only a DeprecationWarning.
    if (
        isinstance(value, bool)
        or (isinstance(value, Tensor) and value.dtype == torch.bool)
        or getattr(getattr(value, "dtype", None), "kind", None) == "b"
    ):
        raise ValueError(message)
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(message) from None


def delay_state_space(
    A: Tensor,
    x: Tensor,
    delays: int | Sequence[int] | Tensor,
    B: Tensor | None = None,
    C: Tensor | None = None,
    D: Tensor | None = None,
    zi: Sequence[Tensor] | None = None,
    block_size: int | None = None,
    out_idx: int | None = None,
    absorption: tuple[Tensor, Tensor] | None = None,
    absorption_zi: Tensor | None = None,
    absorption_after_matrix: bool = False,
    unroll_factor: int | None = None,
):
    r"""Compute the outputs of a state-space model with delay lines.

    For delay lengths :math:`m_1, \dots, m_M`, this computes

    .. math::
        r_i[n] &= s_i[n - m_i], \\
        \mathbf{s}[n] &= A \mathbf{r}[n] + B \mathbf{x}[n], \\
        \mathbf{y}[n] &= C \mathbf{r}[n] + D \mathbf{x}[n].

    With every delay equal to 1, this is :func:`state_space` with
    :math:`\mathbf{h} = \mathbf{r}`; :attr:`B`, :attr:`C`, :attr:`D`,
    :attr:`out_idx` and the return value follow the same conventions. The
    sequence is processed in blocks no longer than the shortest delay, so each
    block only reads values written by earlier blocks.

    With :attr:`absorption`, each line also has an IIR filter :math:`h_i`, with
    the transfer function of :func:`lfilter`, for the loss of a feedback delay
    network. By default it filters the delay output, so :math:`r_i = h_i * q_i`
    for :math:`q_i[n] = s_i[n - m_i]`, and :math:`A`, :math:`C` and
    :attr:`out_idx` all see the filtered :math:`\mathbf{r}`. With
    ``absorption_after_matrix=True``, :math:`\mathbf{r} = \mathbf{q}` and only
    the feedback path is filtered:

    .. math::
        \mathbf{s}[n] = H(A \mathbf{r})[n] + B \mathbf{x}[n],

    where :math:`H` applies :math:`h_i` to entry :math:`i`. In both positions
    :math:`B \mathbf{x}` is not filtered.

    Args:
        A (Tensor): the feedback matrix, of shape :math:`(M, M)` or
            :math:`(B, M, M)`.
        x (Tensor): inputs, of shape :math:`(B, N)` or :math:`(B, N, F)`.
        delays (int, Sequence[int], or Tensor): the positive integer delay
            lengths of the :math:`M` lines, as Python, NumPy or integer tensor
            values; a scalar gives a single delay line.
        B (Tensor, optional): see :func:`state_space`. Default: ``None``: a
            2-D :attr:`x` enters the first delay line, and a 3-D one needs
            :math:`F = M`.
        C (Tensor, optional): see :func:`state_space`. Default: ``None``,
            which outputs every delay line.
        D (Tensor, optional): see :func:`state_space`. Default: ``None``.
        zi (Sequence[Tensor], optional): one initial queue per delay line, of
            shape :math:`(m_i)` or :math:`(B, m_i)`. Each queue is output first,
            so ``zi[i][..., 0]`` is the next value delay line :math:`i` emits.
            Default: ``None``, all zero.
        block_size (int, optional): the processing block length, at most
            :math:`\min_i m_i`. Default: ``None``, which uses
            :math:`\min_i m_i`.
        out_idx (int, optional): output only this delay line, instead of using
            :attr:`C`. Default: ``None``.
        absorption (tuple of Tensor, optional): the per-line filters
            :math:`(b, a)`, as in :func:`lfilter`, so :math:`a` omits the
            leading 1. :math:`b` has shape :math:`(M, M_b + 1)` or
            :math:`(B, M, M_b + 1)`, and :math:`a` has shape :math:`(M, M_a)`
            or :math:`(B, M, M_a)`, where :math:`M_a` may be 0. Batched and
            unbatched coefficients can be mixed. Default: ``None``, no
            filters.
        absorption_zi (Tensor, optional): the initial states of the filters,
            as the :attr:`zi` of :func:`lfilter` with ``form="tdf2"``, of
            shape :math:`(M, K)` or :math:`(B, M, K)`, where
            :math:`K = \max(M_a, M_b)`. Needs :attr:`absorption`.
            Default: ``None``, all zero.
        absorption_after_matrix (bool, optional): put the filters after
            :math:`A` instead of after the delay lines, as above. Needs
            :attr:`absorption`. Default: ``False``.
        unroll_factor (int, optional): how :func:`state_space` runs the
            filters: ``None`` runs native kernels, and an integer
            :math:`\ge 1` runs PyTorch ops on any device, with ``1`` the
            sequential loop. Needs :attr:`absorption`. Default: ``None``.

    Returns:
        Tensor or tuple: the outputs, as :func:`state_space`, and with
        :attr:`zi`, a tuple holding one final queue per delay line. By
        default, the queues hold the values before the filters:
        ``zi[i][..., 0]`` is the next value line :math:`i` emits before its
        filter. With ``absorption_after_matrix=True``, they hold the filtered
        feedback instead. With :attr:`absorption`, and :attr:`zi` or
        :attr:`absorption_zi`, a tuple ``(y, zf, absorption_zf)``, where
        ``absorption_zf`` is the final state of the filters, of shape
        :math:`(B, M, K)`, and whichever of the two initial states is not
        given starts at zero.

    Raises:
        ValueError: if a delay or :attr:`block_size` is invalid, if
            :attr:`zi` does not hold one queue per delay line, if both
            :attr:`C` and :attr:`out_idx` are given, if :attr:`D` gives a
            different number of outputs from :attr:`C`, :attr:`out_idx` or
            the states, if :attr:`absorption` is not a pair of tensors, if
            its coefficients or :attr:`absorption_zi` are not 2-D or 3-D or
            do not match :math:`M`, the batch size or :math:`K`, if
            :math:`b` is empty, if the coefficients or :attr:`absorption_zi`
            differ from :attr:`x` in dtype or device, if
            :attr:`unroll_factor` is not ``None`` or an integer of at least 1,
            if :attr:`absorption_zi`, :attr:`absorption_after_matrix` or
            :attr:`unroll_factor` is given without :attr:`absorption`, or if
            the other shapes do not match.
        TypeError: if a queue in :attr:`zi`, either coefficient in
            :attr:`absorption`, or :attr:`absorption_zi` is not a tensor.

    Example::

        >>> from philtorch.lti import delay_state_space
        >>> # One delay line of 3 samples that feeds back half its output.
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        >>> delay_state_space(torch.tensor([[0.5]]), x, [3]).squeeze(-1)
        tensor([[0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.5000]])
        >>> # The same, with a one-pole lowpass filter on the delay line.
        >>> absorption = torch.tensor([[0.5]]), torch.tensor([[-0.5]])
        >>> delay_state_space(
        ...     torch.tensor([[0.5]]), x, [3], absorption=absorption
        ... ).squeeze(-1)
        tensor([[0.0000, 0.0000, 0.0000, 0.5000, 0.2500, 0.1250, 0.1875]])
    """
    if x.dim() not in (
        2,
        3,
    ):
        raise ValueError(f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}")
    if A.dim() not in (2, 3):
        raise ValueError(f"State matrix A must be 2D or 3D, got {A.shape}")
    if A.size(-2) != A.size(-1):
        raise ValueError(f"State matrix A must be square, got {A.shape}")
    if not (C is None or out_idx is None):
        raise ValueError("C and out_idx cannot be used together. Use either C or out_idx.")
    if absorption is None:
        if absorption_zi is not None:
            raise ValueError("absorption_zi needs absorption")
        if absorption_after_matrix:
            raise ValueError("absorption_after_matrix needs absorption")
        if unroll_factor is not None:
            raise ValueError("unroll_factor needs absorption")
    elif unroll_factor is not None:
        unroll_factor = _as_int(
            unroll_factor, f"unroll_factor must be None or at least 1, got {unroll_factor}"
        )
        if unroll_factor < 1:
            raise ValueError(f"unroll_factor must be None or at least 1, got {unroll_factor}")

    # Anything that is not a sequence (int, NumPy scalar, 0-d array/tensor) is
    # a single delay line, so a non-integer scalar gets the ValueError below.
    if not isinstance(delays, Iterable) or getattr(delays, "ndim", None) == 0:
        delays = (delays,)
    delays = tuple(_as_int(delay, "Every delay must be a positive integer") for delay in delays)
    if not delays:
        raise ValueError("delays must contain at least one delay line")
    if any(delay < 1 for delay in delays):
        raise ValueError("Every delay must be a positive integer")

    batch_size, samples, *_ = x.shape
    M = len(delays)
    if A.size(-1) != M:
        raise ValueError(
            f"Last dimension of A must match the number of delays, got A: {A.size(-1)}, delays: {M}"
        )
    if A.dim() == 3:
        if A.size(0) != batch_size:
            raise ValueError(
                f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {batch_size}"
            )

    if block_size is None:
        block_size = min(delays)
    block_size = _as_int(block_size, "block_size must be an integer")
    if block_size < 1 or block_size > min(delays):
        raise ValueError("block_size must satisfy 1 <= block_size <= min(delays)")

    if B is None and x.dim() == 3:
        if x.size(-1) != M:
            raise ValueError(
                f"Last dimension of x must match the number of delays when B is None, "
                f"got x: {x.size(-1)}, delays: {M}"
            )
    Bx = _ssm_B(B, x, batch_size, M)

    absorb = absorption is not None
    if absorb:
        if not isinstance(absorption, (tuple, list)) or len(absorption) != 2:
            raise ValueError("absorption must be a pair of tensors (b, a)")
        for name, coefficients in zip(("b", "a"), absorption):
            if not isinstance(coefficients, Tensor):
                raise TypeError(f"absorption {name} must be a Tensor")
            if coefficients.dim() not in (2, 3):
                raise ValueError(f"absorption {name} must be 2D or 3D, got {coefficients.shape}")
            if coefficients.size(-2) != M:
                raise ValueError(
                    f"Second to last dimension of absorption {name} must match the number "
                    f"of delays, got {coefficients.size(-2)}, delays: {M}"
                )
            if coefficients.dim() == 3 and coefficients.size(0) != batch_size:
                raise ValueError(
                    f"Batch size of absorption {name} must match batch size of x, "
                    f"got {coefficients.size(0)}, x: {batch_size}"
                )
        if absorption[0].size(-1) == 0:
            raise ValueError("absorption b must have at least one coefficient")
        checked = {"b": absorption[0], "a": absorption[1]}
        if isinstance(absorption_zi, Tensor):
            checked["absorption_zi"] = absorption_zi
        if any(t.dtype != x.dtype or t.device != x.device for t in checked.values()):
            got = ", ".join(f"{name}: {t.dtype} on {t.device}" for name, t in checked.items())
            raise ValueError(
                "absorption b, a and absorption_zi must have the dtype and device of x, "
                f"got {got}, x: {x.dtype} on {x.device}"
            )
        filter_b, filter_a = (
            c.expand(batch_size, -1, -1) if c.dim() == 2 else c for c in absorption
        )
        filter_b = filter_b.reshape(batch_size * M, filter_b.size(-1))
        filter_a = filter_a.reshape(batch_size * M, filter_a.size(-1))
        K = max(filter_b.size(-1) - 1, filter_a.size(-1))
        # The transposed direct form II of lfilter, built once for all blocks.
        filter_b = F.pad(filter_b, (0, K + 1 - filter_b.size(-1)))
        filter_a = F.pad(filter_a, (0, K - filter_a.size(-1)))
        filter_D = filter_b[:, 0]
        filter_A = companion(filter_a).mT if K else None
        filter_B = filter_b[:, 1:] - filter_b[:, :1] * filter_a
        if absorption_zi is None:
            absorption_state = x.new_zeros(batch_size * M, K)
        else:
            if not isinstance(absorption_zi, Tensor):
                raise TypeError("absorption_zi must be a Tensor")
            if absorption_zi.dim() not in (2, 3):
                raise ValueError(f"absorption_zi must be 2D or 3D, got {absorption_zi.shape}")
            if absorption_zi.size(-2) != M:
                raise ValueError(
                    f"Second to last dimension of absorption_zi must match the number of "
                    f"delays, got {absorption_zi.size(-2)}, delays: {M}"
                )
            if absorption_zi.size(-1) != K:
                raise ValueError(
                    f"Last dimension of absorption_zi must be max(len(a), len(b) - 1) = {K}, "
                    f"got {absorption_zi.size(-1)}"
                )
            if absorption_zi.dim() == 3 and absorption_zi.size(0) != batch_size:
                raise ValueError(
                    f"Batch size of absorption_zi must match batch size of x, "
                    f"got absorption_zi: {absorption_zi.size(0)}, x: {batch_size}"
                )
            absorption_state = absorption_zi.expand(batch_size, -1, -1).reshape(batch_size * M, K)

    def absorb_block(signal: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Filter a (B, L, M) block line by line, from and to a (B * M, K) state."""
        if not K:
            # A pure gain has no state.
            return signal * filter_D.view(batch_size, 1, M), state
        length = signal.size(1)
        flat = signal.permute(0, 2, 1).reshape(batch_size * M, length)
        # unroll_factor=1 would make state_space warn that its meaning changed
        # in 0.8, but this argument is new. A factor of at least the block
        # length runs the same sequential loop.
        factor = length + 1 if unroll_factor == 1 else unroll_factor
        flat, state = state_space(
            filter_A,
            flat,
            B=filter_B,
            D=filter_D,
            zi=state,
            unroll_factor=factor,
            out_idx=0,
        )
        return flat.reshape(batch_size, M, length).permute(0, 2, 1), state

    return_zf = zi is not None
    return_absorption_zf = absorb and (return_zf or absorption_zi is not None)
    if zi is None:
        initial = tuple(x.new_zeros(batch_size, delay) for delay in delays)
    else:
        if isinstance(zi, Tensor) or len(zi) != M:
            raise ValueError(f"zi must contain one state for each of the {M} delays")
        expanded_states = []
        for index, (state, delay) in enumerate(zip(zi, delays)):
            if not isinstance(state, Tensor):
                raise TypeError(f"zi[{index}] must be a Tensor")
            if state.dim() not in (
                1,
                2,
            ):
                raise ValueError(
                    f"Initial delay state zi[{index}] must be 1D or 2D, got {state.shape}"
                )
            if state.size(-1) != delay:
                raise ValueError(
                    f"Last dimension of zi[{index}] must match delay {delay}, got {state.size(-1)}"
                )
            if state.dim() == 1:
                state = state.unsqueeze(0).expand(batch_size, -1)
            else:
                if state.size(0) != batch_size:
                    raise ValueError(
                        f"Batch size of zi[{index}] must match batch size of x, "
                        f"got zi: {state.size(0)}, x: {batch_size}"
                    )
            expanded_states.append(state)
        initial = tuple(expanded_states)

    # Each line reads the stream initial ++ s at position t - m_i for time t.
    # The stream is stored in chunks of block_size samples of all lines,
    # behind a prefix of `prefix` samples (a multiple of block_size) that
    # holds the initial queues right-aligned, so stream position t sits at
    # index t + prefix. At block `block`, line i reads block_size consecutive
    # positions from chunk `block` on, at the same offsets every time, in at
    # most two chunks. Only the `needed` chunks, relative to `block`, are
    # joined and read.
    history = max(delays)
    prefix = -(-history // block_size) * block_size
    queues = torch.stack(
        [F.pad(state, (prefix - delay, 0)) for state, delay in zip(initial, delays)], -1
    )
    chunks: list[Tensor | None] = list(queues.split(block_size, dim=1))
    needed = sorted(
        {(prefix - delay + k) // block_size for delay in delays for k in (0, block_size - 1)}
    )
    position = (
        torch.arange(block_size, device=x.device)[:, None]
        + prefix
        - torch.tensor(delays, device=x.device)
    )
    rank = torch.zeros(needed[-1] + 1, dtype=position.dtype, device=x.device)
    rank[needed] = torch.arange(len(needed), device=x.device)
    # (block_size, M) positions in the joined chunks.
    index = rank[position // block_size] * block_size + position % block_size
    lines = torch.arange(M, device=x.device)
    # The final queues are read from the last chunks, so keep those.
    keep_from = (
        (samples + prefix - history) // block_size if return_zf or return_absorption_zf else None
    )
    released = 0

    outputs, pending, pending_start = [], [], 0
    for block, start in enumerate(range(0, samples, block_size)):
        stop = min(start + block_size, samples)
        recent = [chunks[block + j] for j in needed]
        stream = recent[0] if len(needed) == 1 else torch.cat(recent, dim=1)
        # Advanced indexing saves only the index for backward, not the stream.
        r = stream[:, index[: stop - start], lines]
        # Release chunks no later block reads, so their memory is reused.
        release = block + 1 + needed[0]
        if keep_from is not None:
            release = min(release, keep_from)
        for c in range(released, release):
            chunks[c] = None
        released = max(released, release)
        if absorb and not absorption_after_matrix:
            r, absorption_state = absorb_block(r, absorption_state)
        feedback = r @ A.mT
        if absorb and absorption_after_matrix:
            feedback, absorption_state = absorb_block(feedback, absorption_state)
        if Bx.dim() == 2:
            # A 2-D Bx enters the first line only.
            feedback = feedback + F.pad(Bx[:, start:stop, None], (0, M - 1))
        else:
            feedback = feedback + Bx[:, start:stop]
        chunks.append(feedback)
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

    if return_zf or return_absorption_zf:
        first = (samples + prefix - history) // block_size
        tail = torch.cat(chunks[first:], dim=1)
        offset = samples + prefix - first * block_size
        zf = tuple(
            tail[:, offset - delay : offset, line].contiguous() for line, delay in enumerate(delays)
        )
        if absorb:
            return y, zf, absorption_state.reshape(batch_size, M, K)
        return y, zf
    return y
