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


def _initial_queues(
    zi: Sequence[Tensor] | None, delays: tuple[int, ...], batch_size: int, x: Tensor
) -> tuple[Tensor, ...]:
    """Check :attr:`zi` and return one ``(B, m_i)`` initial queue per line."""
    if zi is None:
        return tuple(x.new_zeros(batch_size, delay) for delay in delays)
    if isinstance(zi, Tensor) or len(zi) != len(delays):
        raise ValueError(f"zi must contain one state for each of the {len(delays)} delays")
    queues = []
    for index, (state, delay) in enumerate(zip(zi, delays)):
        if not isinstance(state, Tensor):
            raise TypeError(f"zi[{index}] must be a Tensor")
        if state.dim() not in (1, 2):
            raise ValueError(f"Initial delay state zi[{index}] must be 1D or 2D, got {state.shape}")
        if state.size(-1) != delay:
            raise ValueError(
                f"Last dimension of zi[{index}] must match delay {delay}, got {state.size(-1)}"
            )
        if state.dim() == 2 and state.size(0) != batch_size:
            raise ValueError(
                f"Batch size of zi[{index}] must match batch size of x, "
                f"got zi: {state.size(0)}, x: {batch_size}"
            )
        queues.append(state.expand(batch_size, -1))
    return tuple(queues)


def _absorption_filters(
    absorption: tuple[Tensor, Tensor],
    absorption_zi: Tensor | None,
    M: int,
    batch_size: int,
    x: Tensor,
) -> tuple[Tensor | None, Tensor, Tensor, int, Tensor]:
    """Check the absorption filters and build their transposed direct form II.

    Returns the state-space ``A``, ``B`` and ``D`` of the ``B * M`` filters,
    as in :func:`lfilter` with ``form="tdf2"``, their order ``K`` (``A`` is
    ``None`` when it is 0), and their ``(B * M, K)`` initial state.
    """
    if not isinstance(absorption, (tuple, list)) or len(absorption) != 2:
        raise ValueError("absorption must be a pair of tensors (b, a)")
    named = {"absorption b": absorption[0], "absorption a": absorption[1]}
    if absorption_zi is not None:
        named["absorption_zi"] = absorption_zi
    for name, tensor in named.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if tensor.dim() not in (2, 3):
            raise ValueError(f"{name} must be 2D or 3D, got {tensor.shape}")
        if tensor.size(-2) != M:
            raise ValueError(
                f"Second to last dimension of {name} must match the number of delays, "
                f"got {tensor.size(-2)}, delays: {M}"
            )
        if tensor.dim() == 3 and tensor.size(0) != batch_size:
            raise ValueError(
                f"Batch size of {name} must match batch size of x, "
                f"got {tensor.size(0)}, x: {batch_size}"
            )
    if absorption[0].size(-1) == 0:
        raise ValueError("absorption b must have at least one coefficient")
    if any(t.dtype != x.dtype or t.device != x.device for t in named.values()):
        got = ", ".join(f"{name}: {t.dtype} on {t.device}" for name, t in named.items())
        raise ValueError(
            "absorption b, a and absorption_zi must have the dtype and device of x, "
            f"got {got}, x: {x.dtype} on {x.device}"
        )

    b, a = (c.expand(batch_size, M, -1).reshape(batch_size * M, c.size(-1)) for c in absorption)
    K = max(b.size(-1) - 1, a.size(-1))
    if absorption_zi is None:
        state = x.new_zeros(batch_size * M, K)
    elif absorption_zi.size(-1) != K:
        raise ValueError(
            f"Last dimension of absorption_zi must be max(len(a), len(b) - 1) = {K}, "
            f"got {absorption_zi.size(-1)}"
        )
    else:
        state = absorption_zi.expand(batch_size, M, K).reshape(batch_size * M, K)
    b = F.pad(b, (0, K + 1 - b.size(-1)))
    a = F.pad(a, (0, K - a.size(-1)))
    A = companion(a).mT if K else None
    return A, b[:, 1:] - b[:, :1] * a, b[:, 0], K, state


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
        feedback plus :math:`B \mathbf{x}`, the values the lines emit. With
        :attr:`absorption`, and :attr:`zi` or :attr:`absorption_zi`, a tuple
        ``(y, zf, absorption_zf)``, where ``absorption_zf`` is the final
        state of the filters, of shape :math:`(B, M, K)`, and whichever of
        the two initial states is not given starts at zero.

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
    if x.dim() not in (2, 3):
        raise ValueError(f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}")
    if A.dim() not in (2, 3):
        raise ValueError(f"State matrix A must be 2D or 3D, got {A.shape}")
    if A.size(-2) != A.size(-1):
        raise ValueError(f"State matrix A must be square, got {A.shape}")
    if not (C is None or out_idx is None):
        raise ValueError("C and out_idx cannot be used together. Use either C or out_idx.")
    absorb = absorption is not None
    if not absorb:
        if absorption_zi is not None:
            raise ValueError("absorption_zi needs absorption")
        if absorption_after_matrix:
            raise ValueError("absorption_after_matrix needs absorption")
        if unroll_factor is not None:
            raise ValueError("unroll_factor needs absorption")
    elif unroll_factor is not None:
        message = f"unroll_factor must be None or at least 1, got {unroll_factor}"
        unroll_factor = _as_int(unroll_factor, message)
        if unroll_factor < 1:
            raise ValueError(message)

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
    if A.dim() == 3 and A.size(0) != batch_size:
        raise ValueError(
            f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {batch_size}"
        )

    if block_size is None:
        block_size = min(delays)
    block_size = _as_int(block_size, "block_size must be an integer")
    if block_size < 1 or block_size > min(delays):
        raise ValueError("block_size must satisfy 1 <= block_size <= min(delays)")

    if B is None and x.dim() == 3 and x.size(-1) != M:
        raise ValueError(
            f"Last dimension of x must match the number of delays when B is None, "
            f"got x: {x.size(-1)}, delays: {M}"
        )
    Bx = _ssm_B(B, x, batch_size, M)
    if absorb:
        filter_A, filter_B, filter_D, K, absorption_state = _absorption_filters(
            absorption, absorption_zi, M, batch_size, x
        )
    initial = _initial_queues(zi, delays, batch_size, x)
    return_states = zi is not None or absorption_zi is not None

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
        # TODO(#85): pass unroll_factor through once that warning is removed.
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

    # Line i reads the stream initial ++ s at position t - m_i for time t. The
    # stream is kept in chunks of block_size samples of all lines, behind a
    # prefix that holds the initial queues right-aligned. The prefix is the
    # shortest multiple of block_size covering the longest delay, so the
    # positions a block reads, relative to its first chunk (chunk `block`), are
    # the same for every block, and lie in at most two chunks per line.
    history = max(delays)
    prefix = -(-history // block_size) * block_size
    chunks: list[Tensor | None] = list(
        torch.stack(
            [F.pad(queue, (prefix - delay, 0)) for queue, delay in zip(initial, delays)], -1
        ).split(block_size, dim=1)
    )
    # Line i reads chunks (prefix - m_i) // block_size through
    # (prefix - m_i + block_size - 1) // block_size.
    needed = sorted(
        {(prefix - delay + k) // block_size for delay in delays for k in (0, block_size - 1)}
    )
    # (block_size, M) positions in the needed chunks, joined. They depend on
    # Python ints only, so they are built on the CPU and copied once.
    position = torch.arange(block_size)[:, None] + prefix - torch.tensor(delays)
    index = torch.searchsorted(torch.tensor(needed), position // block_size) * block_size
    index = (index + position % block_size).to(x.device, non_blocking=True)
    lines = torch.arange(M, device=x.device)
    # The final queues are the last `history` samples of the stream.
    tail = (samples + prefix - history) // block_size

    # Process spans of whole blocks of about _OUTPUT_CHUNK samples: a 2-D Bx,
    # which enters the first line only, is padded to (B, span, M), and the
    # delay outputs are mapped to y, a span at a time. Per block is slow for
    # short blocks, and all at once materialises (B, N, M) tensors.
    span = -(-_OUTPUT_CHUNK // block_size) * block_size
    outputs = []
    for span_start in range(0, samples, span):
        span_stop = min(span_start + span, samples)
        Bx_span = Bx[:, span_start:span_stop]
        if Bx_span.dim() == 2:
            Bx_span = F.pad(Bx_span.unsqueeze(-1), (0, M - 1))
        pending = []
        for start in range(span_start, span_stop, block_size):
            block = start // block_size
            stop = min(start + block_size, span_stop)
            joined = [chunks[block + j] for j in needed]
            stream = joined[0] if len(joined) == 1 else torch.cat(joined, dim=1)
            # Advanced indexing saves only the index for backward, not the stream.
            r = stream[:, index[: stop - start], lines]
            if block < tail:
                # No later block reads this chunk, so its memory is reused.
                chunks[block] = None
            if absorb and not absorption_after_matrix:
                r, absorption_state = absorb_block(r, absorption_state)
            feedback = r @ A.mT
            if absorb and absorption_after_matrix:
                feedback, absorption_state = absorb_block(feedback, absorption_state)
            chunks.append(feedback + Bx_span[:, start - span_start : stop - span_start])
            pending.append(r if out_idx is None else r[..., out_idx])
        h = pending[0] if len(pending) == 1 else torch.cat(pending, dim=1)
        outputs.append(_ssm_C_D(h, x[:, span_start:span_stop], C, D, batch_size, M))

    if outputs:
        y = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
    else:
        h = x.new_empty(batch_size, 0, M)
        if out_idx is not None:
            h = h[..., out_idx]
        y = _ssm_C_D(h, x, C, D, batch_size, M)

    if not return_states:
        return y
    stream = torch.cat(chunks[tail:], dim=1)
    end = samples + prefix - tail * block_size
    zf = tuple(stream[:, end - delay : end, line].contiguous() for line, delay in enumerate(delays))
    if absorb:
        return y, zf, absorption_state.reshape(batch_size, M, K)
    return y, zf
