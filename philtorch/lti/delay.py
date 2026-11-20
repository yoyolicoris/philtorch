"""State-space models with delay lines."""

import operator
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .. import _ssm
from ..utils import _check_shared
from .conversions import _sos2ss
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
        _check_shared(**{f"zi[{index}]": state}, x=x)
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
    absorption: Tensor,
    absorption_zi: Tensor | None,
    M: int,
    batch_size: int,
    x: Tensor,
) -> tuple[Tensor | tuple[Tensor, Tensor, Tensor, Tensor], Tensor | None]:
    """Check the absorption filters and build them.

    Returns the filters and their initial state. For gains, the filters are the
    gains, (B, 1, M), and the state is ``None``. For sections, the filters are
    the state-space matrices of the ``B * M`` cascades, and the state is their
    ``(B * M, 2K)`` initial state, for ``K`` sections.
    """
    if not isinstance(absorption, Tensor):
        raise TypeError("absorption must be a Tensor")
    if absorption.dim() not in (1, 2, 3, 4):
        raise ValueError(f"absorption must be 1D to 4D, got {absorption.shape}")
    gains = absorption.dim() <= 2
    # The lines are the last dimension of gains, the third to last of sections.
    lines = absorption.size(-1 if gains else -3)
    if lines != M:
        raise ValueError(
            f"The lines dimension of absorption must match the number of delays, "
            f"got {lines}, delays: {M}"
        )
    # Batched gains are 2-D and batched sections 4-D.
    if absorption.dim() % 2 == 0 and absorption.size(0) != batch_size:
        raise ValueError(
            f"Batch size of absorption must match batch size of x, "
            f"got {absorption.size(0)}, x: {batch_size}"
        )
    if gains:
        if absorption_zi is not None:
            raise ValueError("absorption_zi must be None for gains, which have no state")
        _check_shared(absorption=absorption, x=x)
        return absorption.expand(batch_size, M).unsqueeze(1), None

    K = absorption.size(-2)
    if absorption.size(-1) != 5:
        raise ValueError(
            f"The last dimension of absorption sections must be 5, got {absorption.size(-1)}"
        )
    if K == 0:
        raise ValueError("absorption must have at least one section")
    if absorption_zi is not None:
        if not isinstance(absorption_zi, Tensor):
            raise TypeError("absorption_zi must be a Tensor")
        if absorption_zi.dim() not in (3, 4):
            raise ValueError(f"absorption_zi must be 3D or 4D, got {absorption_zi.shape}")
        if absorption_zi.shape[-3:] != (M, K, 2):
            raise ValueError(
                f"The last three dimensions of absorption_zi must be ({M}, {K}, 2), "
                f"got {tuple(absorption_zi.shape[-3:])}"
            )
        if absorption_zi.dim() == 4 and absorption_zi.size(0) != batch_size:
            raise ValueError(
                f"Batch size of absorption_zi must match batch size of x, "
                f"got {absorption_zi.size(0)}, x: {batch_size}"
            )
    _check_shared(absorption=absorption, absorption_zi=absorption_zi, x=x)
    if absorption_zi is None:
        state = x.new_zeros(batch_size * M, 2 * K)
    else:
        state = absorption_zi.expand(batch_size, M, K, 2).reshape(batch_size * M, 2 * K)
    return _sos2ss(absorption.expand(batch_size, M, K, 5).reshape(batch_size * M, K, 5)), state


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
    absorption: Tensor | None = None,
    absorption_zi: Tensor | None = None,
    absorption_after_matrix: bool = False,
    unroll_factor: int | None = None,
):
    r"""Compute the outputs of a state-space model with delay lines.

    For delay lengths :math:`m_1, \dots, m_M`, with :math:`s_i[n]` the output
    of delay line :math:`i`, this computes the recursion of a feedback delay
    network,

    .. math::
        \mathbf{s}[n + \mathbf{m}] &= A \mathbf{s}[n] + B \mathbf{x}[n], \\
        \mathbf{y}[n] &= C \mathbf{s}[n] + D \mathbf{x}[n],

    where :math:`\mathbf{s}[n + \mathbf{m}]` stacks the :math:`s_i[n + m_i]`:
    each line emits what was written into it :math:`m_i` steps earlier. With
    every delay equal to 1, this is :func:`state_space` with
    :math:`\mathbf{h} = \mathbf{s}`; :attr:`B`, :attr:`C`, :attr:`D`,
    :attr:`out_idx` and the return value follow the same conventions.

    With :attr:`absorption`, each line also has a filter :math:`g_i`, for the
    loss of a feedback delay network: a cascade of second-order sections, as in
    :func:`sosfilt`, or a gain. Write :math:`\hat{s}_i = g_i \ast s_i` for the
    filtered output of line :math:`i`, the convolution of :math:`s_i` with the
    impulse response of :math:`g_i`. By default the filtered output replaces
    :math:`\mathbf{s}` on the right of both equations, so :math:`A`, :math:`C`
    and :attr:`out_idx` all see it:

    .. math::
        \mathbf{s}[n + \mathbf{m}] &= A \hat{\mathbf s}[n] + B \mathbf{x}[n], \\
        \mathbf{y}[n] &= C \hat{\mathbf s}[n] + D \mathbf{x}[n].

    With ``absorption_after_matrix=True``, only the feedback path is filtered,
    and :math:`\mathbf{y}` is as without filters:

    .. math::
        \mathbf{s}[n + \mathbf{m}] = G(A \mathbf{s})[n] + B \mathbf{x}[n],

    where :math:`G` filters entry :math:`i` by :math:`g_i`. In both positions
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
        zi (Sequence[Tensor], optional): the first outputs of each delay line,
            :math:`s_i[0], \dots, s_i[m_i - 1]`, as one queue per line, of
            shape :math:`(m_i)` or :math:`(B, m_i)`. Default: ``None``, all
            zero.
        block_size (int, optional): the processing block length, at most
            :math:`\min_i m_i`. Default: ``None``, which uses
            :math:`\min_i m_i`.
        out_idx (int, optional): output only this delay line, instead of using
            :attr:`C`. Default: ``None``.
        absorption (Tensor, optional): the per-line filters :math:`g_i`, either
            as sections of shape :math:`(M, K, 5)` or :math:`(B, M, K, 5)`,
            :math:`K \ge 1`, one cascade of :math:`K` sections per line, each
            section a row :math:`b_0, b_1, b_2, a_1, a_2` as in
            :func:`sosfilt`, or as gains of shape :math:`(M)` or
            :math:`(B, M)`, one per line. The number of dimensions tells them
            apart. It must have the dtype and device of :attr:`x`. Default:
            ``None``, no filters.
        absorption_zi (Tensor, optional): the initial states of the cascades,
            as the ``zi`` of :func:`sosfilt` for each line, of shape
            :math:`(M, K, 2)` or :math:`(B, M, K, 2)`. It must have the dtype
            and device of :attr:`x`, and be ``None`` for gains, which have no
            state. Needs :attr:`absorption`. Default: ``None``, all zero.
        absorption_after_matrix (bool, optional): put the filters after
            :math:`A` instead of after the delay lines, as above. Needs
            :attr:`absorption`. Default: ``False``.
        unroll_factor (int, optional): the ``unroll_factor`` of the recursion
            that runs the filters, one block of at most :attr:`block_size`
            samples at a time (see :func:`state_space`): ``None`` runs native
            kernels; an integer :math:`\ge 1` runs PyTorch ops on any device,
            the sequential loop for ``1`` or any value of at least the block
            length. Gains ignore it. Needs :attr:`absorption`. Default:
            ``None``.

    Returns:
        Tensor or (Tensor, tuple of Tensor) or (Tensor, tuple of Tensor,
        Tensor): the outputs, as :func:`state_space`. With
        :attr:`zi`, the pair ``(y, zf)``, where ``zf`` is a tuple of the next
        outputs of each delay line, :math:`s_i[N], \dots, s_i[N + m_i - 1]`,
        one queue per line, of shape :math:`(B, m_i)`: pass it as :attr:`zi` to
        continue.
        With :attr:`absorption` as sections, and :attr:`zi` or
        :attr:`absorption_zi`, the triple ``(y, zf, absorption_zf)``, where
        ``absorption_zf`` is the final state of the cascades, of shape
        :math:`(B, M, K, 2)`; whichever initial state is not given starts at
        zero. With gains, which have no state, the return value is that
        without :attr:`absorption`.

    Raises:
        ValueError: if a delay or :attr:`block_size` is invalid, if
            :attr:`zi` does not hold one queue per delay line, if both
            :attr:`C` and :attr:`out_idx` are given, if :attr:`D` gives a
            different number of outputs from :attr:`C`, :attr:`out_idx` or
            the states, if :attr:`absorption` does not have 1 to 4
            dimensions, if its lines or batch size do not match :math:`M` or
            the batch size of :attr:`x`, if its sections do not have 5
            coefficients, if there is no section, if :attr:`absorption_zi` is
            not of shape :math:`(M, K, 2)` or :math:`(B, M, K, 2)`, or is
            given with gains, if :attr:`A`, :attr:`B`, :attr:`C`, :attr:`D`, a queue in
            :attr:`zi`, :attr:`absorption` or :attr:`absorption_zi` differ from
            :attr:`x` in dtype or device, if :attr:`unroll_factor` is not
            ``None`` or an integer of at least 1, if :attr:`absorption_zi`,
            :attr:`absorption_after_matrix` or :attr:`unroll_factor` is given
            without :attr:`absorption`, or if the other shapes do not match.
        TypeError: if a queue in :attr:`zi`, :attr:`absorption` or
            :attr:`absorption_zi` is not a tensor.

    Example::

        >>> from philtorch.lti import delay_state_space
        >>> # m_1 = 3, s[n + 3] = 0.5 s[n] + x[n], y[n] = s[n]
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        >>> delay_state_space(torch.tensor([[0.5]]), x, [3]).squeeze(-1)
        tensor([[0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.5000]])
        >>> # the same, with g_1 = 0.5 / (1 - 0.5 z^-1) on the delay line
        >>> absorption = torch.tensor([[[0.5, 0.0, 0.0, -0.5, 0.0]]])
        >>> delay_state_space(
        ...     torch.tensor([[0.5]]), x, [3], absorption=absorption
        ... ).squeeze(-1)
        tensor([[0.0000, 0.0000, 0.0000, 0.5000, 0.2500, 0.1250, 0.1875]])
    """
    _ssm.check_arguments(x, C, out_idx)
    if A.dim() not in (2, 3):
        raise ValueError(f"State matrix A must be 2D or 3D, got {A.shape}")
    if A.size(-2) != A.size(-1):
        raise ValueError(f"State matrix A must be square, got {A.shape}")
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
    _check_shared(A=A, B=B, C=C, D=D, x=x)
    Bx = _ssm_B(B, x, batch_size, M)
    filters, absorption_state = None, None
    if absorb:
        filters, absorption_state = _absorption_filters(absorption, absorption_zi, M, batch_size, x)
    initial = _initial_queues(zi, delays, batch_size, x)
    return_states = zi is not None or absorption_zi is not None

    def absorb_block(signal: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Filter a (B, L, M) block line by line, from and to a (B * M, 2K) state."""
        if state is None:
            # Gains have no state, and need no recursion.
            return signal * filters, state
        length = signal.size(1)
        flat = signal.permute(0, 2, 1).reshape(batch_size * M, length)
        filter_A, filter_B, filter_C, filter_D = filters
        flat, state = state_space(
            filter_A,
            flat,
            B=filter_B,
            C=filter_C,
            D=filter_D,
            zi=state,
            unroll_factor=unroll_factor,
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
        h = chunks[0][:, :0]
        if out_idx is not None:
            h = h[..., out_idx]
        y = _ssm_C_D(h, x, C, D, batch_size, M)

    if not return_states:
        return y
    stream = torch.cat(chunks[tail:], dim=1)
    end = samples + prefix - tail * block_size
    zf = tuple(stream[:, end - delay : end, line].contiguous() for line, delay in enumerate(delays))
    if absorption_state is not None:
        return y, zf, absorption_state.view(batch_size, M, -1, 2)
    return y, zf
