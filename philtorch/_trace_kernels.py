"""Following backpointers in parallel, for Viterbi decoding's traceback.

Each step's backpointers are a map of K states, and following them is a
chain of maps, composed in two levels as the HMM message chains are: each
chunk's maps composed, the chain of the compositions one level up, then
each chunk from its start. So a traceback of L steps takes a sequential
depth of a few hundred steps whatever L is, and about L K work.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

# The chunk length: each kernel program takes T sequential steps.
_CHUNK = 64


@triton.jit
def _time(c, s, L, T: tl.constexpr, REVERSE: tl.constexpr):
    """The time of step s of chunk c, a 64-bit index."""
    t = c * T + s
    return (L - 1 - t) if REVERSE else t


@triton.jit
def _follow(maps, n, K, current, mask, RELATIVE: tl.constexpr):
    """The states that ``current`` leads to at step n."""
    step = tl.load(maps + n * K + current, mask=mask, other=0).to(tl.int32)
    return current - step if RELATIVE else step


@triton.jit
def _trace_totals_kernel(
    maps_ptr, total_ptr, L, K, C, stride_mb, T: tl.constexpr, REVERSE: tl.constexpr,
    RELATIVE: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    """Chunk c's composed map, total[k] = F[cT + T - 1](... F[cT](k)), in chain order."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, L - c * T)
    maps = maps_ptr + b * stride_mb
    states = tl.arange(0, BK)
    current = states
    for s in range(0, steps):
        current = _follow(maps, _time(c, s, L, T, REVERSE), K, current, states < K, RELATIVE)
    tl.store(total_ptr + pid.to(tl.int64) * K + states, current, mask=states < K)


@triton.jit
def _trace_sweep_kernel(
    start_ptr, maps_ptr, out_ptr, L, K, C, stride_mb, T: tl.constexpr, REVERSE: tl.constexpr,
    RELATIVE: tl.constexpr,
):  # fmt: skip
    """x[t] = F[t](x[t - 1]) through chunk c from its start, written at time n."""
    pid = tl.program_id(0)
    b = (pid // C).to(tl.int64)
    c = (pid % C).to(tl.int64)
    steps = tl.minimum(T, L - c * T)
    maps = maps_ptr + b * stride_mb
    current = tl.load(start_ptr + pid)
    for s in range(0, steps):
        n = _time(c, s, L, T, REVERSE)
        current = _follow(maps, n, K, current, True, RELATIVE)
        tl.store(out_ptr + b * L + n, current)


def trace(x0: Tensor, maps: Tensor, reverse: bool = False, relative: bool = False) -> Tensor:
    """The states x[t] = maps[t][x[t - 1]], t = 0, ..., L - 1, from x[-1] = x0.

    Args:
        x0: the starting states, (B,) int32.
        maps: (B, L, K) integers, each step's K values contiguous, the steps
            contiguous; maps[b, t, k] is the state that state k leads to at
            step t, or with ``relative`` how many states below k it is,
            which fits a narrow type such as int8 whatever K is.
        reverse: run from time L - 1 down to 0; the state after the step at
            time n is still written at n.
        relative: read the maps as offsets.

    Returns:
        The states, (B, L) int32, each written at its step's time.
    """
    B, L, K = maps.shape
    out = maps.new_empty(B, L, dtype=torch.int32)
    if B == 0 or L == 0:
        return out
    assert maps.stride()[1:] == (K, 1), "steps of K contiguous values"
    T = _CHUNK
    C = triton.cdiv(L, T)
    flags = dict(T=T, REVERSE=reverse, RELATIVE=relative, num_warps=1)
    if C == 1:
        starts = x0.contiguous()
    else:
        # The chunks' compositions are absolute maps, whatever the steps' are.
        totals = maps.new_empty(B, C, K, dtype=torch.int32)
        BK = max(triton.next_power_of_2(K), 2)
        _trace_totals_kernel[(B * C,)](
            maps, totals, L, K, C, maps.stride(0), **flags, BK=BK
        )  # fmt: skip
        ends = trace(x0, totals)
        starts = torch.cat([x0.unsqueeze(1), ends[:, :-1]], dim=1).contiguous()
    _trace_sweep_kernel[(B * C,)](starts, maps, out, L, K, C, maps.stride(0), **flags)
    return out
