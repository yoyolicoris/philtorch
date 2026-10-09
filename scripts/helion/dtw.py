"""Helion versions of philtorch/prototype/_dtw_triton.py's kernels (dev only).

The production kernels are hand-written Triton. These are their Helion
originals, kept to explore and tune new variants: scripts/helion/aot.py tunes
them, and the configurations it finds go into the Triton module by hand. See
that module for what the kernels compute.

Inside a kernel, constants are spelled out (``float("inf")``): a module
constant would compile to an attribute of this module.
"""

import os

import helion
import helion.language as hl
import torch
from helion.experimental import aot_kernel
from torch import Tensor

# Settings for the tuning (scripts/helion/aot.py). Forked precompile processes
# can fail in a multithreaded parent, so spawn them. The adaptive compile
# timeout carries over from one shape's tuning to the next, so after a small
# shape every configuration of a long one times out: fix the timeout instead.
_SETTINGS = {
    "autotune_precompile": "spawn",
    "autotune_adaptive_timeout": False,
    "autotune_compile_timeout": 120,
} | ({} if os.environ.get("HELION_AUTOTUNE_EFFORT") else {"autotune_effort": "quick"})
# Only the row length chooses the configuration: the batch and the number of
# rows, a loop inside each program, are marked as batched dimensions.
_BATCHED = [0, 1, None]
# Seeds for the tuning, which can otherwise start from Helion's default alone:
# many warps for long rows.
_DTW_DP_FALLBACK = helion.Config(num_warps=16)
_DAG_FORWARD_FALLBACK = helion.Config(num_warps=8)


def _compose_min_plus(a_l, c_l, ga_l, gc_l, a_r, c_r, ga_r, gc_r):
    """Compose x -> min(A, C + x) maps, the left (earlier) one first.

    (A, C) is the whole segment's map, (GA, GC) the segment's without its
    last map; a single map's G is the identity, A = inf and C = 0.
    """
    return (
        torch.minimum(a_r, c_r + a_l),
        c_r + c_l,
        torch.minimum(ga_r, gc_r + a_l),
        gc_r + c_l,
    )


def _softmin(x, y):
    """-log(e^-x + e^-y) as min(x, y) - log(1 + e^-|x - y|); wrong if both are inf."""
    return torch.minimum(x, y) - torch.log(1 + torch.exp(-torch.abs(x - y)))


def _compose_softmin_plus(a_l, c_l, ga_l, gc_l, a_r, c_r, ga_r, gc_r):
    """As :func:`_compose_min_plus` with min replaced by softmin.

    Unguarded: inside these scans at most one of the two terms is inf (only G
    starts at the identity, and A is finite).
    """
    return _softmin(a_r, c_r + a_l), c_r + c_l, _softmin(ga_r, gc_r + a_l), gc_r + c_l


def _compose_min_plus_whole(a_l, c_l, a_r, c_r):
    """:func:`_compose_min_plus` without G."""
    return torch.minimum(a_r, c_r + a_l), c_r + c_l


def _compose_softmin_plus_whole(a_l, c_l, a_r, c_r):
    """:func:`_compose_softmin_plus` without G."""
    return _softmin(a_r, c_r + a_l), c_r + c_l


def _compose_affine(a_l, b_l, ga_l, gb_l, a_r, b_r, ga_r, gb_r):
    """Compose x -> a x + b maps, the left (earlier) one first, with G as above."""
    return a_r * a_l, a_r * b_l + b_r, ga_r * a_l, ga_r * b_l + gb_r


# Dynamic shapes, so one compiled kernel serves every length; the configuration
# still depends on the row length.
@aot_kernel(batched=[_BATCHED, None, None], autotune_seed_configs=_DTW_DP_FALLBACK, **_SETTINGS)
def _dtw_dp_kernel(cost: Tensor, soft: hl.constexpr, diag: hl.constexpr) -> Tensor:
    """D of cost (B, R, L), with steps up, left and, if ``diag``, up-left."""
    B, R, L = cost.shape
    D = torch.empty_like(cost)
    for tile_b in hl.tile(B, block_size=1):
        prev = hl.cumsum(cost[tile_b, 0, :], dim=1)
        D[tile_b, 0, :] = prev
        for n in hl.grid(1, R):
            c = cost[tile_b, n, :]
            identity_a = torch.full_like(c, float("inf"))
            identity_c = torch.zeros_like(c)
            if not diag:
                # D(j) = min(c(j) + prev(j), c(j) + D(j - 1)): a scan of the
                # maps x -> min(A, C + x), whose whole value is D itself.
                if soft:
                    prev = hl.associative_scan(_compose_softmin_plus_whole, (c + prev, c), dim=1)[0]
                else:
                    prev = hl.associative_scan(_compose_min_plus_whole, (c + prev, c), dim=1)[0]
            elif soft:
                # c + prev and prev are finite, so the plain form is safe.
                alpha = _softmin(prev, c + prev)
                s_left = hl.associative_scan(
                    _compose_softmin_plus, (alpha, c, identity_a, identity_c), dim=1
                )[2]
                # s_left is inf only in column 0, where prev is finite.
                prev = c + _softmin(prev, s_left)
            else:
                alpha = torch.minimum(prev, c + prev)
                s_left = hl.associative_scan(
                    _compose_min_plus, (alpha, c, identity_a, identity_c), dim=1
                )[2]
                prev = c + torch.minimum(prev, s_left)
            D[tile_b, n, :] = prev
    return D


@aot_kernel(batched=[_BATCHED] * 4, autotune_seed_configs=_DAG_FORWARD_FALLBACK, **_SETTINGS)
def _dag_forward_kernel(
    w_down: Tensor, w_right_next: Tensor, w_diag_next: Tensor, x: Tensor
) -> Tensor:
    """y of the forward accumulation; the *_next weights are taken at column j + 1."""
    B, R, L = x.shape
    y = torch.empty_like(x)
    for tile_b in hl.tile(B, block_size=1):
        y_prev = torch.zeros_like(x[tile_b, 0, :])
        for i in hl.grid(R):
            base = x[tile_b, i, :] + w_down[tile_b, i, :] * y_prev
            a = w_right_next[tile_b, i, :]
            b = a * base + w_diag_next[tile_b, i, :] * y_prev
            identity_a = torch.ones_like(b)
            identity_b = torch.zeros_like(b)
            z_left = hl.associative_scan(_compose_affine, (a, b, identity_a, identity_b), dim=1)[3]
            y_prev = base + z_left
            y[tile_b, i, :] = y_prev
    return y
