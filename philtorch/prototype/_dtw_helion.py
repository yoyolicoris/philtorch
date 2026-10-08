"""A one-kernel row-wise DTW forward pass in Helion (experimental).

Row n of the DTW matrix from row n - 1, with s(j) = min(D_prev(j), D(j)):

    s(j) = min(alpha(j), c(j) + s(j - 1)),  alpha(j) = min(D_prev(j), c(j) + D_prev(j))
    D(j) = c(j) + min(D_prev(j), s(j - 1))

The first line is a scan of the maps x -> min(A, C + x), composed as pairs
(A, C). Carrying a second pair G, each segment's composition without its last
map, gives s(j - 1) at position j as well, so no row is shifted. One program
holds a whole row of one batch item and loops over the rows, so the kernel
launches once.

Only the hard, symmetric distance's forward pass so far: no gradient.
"""

import os

import helion
import helion.language as hl
import torch
from torch import Tensor

_SETTINGS = {} if os.environ.get("HELION_AUTOTUNE_EFFORT") else {"autotune_effort": "quick"}


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


# static_shapes: the best configuration depends strongly on the row length.
@helion.kernel(**_SETTINGS, static_shapes=True)
def dtw_rows_kernel(cost: Tensor) -> Tensor:
    """The last row of the DTW matrix of cost (B, R, L), looping over the R rows."""
    B, R, L = cost.shape
    out = torch.empty([B, L], dtype=cost.dtype, device=cost.device)
    for tile_b in hl.tile(B, block_size=1):
        prev = hl.cumsum(cost[tile_b, 0, :], dim=1)
        for n in hl.grid(1, R):
            c = cost[tile_b, n, :]
            alpha = torch.minimum(prev, c + prev)
            identity_a = torch.full_like(c, float("inf"))
            identity_c = torch.zeros_like(c)
            _, _, s_left, _ = hl.associative_scan(
                _compose_min_plus, (alpha, c, identity_a, identity_c), dim=1
            )
            prev = c + torch.minimum(prev, s_left)
        out[tile_b, :] = prev
    return out
