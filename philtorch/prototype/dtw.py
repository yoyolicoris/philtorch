"""Dynamic time warping parallelized over time (prototype).

The accumulated cost of a row of the DTW matrix is a min-plus matrix product
of the previous row, so the distance is a product of per-row matrices, which
a tree reduction computes in O(log N) rounds of batched matrix products. With
``gamma > 0`` the products are in the log semiring, giving soft-DTW (Cuturi
and Blondel, 2017). The products are the Helion kernels of the HMM prototype,
so the inputs must be CUDA tensors.

The gradient of the distance with respect to the cost matrix is the
alignment: the optimal path's 0/1 indicator for DTW, and the expected
alignment for soft-DTW. Both come from autograd, with no backtracking, and
are themselves differentiable.
"""

from typing import Literal

import torch
from torch import Tensor

from .hmm import _log_matmul, _logsumexp, _max_matmul

StepPattern = Literal["symmetric", "asymmetric"]


def _reduce(combine_fn, x: Tensor) -> Tensor:
    """The product x[:, 0] ⊗ ... ⊗ x[:, n - 1], as a tree of batched products."""
    while x.size(1) > 1:
        n = x.size(1)
        paired = combine_fn(x[:, 0 : n - 1 : 2], x[:, 1:n:2])
        x = torch.cat([paired, x[:, n - 1 :]], dim=1) if n % 2 else paired
    return x[:, 0]


def _row_matrices(cost: Tensor, step_pattern: StepPattern, combine) -> Tensor:
    """T[n - 1][k, m]: the best score from cell (n - 1, k) to cell (n, m).

    Scores are negated costs. With the symmetric steps a path enters row n at
    column k from above, or at k + 1 diagonally, then moves right to m; with
    the asymmetric steps it enters at m from above or diagonally and stops.
    ``combine`` merges the two ways in: max, or logaddexp for soft-DTW.
    """
    rows = cost[:, 1:]
    M = cost.size(-1)
    k = torch.arange(M, device=cost.device)
    row, col = k[:, None], k
    if step_pattern == "asymmetric":
        score = -rows.unsqueeze(-2).expand(*rows.shape[:-1], M, M)
        return torch.where((row == col) | (row + 1 == col), score, float("-inf"))
    # prefix[a] is the cost of row n's first a cells, so a run over columns
    # a..m costs prefix[m + 1] - prefix[a].
    prefix = torch.nn.functional.pad(rows.cumsum(-1), (1, 0))
    end = prefix[..., None, 1:]
    down = -(end - prefix[..., :-1, None])
    diagonal = -(end - prefix[..., 1:, None])
    # Above the diagonal both ways in are possible; on it, only from above.
    # Merging only where both exist keeps -inf out of logaddexp's gradient.
    both = torch.where(row < col, combine(down, diagonal), down)
    return torch.where(row <= col, both, float("-inf"))


def dtw(cost: Tensor, gamma: float = 0.0, step_pattern: StepPattern = "symmetric") -> Tensor:
    """The (soft-)DTW distance between two sequences, from their pairwise costs.

    Args:
        cost (Tensor): cost[b, n, m] of matching frame n of the first sequence
            with frame m of the second, of shape (B, N, M), on CUDA.
        gamma (float): 0 for DTW, the minimum total cost of a path from
            (0, 0) to (N - 1, M - 1); positive for soft-DTW,
            -gamma log sum over paths of exp(-cost / gamma).
        step_pattern (str): ``"symmetric"`` for steps (1, 0), (0, 1) and
            (1, 1); ``"asymmetric"`` for (1, 0) and (1, 1) only, so every
            frame of the first sequence matches one frame of the second.

    Returns:
        Tensor: the distances, of shape (B,). Their gradient with respect to
        ``cost`` is the alignment.
    """
    assert cost.dim() == 3, f"cost must be (B, N, M), got {tuple(cost.shape)}"
    if not cost.is_cuda:
        raise ValueError("dtw runs Helion kernels, which need CUDA tensors.")
    scale = 1.0 / gamma if gamma > 0 else 1.0
    if gamma > 0:
        product, combine, reduce = _log_matmul, torch.logaddexp, _logsumexp
    else:
        product, combine = _max_matmul, torch.maximum

        def reduce(t, dim):
            return t.amax(dim=dim)

    # The first row's scores: from (0, 0), only rightward moves, or none.
    first = -cost[:, 0].cumsum(-1) if step_pattern == "symmetric" else None
    if step_pattern == "asymmetric":
        first = torch.full_like(cost[:, 0], float("-inf"))
        first[:, 0] = -cost[:, 0, 0]
    first = first * scale
    if cost.size(1) == 1:
        score = first[:, -1]
    else:
        T = _row_matrices(cost * scale, step_pattern, combine)
        # Fold the first row into the first transfer, as the HMM folds its prior.
        head = reduce(first.unsqueeze(-1) + T[:, 0], dim=-2)
        T = torch.cat([head.unsqueeze(-2).expand_as(T[:, 0]).unsqueeze(1), T[:, 1:]], dim=1)
        score = _reduce(product, T)[:, 0, -1]
    return -score / scale


def dtw_rowwise(
    cost: Tensor, gamma: float = 0.0, step_pattern: StepPattern = "symmetric"
) -> Tensor:
    """The (soft-)DTW distance, one row at a time with prefix computations.

    After Xiao et al., "Parallelizing Dynamic Time Warping Algorithm Using
    Prefix Computations on GPU" (HPCC 2013), extended to soft-DTW. Within row
    n, with t[m] = cost[n, m] + softmin(D[n - 1, m], D[n - 1, m - 1]) and the
    row's prefix sums y[m] = cost[n, 0] + ... + cost[n, m], the symmetric
    steps give

        D[n, m] = y[m] + softmin over k <= m of (t[k] - y[k]),

    a prefix minimum, or for soft-DTW a prefix logsumexp. So each row is a
    few parallel operations over its cells, and only the rows run one after
    another. The symmetric steps are the same with the sequences swapped, so
    the loop runs over the shorter one: min(N, M) sequential steps, N * M work
    in all, and no M^3 work or B * N * M^2 memory as in :func:`dtw`. The
    asymmetric steps aren't symmetric, so they always loop over N. It runs on
    any device. The arguments and result are those of :func:`dtw`.
    """
    assert cost.dim() == 3, f"cost must be (B, N, M), got {tuple(cost.shape)}"
    if step_pattern == "symmetric" and cost.size(1) > cost.size(2):
        cost = cost.mT
    soft = gamma > 0

    def softmin(a: Tensor, b: Tensor) -> Tensor:
        if not soft:
            return torch.minimum(a, b)
        return -gamma * _logsumexp(torch.stack([-a / gamma, -b / gamma]), dim=0)

    inf = torch.full_like(cost[:, 0, :1], float("inf"))
    # Split into rows once: indexing cost[:, n] in the loop would make every
    # row's backward write a gradient the size of the whole cost matrix.
    costs = cost.unbind(1)
    prefixes = cost.cumsum(-1).unbind(1)  # every row's prefix sums at once
    if step_pattern == "symmetric":
        row = prefixes[0]
    else:
        row = torch.cat([costs[0][:, :1], inf.expand(-1, cost.size(-1) - 1)], dim=-1)
    for n in range(1, cost.size(1)):
        # From the previous row: straight down, or diagonally from m - 1.
        t = costs[n] + softmin(row, torch.cat([inf, row[:, :-1]], dim=-1))
        if step_pattern == "asymmetric":
            row = t
            continue
        y = prefixes[n]
        z = t - y
        if soft:
            row = y - gamma * torch.logcumsumexp(-z / gamma, dim=-1)
        else:
            row = y + z.cummin(-1).values
    return row[:, -1]


def dtw_fused(cost: Tensor) -> Tensor:
    """The DTW distance with symmetric steps, in a single Helion kernel.

    The row-wise prefix method of :func:`dtw_rowwise`, with the loop over the
    shorter sequence inside one kernel. Experimental: hard DTW and the forward
    pass only, so no gradient. Inputs must be CUDA tensors.
    """
    from ._dtw_helion import dtw_rows_kernel

    assert cost.dim() == 3, f"cost must be (B, N, M), got {tuple(cost.shape)}"
    if not cost.is_cuda:
        raise ValueError("dtw_fused runs a Helion kernel, which needs CUDA tensors.")
    if cost.size(1) > cost.size(2):
        cost = cost.mT
    return dtw_rows_kernel(cost.detach().contiguous())[:, -1]
