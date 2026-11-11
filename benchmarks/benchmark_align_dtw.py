"""Benchmark philtorch.align.dtw, optionally against dDTW.

Run from the repository root on a CUDA machine:

    pixi run python benchmarks/benchmark_align_dtw.py

Times are for the distance and its gradient (the alignment), after a
warm-up call that compiles the kernels. To compare with dDTW
(https://github.com/groupmm/dDTW), clone it and set DDTW_SRC to the clone;
its CUDA extension compiles on first use, and it needs min(N, M) <= 1024.
"""

import os
import sys

import torch
from _timing import bench

from philtorch.align import dtw


def load_ddtw(gamma):
    if "DDTW_SRC" not in os.environ:
        return None
    sys.path.insert(0, os.environ["DDTW_SRC"])
    from ddtw import dDTW

    return dDTW(
        min_function="softmin" if gamma > 0 else "hardmin",
        gamma=gamma or 1.0,
        normalization="none",
        backend="cuda_cpp",
    )


def main():
    print(torch.cuda.get_device_name(), flush=True)
    for gamma in (0.0, 0.5):
        reference = load_ddtw(gamma)
        for B, N, M in (
            (1, 1024, 1024),
            (16, 1024, 1024),
            (64, 256, 256),
            (1, 10_000, 64),
            (1, 4096, 4096),
            (1, 8000, 8000),
        ):
            cost = torch.rand(B, N, M, device="cuda", requires_grad=True)
            ours = bench(lambda: torch.autograd.grad(dtw(cost, gamma).sum(), cost), width=9)
            row = f"gamma={gamma} B={B:2d} N={N:>6,} M={M:>5,}: philtorch {ours}"
            if reference is not None:
                theirs = bench(lambda: torch.autograd.grad(reference(C=cost).sum(), cost), width=9)
                row += f" | dDTW {theirs}"
            print(row, flush=True)


if __name__ == "__main__":
    main()
