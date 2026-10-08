"""Benchmark the prototype HMM Viterbi and DTW against torbi and dDTW.

Run from the repository root on a CUDA machine:

    pixi run python benchmarks/benchmark_prototype_semiring.py {viterbi,convcode,dtw}

The baselines are built from source against the environment's PyTorch,
since their wheels pin older versions:

* torbi (https://github.com/maxrmorrison/torbi), Viterbi decoding:

      git clone https://github.com/maxrmorrison/torbi
      cd torbi && TORCH_CUDA_ARCH_LIST="12.0" pixi run --manifest-path \\
          /path/to/philtorch/pixi.toml python setup.py build_ext --inplace

  then set TORBI_LIB to the built ``torbi/_C*.so``. torbi needs at least 32
  states: its kernel reads the first 32 entries of each row unchecked, so
  smaller models are padded to 32 with impossible states.

* dDTW (https://github.com/groupmm/dDTW), DTW and soft-DTW: clone it and set
  DDTW_SRC to the clone. Its CUDA extension compiles on first use.

Times include a warm-up call, so Helion's autotuning, which caches its
results, is excluded. Results on an RTX 5060 Ti (2026-10-08): torbi and
dDTW win 10-100x from 32 states or template frames up; the scans win only
with few states and long sequences, e.g. Viterbi with K <= 16 on one
100,000-step sequence, 3.5-5x faster than torbi.
"""

import os
import sys
import time

import torch

from philtorch.prototype.dtw import dtw
from philtorch.prototype.hmm import viterbi

IMPOSSIBLE = -1e30  # torbi-safe -inf


def bench(fn, n=5):
    """Mean milliseconds per call after a warm-up call, or OOM."""
    try:
        torch.cuda.empty_cache()
        fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return f"{1e3 * (time.perf_counter() - start) / n:9.2f} ms"
    except torch.OutOfMemoryError:
        return "      OOM   "


def load_torbi():
    torch.ops.load_library(os.environ["TORBI_LIB"])

    def decode(log_emit, log_trans, log_init):
        """torbi's classical HMM, with the transition indexed [next, previous]."""
        B, N, K = log_emit.shape
        pad = max(32 - K, 0)
        log_emit = torch.nn.functional.pad(log_emit, (0, pad), value=IMPOSSIBLE).contiguous()
        log_trans = torch.nn.functional.pad(log_trans, (0, pad, 0, pad), value=IMPOSSIBLE)
        log_init = torch.nn.functional.pad(log_init, (0, pad), value=IMPOSSIBLE)
        frames = torch.full((B,), N, dtype=torch.int32, device=log_emit.device)
        return torch.ops.torbi.viterbi_decode(log_emit, frames, log_trans.mT.contiguous(), log_init)

    return decode


def bench_viterbi():
    torbi = load_torbi()
    for B in (1, 16):
        for K in (32, 64, 128, 256):
            for N in (1_000, 10_000, 100_000):
                log_emit = torch.randn(B, N, K, device="cuda") * 2
                log_trans = torch.randn(K, K, device="cuda").log_softmax(-1)
                log_init = torch.randn(K, device="cuda").log_softmax(-1)
                ours = bench(lambda: viterbi(log_emit, log_trans, log_init))
                theirs = bench(lambda: torbi(log_emit, log_trans, log_init))
                label = f"B={B:2d} K={K:3d} N={N:>7,}"
                print(f"{label}: philtorch {ours} | torbi {theirs}", flush=True)


def convolutional_trellis(constraint_length):
    """A rate-1/n code's trellis, with the last K_c input bits as the state."""
    K = 2**constraint_length
    state = torch.arange(K)
    successors = torch.stack([((state << 1) | bit) & (K - 1) for bit in (0, 1)], dim=1)
    log_trans = torch.full((K, K), IMPOSSIBLE)
    log_trans[state[:, None], successors] = torch.tensor(0.5).log()
    return log_trans


def bench_convcode():
    torbi = load_torbi()
    for constraint_length in (2, 3, 4, 5):
        log_trans = convolutional_trellis(constraint_length).cuda()
        K = log_trans.size(0)
        log_init = torch.full((K,), IMPOSSIBLE, device="cuda")
        log_init[0] = 0.0
        for B in (1, 64):
            for N in (10_000, 100_000):
                log_emit = torch.randn(B, N, K, device="cuda")
                ours = bench(lambda: viterbi(log_emit, log_trans, log_init))
                theirs = bench(lambda: torbi(log_emit, log_trans, log_init))
                label = f"K={K:2d} B={B:2d} N={N:>7,}"
                print(f"{label}: philtorch {ours} | torbi (padded to 32) {theirs}", flush=True)


def bench_dtw():
    sys.path.insert(0, os.environ["DDTW_SRC"])
    from ddtw import dDTW

    for gamma in (0.0, 0.5):
        reference = dDTW(
            min_function="softmin" if gamma > 0 else "hardmin",
            gamma=gamma or 1.0,
            normalization="none",
            backend="cuda_cpp",
        )
        for B in (1, 16):
            for M in (8, 32, 64, 128, 256):
                for N in (1_000, 10_000, 30_000):  # dDTW stores lengths as int16
                    cost = torch.rand(B, N, M, device="cuda", requires_grad=True)
                    # Forward and backward: the distance and its alignment.
                    ours = bench(lambda: torch.autograd.grad(dtw(cost, gamma).sum(), cost))
                    theirs = bench(lambda: torch.autograd.grad(reference(C=cost), cost))
                    label = f"gamma={gamma} B={B:2d} M={M:3d} N={N:>6,}"
                    print(f"{label}: philtorch {ours} | dDTW {theirs}", flush=True)


if __name__ == "__main__":
    # The guard matters: Helion's autotuner spawns subprocesses.
    print(torch.cuda.get_device_name(), flush=True)
    {"viterbi": bench_viterbi, "convcode": bench_convcode, "dtw": bench_dtw}[sys.argv[1]]()
