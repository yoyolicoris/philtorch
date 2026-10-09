"""Benchmark philtorch.estimation's HMM functions, optionally against torbi.

Run from the repository root on a CUDA machine:

    pixi run python benchmarks/benchmark_estimation_hmm.py

Each function runs twice: without gradients, which takes the chunked Triton
kernels, and with gradients, which takes the scan of differentiable matrix
products (the time includes the backward pass). Times exclude a warm-up call,
which compiles the kernels.

To compare Viterbi decoding with torbi (https://github.com/maxrmorrison/torbi),
build it against the environment's PyTorch, as its wheels pin older versions:

    git clone https://github.com/maxrmorrison/torbi
    cd torbi && TORCH_CUDA_ARCH_LIST="12.0" pixi run --manifest-path \\
        /path/to/philtorch/pixi.toml python setup.py build_ext --inplace

then set TORBI_LIB to the built ``torbi/_C*.so``. torbi needs at least 32
states: its kernel reads the first 32 entries of each row unchecked, so
smaller models are padded to 32 with impossible states.
"""

import os
import time

import torch

from philtorch.estimation import hmm_filter, hmm_smoother, hmm_viterbi

IMPOSSIBLE = -1e30  # torbi-safe -inf


def bench(fn, n=10):
    """Mean milliseconds per call after a warm-up call, or the error's name."""
    try:
        torch.cuda.empty_cache()
        fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return f"{1e3 * (time.perf_counter() - start) / n:8.2f} ms"
    except (torch.OutOfMemoryError, RuntimeError) as e:
        return f"{type(e).__name__:>11}"


def load_torbi():
    if "TORBI_LIB" not in os.environ:
        return None
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


def with_grad(fn, log_emit, log_trans, log_init):
    def run():
        emit = log_emit.detach().requires_grad_()
        score = fn(emit, log_trans, log_init)[0]
        torch.autograd.grad(score.sum(), emit)

    return run


def main():
    print(torch.cuda.get_device_name(), flush=True)
    torbi = load_torbi()
    for B in (1, 16):
        for K in (2, 4, 8, 16, 32, 64):
            for N in (1_000, 10_000, 100_000):
                log_emit = torch.randn(B, N, K, device="cuda") * 2
                log_trans = torch.randn(K, K, device="cuda").log_softmax(-1)
                log_init = torch.randn(K, device="cuda").log_softmax(-1)
                args = (log_emit, log_trans, log_init)
                row = [f"B={B:2d} K={K:2d} N={N:>7,}"]
                for name, fn in (("filter", hmm_filter), ("smoother", hmm_smoother)):
                    row.append(f"{name} {bench(lambda: fn(*args))}")
                row.append(f"viterbi {bench(lambda: hmm_viterbi(*args))}")
                row.append(f"filter+grad {bench(with_grad(hmm_filter, *args))}")
                if torbi is not None:
                    row.append(f"torbi {bench(lambda: torbi(*args))}")
                print(" | ".join(row), flush=True)


if __name__ == "__main__":
    main()
