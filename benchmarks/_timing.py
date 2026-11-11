"""The benchmarks' timing function."""

import time

import torch


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, n=10, width=8):
    """Mean milliseconds per call after a warm-up call, or the error's name."""
    try:
        torch.cuda.empty_cache()
        fn()
        _synchronize()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        _synchronize()
        return f"{1e3 * (time.perf_counter() - start) / n:{width}.2f} ms"
    # Out of memory and no kernel are RuntimeErrors too.
    except RuntimeError as e:
        return f"{type(e).__name__:>{width + 3}}"
