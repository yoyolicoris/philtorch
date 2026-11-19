"""The benchmarks' timing function."""

import torch
import torch.utils.benchmark as benchmark


def bench(fn, width=8):
    """Median milliseconds per call by ``Timer.blocked_autorange``, after a warm-up
    call, or the error's name. The timer synchronizes CUDA at each boundary."""
    try:
        torch.cuda.empty_cache()
        fn()
        measurement = benchmark.Timer(
            stmt="fn()", globals={"fn": fn}, num_threads=torch.get_num_threads()
        ).blocked_autorange(min_run_time=0.5)
        return f"{1e3 * measurement.median:{width}.2f} ms"
    # Out of memory and no kernel are RuntimeErrors too.
    except RuntimeError as e:
        return f"{type(e).__name__:>{width + 3}}"
