"""Benchmark non-linear recurrence solver (newton_solve) against torch scan.

The baseline is ``scan`` under ``torch.compile`` with the default Inductor
backend (compile time excluded); eager ``scan`` is a Python loop over T.

Run from the repository root on a CUDA machine:

    pixi run python benchmarks/benchmark_nl.py
"""

import time

import torch
from torch._higher_order_ops import scan

from philtorch.nl import newton_solve


def fail(e):
    """Table cell for a failed run: n/a and the exception type."""
    return f"{('n/a ' + type(e).__name__)[:15]:>16}"


def bench(fn, warmup=3, reps=10):
    """Mean milliseconds per call; warm-up calls absorb compilation."""
    try:
        for _ in range(warmup):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return f"{1e3 * (time.perf_counter() - start) / reps:16.2f}"
    except Exception as e:
        return fail(e)


def bench_mem(fn):
    """Peak allocated megabytes during call."""
    if not torch.cuda.is_available():
        return f"{'-':>16}"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        torch.cuda.synchronize()
        return f"{torch.cuda.max_memory_allocated() / (1024 * 1024):16.2f}"
    except Exception as e:
        return fail(e)


def main():
    # Inductor lowers scan to a while_loop that reads its index with .item(),
    # which fails to compile (DataDependentOutputException) without this.
    torch._dynamo.config.capture_scalar_outputs = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    print(f"Device: {dev_name} ({device})", flush=True)

    names = ("scan", "newton")
    header = f"{'Configuration':<22s}" + "".join(
        f"{f'{n} ({m})':>16s}" for m in ("fwd", "f+b") for n in names
    )
    print(header)
    print("-" * len(header))

    torch.manual_seed(0)
    A = torch.tensor([[0.3, 0.1], [0.0, 0.2]], dtype=torch.float32, device=device)
    all_rows = []

    for M in (1, 2):  # M=1 is the scalar case, with no trailing state dim
        if M == 1:

            def fn(y, xt):
                return torch.tanh(0.5 * y + xt)

        else:

            def fn(y, xt):
                return torch.tanh(y @ A.T + xt)

        def step(carry, xt):
            # Compiled scan rejects a carry that aliases the output.
            y = fn(carry, xt)
            return y.clone(), y.clone()

        def run_scan(init, x):
            return scan(step, init=init, xs=x, dim=1)[1]

        def run_newton(init, x):
            return newton_solve(fn, x, init)[0]

        for B in (4, 16):
            for T in (1024, 4096, 16384):
                shape = (B, T) if M == 1 else (B, T, M)
                x = torch.randn(shape, dtype=torch.float32, device=device)
                init = torch.zeros(shape[:1] + shape[2:], device=device)

                # Compile once per configuration; the warm-up calls in bench
                # run it, so compilation is not timed.
                torch._dynamo.reset()
                runs = (torch.compile(run_scan), run_newton)

                def fwd(run):
                    def call():
                        with torch.no_grad():
                            run(init, x)

                    return call

                def fwd_bwd(run):
                    def call():
                        x_in = x.detach().requires_grad_(True)
                        init_in = init.detach().requires_grad_(True)
                        run(init_in, x_in).sum().backward()

                    return call

                calls = [f(r) for f in (fwd, fwd_bwd) for r in runs]
                times = [bench(c) for c in calls]
                mems = [bench_mem(c) for c in calls]

                cfg = f"M={M}, B={B}, T={T}"
                print(f"{cfg:<22s}" + "".join(times), flush=True)
                all_rows.append((cfg, mems))

    print("\nPeak Memory (MB):")
    print(header)
    print("-" * len(header))
    for cfg, mems in all_rows:
        print(f"{cfg:<22s}" + "".join(mems))


if __name__ == "__main__":
    main()
