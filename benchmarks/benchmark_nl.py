"""Benchmark non-linear recurrence solver (newton_solve) against torch.scan baseline.

Run from the repository root on a CUDA machine:

    pixi run python benchmarks/benchmark_nl.py
"""

import time

import torch
from torch._higher_order_ops import scan

from philtorch.nl import newton_solve


def bench(fn, warmup=3, reps=10):
    """Mean milliseconds per call."""
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
        return f"{1e3 * (time.perf_counter() - start) / reps:9.2f}"
    except RuntimeError as e:
        return f"{type(e).__name__[:9]:>9}"


def bench_mem(fn):
    """Peak allocated megabytes during call."""
    if not torch.cuda.is_available():
        return f"{'-':>9}"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return f"{mb:9.2f}"
    except RuntimeError as e:
        return f"{type(e).__name__[:9]:>9}"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    print(f"Device: {dev_name} ({device})", flush=True)

    header = (
        f"{'Configuration':<22s}"
        f"{'scan (fwd)':>12s}"
        f"{'newton (fwd)':>14s}"
        f"{'scan (f+b)':>14s}"
        f"{'newton (f+b)':>14s}"
        f"{'scan MB(f+b)':>14s}"
        f"{'newton MB(f+b)':>16s}"
    )
    print(header)
    print("-" * len(header))

    torch.manual_seed(0)
    all_rows = []

    # 1. Scalar case
    for B in (4, 16):
        for T in (1024, 4096, 16384):
            x = torch.randn(B, T, dtype=torch.float32, device=device, requires_grad=True)
            init = torch.zeros(B, dtype=torch.float32, device=device, requires_grad=True)

            def step_scalar(carry, xt):
                y = torch.tanh(0.5 * carry + xt)
                return y, y.clone()

            def fn_scalar(y, xt):
                return torch.tanh(0.5 * y + xt)

            def scan_fwd():
                with torch.no_grad():
                    scan(step_scalar, init=init, xs=x, dim=1)

            def newton_fwd():
                with torch.no_grad():
                    newton_solve(fn_scalar, x, init)

            def scan_fb():
                x_in = x.detach().requires_grad_(True)
                init_in = init.detach().requires_grad_(True)
                _, out = scan(step_scalar, init=init_in, xs=x_in, dim=1)
                out.sum().backward()

            def newton_fb():
                x_in = x.detach().requires_grad_(True)
                init_in = init.detach().requires_grad_(True)
                out, _ = newton_solve(fn_scalar, x_in, init_in)
                out.sum().backward()

            t_scan_fwd = bench(scan_fwd)
            t_newton_fwd = bench(newton_fwd)
            t_scan_fb = bench(scan_fb)
            t_newton_fb = bench(newton_fb)

            m_scan_fwd = bench_mem(scan_fwd)
            m_newton_fwd = bench_mem(newton_fwd)
            m_scan_fb = bench_mem(scan_fb)
            m_newton_fb = bench_mem(newton_fb)

            cfg = f"M=1, B={B}, T={T}"
            print(
                f"{cfg:<22s}{t_scan_fwd:>12s}{t_newton_fwd:>14s}{t_scan_fb:>14s}{t_newton_fb:>14s}"
                f"{m_scan_fb:>14s}{m_newton_fb:>16s}",
                flush=True,
            )
            all_rows.append(
                (
                    cfg,
                    t_scan_fwd,
                    t_newton_fwd,
                    t_scan_fb,
                    t_newton_fb,
                    m_scan_fwd,
                    m_newton_fwd,
                    m_scan_fb,
                    m_newton_fb,
                )
            )

    # 2. Vector case (M=2)
    M = 2
    A = torch.tensor([[0.3, 0.1], [0.0, 0.2]], dtype=torch.float32, device=device)
    for B in (4, 16):
        for T in (1024, 4096, 16384):
            x = torch.randn(B, T, M, dtype=torch.float32, device=device, requires_grad=True)
            init = torch.zeros(B, M, dtype=torch.float32, device=device, requires_grad=True)

            def step_vec(carry, xt):
                y = torch.tanh(carry @ A.T + xt)
                return y, y.clone()

            def fn_vec(y, xt):
                return torch.tanh(y @ A.T + xt)

            def scan_fwd():
                with torch.no_grad():
                    scan(step_vec, init=init, xs=x, dim=1)

            def newton_fwd():
                with torch.no_grad():
                    newton_solve(fn_vec, x, init)

            def scan_fb():
                x_in = x.detach().requires_grad_(True)
                init_in = init.detach().requires_grad_(True)
                _, out = scan(step_vec, init=init_in, xs=x_in, dim=1)
                out.sum().backward()

            def newton_fb():
                x_in = x.detach().requires_grad_(True)
                init_in = init.detach().requires_grad_(True)
                out, _ = newton_solve(fn_vec, x_in, init_in)
                out.sum().backward()

            t_scan_fwd = bench(scan_fwd)
            t_newton_fwd = bench(newton_fwd)
            t_scan_fb = bench(scan_fb)
            t_newton_fb = bench(newton_fb)

            m_scan_fwd = bench_mem(scan_fwd)
            m_newton_fwd = bench_mem(newton_fwd)
            m_scan_fb = bench_mem(scan_fb)
            m_newton_fb = bench_mem(newton_fb)

            cfg = f"M={M}, B={B}, T={T}"
            print(
                f"{cfg:<22s}{t_scan_fwd:>12s}{t_newton_fwd:>14s}{t_scan_fb:>14s}{t_newton_fb:>14s}"
                f"{m_scan_fb:>14s}{m_newton_fb:>16s}",
                flush=True,
            )
            all_rows.append(
                (
                    cfg,
                    t_scan_fwd,
                    t_newton_fwd,
                    t_scan_fb,
                    t_newton_fb,
                    m_scan_fwd,
                    m_newton_fwd,
                    m_scan_fb,
                    m_newton_fb,
                )
            )

    print("\nPeak Memory (MB):")
    mem_header = (
        f"{'Configuration':<22s}"
        f"{'scan (fwd)':>12s}"
        f"{'newton (fwd)':>14s}"
        f"{'scan (f+b)':>14s}"
        f"{'newton (f+b)':>14s}"
    )
    print(mem_header)
    print("-" * len(mem_header))
    for row in all_rows:
        cfg, _, _, _, _, ms_fwd, mn_fwd, ms_fb, mn_fb = row
        print(f"{cfg:<22s}{ms_fwd:>12s}{mn_fwd:>14s}{ms_fb:>14s}{mn_fb:>14s}")


if __name__ == "__main__":
    main()
