"""Benchmark the general recurrence operator against the unrolled recursion.

Run from the repository root on a CUDA machine:

    pixi run python benchmarks/benchmark_recur.py

Times h[t] = A[t] h[t - 1] + x[t] with philtorch._recur.recurrence and with
philtorch.lti / philtorch.lpv's state_space_recursion at unroll_factor=32,
the block-unrolled PyTorch recursion and the fastest path today where no
kernel exists, forward alone and forward plus backward, after a warm-up call
that compiles the kernels.
"""

import time

import torch

from philtorch._recur import recurrence
from philtorch.lpv import ssm as lpv_ssm
from philtorch.lti import ssm as lti_ssm


def bench(fn, budget=1.0):
    """Mean milliseconds per call, or the error's name."""
    try:
        fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        n = max(1, min(20, int(budget / max(time.perf_counter() - start, 1e-6))))
        start = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return f"{1e3 * (time.perf_counter() - start) / n:9.2f}"
    except (torch.OutOfMemoryError, RuntimeError, NotImplementedError) as e:
        return f"{type(e).__name__[:9]:>9}"


def main():
    print(torch.cuda.get_device_name(), flush=True)
    columns = ["op", "unroll32", "op+bw", "unroll32+bw"]
    print(f"{'case':36s}" + "".join(f"{c:>11s}" for c in columns), flush=True)
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.complex64):
        for varying in (False, True):
            for M in (2, 3, 4, 8, 16):
                for B, T in [(1, 65536), (16, 65536), (256, 4096)]:
                    A = torch.randn(B if varying else 1, T if varying else 1, M, M, dtype=dtype,
                                    device="cuda")  # fmt: skip
                    A = A / torch.linalg.matrix_norm(A, ord=2, keepdim=True) * 0.97
                    x = torch.randn(B, T, M, dtype=dtype, device="cuda")
                    zi = torch.zeros(B, M, dtype=dtype, device="cuda")
                    # state_space_recursion's layouts: (M, M), or (T, M, M) shared
                    # over the batch, or (B, T, M, M).
                    A_ssm = A[0, 0] if not varying else A
                    ssm = lpv_ssm if varying else lti_ssm
                    inputs = [t.detach().requires_grad_() for t in (A, zi, x)]
                    ssm_inputs = [A_ssm.detach().requires_grad_(), *inputs[1:]]

                    def backward(fn, args):
                        h = fn(*args)
                        torch.autograd.grad(h.abs().square().sum(), args)

                    def unrolled(A_, zi_, x_):
                        return ssm.state_space_recursion(A_, zi_, x_, unroll_factor=32)

                    times = {
                        "op": bench(lambda: recurrence(A, zi, x)),
                        "unroll32": bench(lambda: unrolled(A_ssm, zi, x)),
                        "op+bw": bench(lambda: backward(recurrence, inputs)),
                        "unroll32+bw": bench(lambda: backward(unrolled, ssm_inputs)),
                    }
                    case = (
                        f"{str(dtype)[6:]:9s} {'LPV' if varying else 'LTI'} M={M:2d} "
                        f"B={B:3d} T={T:5d}"
                    )
                    print(f"{case:36s}" + "".join(f"{times[c]:>11s}" for c in columns), flush=True)


if __name__ == "__main__":
    main()
