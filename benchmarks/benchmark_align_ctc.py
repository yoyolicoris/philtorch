"""Benchmark philtorch.align.ctc_loss against torch.nn.functional.ctc_loss.

Run from the repository root on a CUDA machine:

    pixi run python benchmarks/benchmark_align_ctc.py

Times are for the losses and their gradient with respect to the logits,
through a log_softmax, after a warm-up call that compiles the kernels.
"""

import torch
import torch.nn.functional as F
from _timing import bench

from philtorch.align import ctc_loss


def main():
    print(torch.cuda.get_device_name(), flush=True)
    C = 32
    for B in (1, 16):
        for N in (1_000, 10_000):
            for U in (4, 15, 31, 63, 127):
                logits = torch.randn(B, N, C, device="cuda", requires_grad=True)
                targets = torch.randint(1, C, (B, U), device="cuda")
                lengths = torch.full((B,), N), torch.full((B,), U)

                def ours():
                    loss = ctc_loss(logits.log_softmax(-1), targets, *lengths, reduction="sum")
                    torch.autograd.grad(loss, logits)

                def pytorch():
                    log_probs = logits.log_softmax(-1).transpose(0, 1)
                    loss = F.ctc_loss(log_probs, targets, *lengths, reduction="sum")
                    torch.autograd.grad(loss, logits)

                print(
                    f"B={B:2d} N={N:>6,} U={U:3d}: philtorch {bench(ours)} | "
                    f"F.ctc_loss {bench(pytorch)}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
