"""Tune the Helion versions of the prototype kernels ahead of time (dev only).

The production kernels are hand-written Triton (philtorch/prototype/
_semiring_triton.py and _dtw_triton.py). Their Helion originals, semiring.py
and dtw.py here, are tuned with Helion's AOT runner to learn which
configurations win for which shapes; those go into the Triton modules by
hand. From the repository root:

    pixi run python -m helion.experimental.aot_runner --output-dir .helion_aot \\
        --file-header @scripts/helion/aot_header.txt -- python scripts/helion/aot.py
    pixi run python scripts/helion/aot.py tidy

The runner runs this script once per phase, with HELION_AOT_MODE set:

* collect: every call with a new shape autotunes it, at the effort in
  HELION_AUTOTUNE_EFFORT ("quick" unless set).
* measure: every call times every configuration collected for its kernel.
* build (no script run): fits a decision tree per kernel that picks a
  configuration from the shape, written next to the kernel's source as
  ``_helion_aot_<module>_<device>_<compute capability>.py``: the record the
  Triton configurations are read from. Every kernel of a module goes into
  one file, so tune them all in one run.
* evaluate: checks the tree against the measurements.

The batch dimensions are marked ``batched`` in each kernel's decorator, so
only the matrix sizes (semiring products) or the row length (DTW) choose the
configuration; collect tunes a few of those while measure covers more. The
inputs are distinct tensors with sizes of at least 2: Helion compiles a first
call's aliased arguments or size-1 dimensions into code it reuses for every
call. The last command trims the generated files' ends for Ruff, which
aot_header.txt otherwise keeps off them.
"""

import faulthandler
import os
import signal
import sys
from pathlib import Path

import dtw as dtw_kernels
import semiring as semiring_kernels
import torch

COLLECT = os.environ.get("HELION_AOT_MODE") == "collect"

# Matrix sizes: HMM states, or DTW template frames in the matrix scan.
SIZES = (4, 16, 64) if COLLECT else (2, 4, 8, 16, 32, 64, 128)
# DTW row lengths, N for N x N cost matrices. The number of rows doesn't choose
# the configuration, so the tuning runs use a few.
LENGTHS = (256, 1024, 4096, 8192) if COLLECT else (128, 256, 512, 1024, 2048, 4096, 8192)
ROWS = 16
# soft, diag: the symmetric steps, and the orthogonal ones (also the asymmetric).
DTW_VARIANTS = [(soft, diag) for soft in (False, True) for diag in (False, True)]


def tune_semiring(device):
    for K in SIZES:
        P = max(2**16 // K**2, 64)
        a, b = (torch.randn(P, K, K, device=device) for _ in range(2))
        semiring_kernels._log_bmm_kernel(a, b)
        semiring_kernels._max_bmm_kernel(a, b)
        c = semiring_kernels._log_bmm_kernel(a, b)
        x, y, z = (t.clone() for t in (a, c, b))
        for kernel in (
            semiring_kernels._contract_over_i,
            semiring_kernels._contract_over_j,
            semiring_kernels._contract_over_k,
        ):
            for is_max in (False, True):
                kernel(a, b, c, x, y, z, is_max)


def tune_dtw(device):
    for L in LENGTHS:
        cost = torch.rand(4, ROWS, L, device=device)
        for soft, diag in DTW_VARIANTS:
            dtw_kernels._dtw_dp_kernel(cost, soft, diag)
        weights = [torch.rand_like(cost) / 3 for _ in range(3)]
        dtw_kernels._dag_forward_kernel(*weights, cost)


def tidy():
    """One newline at the end of each heuristic file, as Ruff's format check wants."""
    for path in Path(__file__).parent.glob("_helion_aot_*.py"):
        path.write_text(path.read_text().rstrip("\n") + "\n")


def main():
    if sys.argv[1:] == ["tidy"]:
        tidy()
        return
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(device), flush=True)
    which = sys.argv[1:] or ["semiring", "dtw"]
    if "semiring" in which:
        tune_semiring(device)
    if "dtw" in which:
        tune_dtw(device)
    torch.cuda.synchronize()


if __name__ == "__main__":
    # The guard matters: Helion's autotuner spawns subprocesses.
    # `kill -USR1 <pid>` prints every thread's stack, for a run that seems stuck.
    faulthandler.register(signal.SIGUSR1)
    main()
