"""Tune the prototype Helion kernels ahead of time (AOT) on this GPU.

Helion's AOT runner runs this script once per phase, with HELION_AOT_MODE set:

    pixi run python -m helion.experimental.aot_runner --output-dir .helion_aot \\
        --file-header @scripts/helion_aot_header.txt -- python scripts/helion_aot.py
    pixi run python scripts/helion_aot.py standalone

* collect: every call with a new shape autotunes it, at the effort in
  HELION_AUTOTUNE_EFFORT ("quick" unless set; "full" for a release).
* measure: every call times every configuration collected for its kernel.
* build (no script run): fits a decision tree per kernel that picks a
  configuration from the shape, written next to the kernel's source as
  ``_helion_aot_<module>_<device>_<compute capability>.py``. Every kernel of
  a source file goes into one file, so tune them all in one run.
* evaluate: checks the tree against the measurements.

The last command writes the standalone Triton files from the heuristic files,
and tidies the heuristic files' ends. Both kinds of file start with
helion_aot_header.txt, which keeps Ruff off generated code.

At run time (HELION_AOT_MODE unset, i.e. "evaluate") nothing autotunes:
:class:`philtorch.prototype._aot.Dispatch` runs the standalone file for the
GPU, else the Helion kernel with its heuristic file (or one for an older
compute capability), else the Helion kernel with a fallback configuration.

The kernels are called directly, with the shapes the ops give them: the
batch dimensions are marked ``batched`` in each kernel's decorator, so only
the matrix sizes (semiring products) or the row length (DTW) choose the
configuration, and collect tunes a few of those while measure covers more.
"""

import faulthandler
import os
import signal
import sys
from pathlib import Path

import torch

from philtorch.prototype import _dtw_helion as dtw_kernels
from philtorch.prototype import _semiring_helion as semiring_kernels

COLLECT = os.environ.get("HELION_AOT_MODE") == "collect"
HEADER = (Path(__file__).parent / "helion_aot_header.txt").read_text()

# Matrix sizes: HMM states, or DTW template frames in the matrix scan.
SIZES = (4, 16, 64) if COLLECT else (2, 4, 8, 16, 32, 64, 128)
# DTW row lengths, N for N x N cost matrices. The number of rows doesn't choose
# the configuration, so the tuning runs use a few.
LENGTHS = (256, 1024, 4096, 8192) if COLLECT else (128, 256, 512, 1024, 2048, 4096, 8192)
ROWS = 16
# soft, diag: the symmetric steps, and the orthogonal ones (also the asymmetric).
# The dtypes to write standalone files for.
DTYPES = (torch.float32, torch.float64)
DTW_VARIANTS = [(soft, diag) for soft in (False, True) for diag in (False, True)]


def tune_semiring(device):
    for K in SIZES:
        P = max(2**16 // K**2, 64)
        a, b = (torch.randn(P, K, K, device=device) for _ in range(2))
        semiring_kernels._log_bmm_kernel(a, b)
        semiring_kernels._max_bmm_kernel(a, b)
        c = semiring_kernels._log_bmm_kernel(a, b)
        # Distinct tensors throughout: Helion compiles a call whose arguments
        # alias each other for that aliasing, and reuses it for any call.
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


def standalone_variants(device, dtype):
    """(kernel, example arguments) for every kernel variant.

    The example arguments fix the dtype and the constexpr arguments: Helion
    specializes the generated code on them, so each combination gets its own
    file. It also
    specializes on sizes of 1 and on arguments that alias each other, so
    every size is at least 2 and every tensor distinct. The kernels use
    dynamic shapes, so the code serves every shape.
    """

    def mats(n):
        return [torch.randn(4, 3, 3, device=device, dtype=dtype) for _ in range(n)]

    for kernel in (semiring_kernels._log_bmm_kernel, semiring_kernels._max_bmm_kernel):
        yield kernel, tuple(mats(2))
    for kernel in (
        semiring_kernels._contract_over_i,
        semiring_kernels._contract_over_j,
        semiring_kernels._contract_over_k,
    ):
        for is_max in (False, True):
            yield kernel, (*mats(6), is_max)
    costs = [torch.rand(2, 3, 5, device=device, dtype=dtype) for _ in range(4)]
    for soft, diag in DTW_VARIANTS:
        yield dtw_kernels._dtw_dp_kernel, (costs[0], soft, diag)
    yield dtw_kernels._dag_forward_kernel, tuple(costs)


def write_standalone(device):
    """Write standalone Triton files, which need no Helion, from the heuristic files.

    Helion's own ``--standalone`` writes one file per kernel, specialized on
    whichever constexpr values the kernel was first called with; this writes
    one per variant instead, where :mod:`philtorch.prototype._aot` looks.
    """
    import importlib.util
    import tempfile

    from helion.autotuner.aot_cache import AOTAutotuneCache, find_heuristic_file
    from helion.experimental.aot_compile import generate_standalone_file
    from helion.runtime.config import Config

    from philtorch.prototype._aot import compute_capability, standalone_path

    cc = compute_capability(device)
    variants = [v for dtype in DTYPES for v in standalone_variants(device, dtype)]
    for kernel, args in variants:
        name = kernel.name
        heuristic_file = find_heuristic_file(kernel.fn.__code__.co_filename, kernel_name=name)
        if heuristic_file is None:
            raise RuntimeError(f"no heuristic file for {name}: run the AOT runner first")
        spec = importlib.util.spec_from_file_location("heuristic", heuristic_file)
        heuristic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(heuristic)
        # The decision tree backend keeps its configurations inside autotune_<kernel>.
        configs = getattr(heuristic, "CONFIGS", None)
        if configs is None:
            configs = AOTAutotuneCache._parse_configs_from_autotune(heuristic, name)
        # Ruff's format check wants one newline at the end, even with fmt: off.
        heuristic_file.write_text(heuristic_file.read_text().rstrip("\n") + "\n")
        bound = kernel.bind(args)
        codes = [bound.to_triton_code(Config(**config)) for config in configs]
        target = standalone_path(kernel, args, cc)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_standalone_file(
                kernel_name=name,
                triton_codes=codes,
                heuristic_code=heuristic_file.read_text(),
                output_dir=Path(tmp),
            )
            target.write_text(HEADER + path.read_text())
        print(f"{target}: {len(codes)} configs", flush=True)


def main():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(device), flush=True)
    which = sys.argv[1:] or ["semiring", "dtw"]
    if which == ["standalone"]:
        write_standalone(device)
        return
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
