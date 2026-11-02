"""Benchmark the scalar (M = 1) linear recurrence on MPS against the CPU kernels.

Measures ``torch.ops.philtorch.lti_recur`` (time-invariant) and
``torch.ops.philtorch.scan`` (time-varying) on the MPS device and, where the CPU
kernel takes the dtype, on the CPU, over a grid of batch sizes B and lengths T.
Three coefficient layouts are timed:

* ``shared``: ``lti_recur`` with ``a`` of shape (1,), one coefficient for all items;
* ``batched``: ``lti_recur`` with ``a`` of shape (B,), one coefficient per item
  (skipped at B = 1, where it is the same as ``shared``);
* ``per_step``: ``scan`` with ``decay`` of shape (B, T), one coefficient per step.

Each is run in float32, float16, bfloat16 and complex64. The CPU column runs the
same op in the same dtype, except for ``scan`` in float16 and bfloat16, which the
CPU kernel does not take (its column is ``n/a``). The inputs are stable
recurrences: ``|a| < 0.9``, with a random phase for complex64.

Per case it reports the median runtime (and IQR) from
``torch.utils.benchmark.Timer.blocked_autorange``, synchronizing the MPS device
at each measurement boundary, the speedup of MPS over CPU, and the peak memory
of one call on each device. The MPS result is first checked against the CPU op
run in float32 (complex64) on the same inputs.

Run it with no other GPU work going on::

    pixi run python benchmarks/benchmark_mps_recur.py
    pixi run python benchmarks/benchmark_mps_recur.py --dtype float16 --layout per_step
"""

import argparse
import gc
import platform
import threading
import time

import torch
import torch.utils.benchmark as benchmark

import philtorch  # noqa: F401 - loads the native operator registrations

CASES = (
    (1, 4_096),
    (1, 262_144),
    (16, 4_096),
    (64, 65_536),
    (256, 4_096),
)
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "complex64": torch.complex64,
}
LAYOUTS = ("shared", "batched", "per_step")
TOLERANCES = {
    torch.float32: (1e-5, 1e-6),
    torch.float16: (2e-2, 2e-2),
    torch.bfloat16: (5e-2, 5e-2),
    torch.complex64: (1e-5, 1e-6),
}
# scan's CPU kernel takes no half precision; lti_recur's takes all four dtypes.
SCAN_CPU_DTYPES = (torch.float32, torch.complex64)
MAX_DECAY = 0.9
NUM_THREADS = torch.get_num_threads()


def random_decay(shape, dtype):
    """Stable coefficients: ``|a| < MAX_DECAY``, with a random phase if complex."""
    if dtype.is_complex:
        magnitude = torch.rand(shape) * MAX_DECAY
        return torch.polar(magnitude, torch.rand(shape) * 2 * torch.pi).to(dtype)
    return ((torch.rand(shape) * 2 - 1) * MAX_DECAY).to(dtype)


def make_function(layout, a, zi, x):
    """The op for a layout, bound to its inputs, as a function of no arguments."""
    if layout == "per_step":
        return lambda: torch.ops.philtorch.scan(x, a, zi)
    return lambda: torch.ops.philtorch.lti_recur(a, zi, x)


def cpu_supports(layout, dtype):
    return layout != "per_step" or dtype in SCAN_CPU_DTYPES


def synchronized_mps_timer():
    torch.mps.synchronize()
    return time.perf_counter()


def measure_runtime(function, timer=time.perf_counter):
    measurement = benchmark.Timer(
        stmt="function()",
        globals={"function": function},
        timer=timer,
        num_threads=NUM_THREADS,
    ).blocked_autorange(min_run_time=0.5)
    return measurement.median * 1_000, measurement.iqr * 1_000


def profile_cpu_peak(function):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        acc_events=True,
    ) as profiler:
        result = function()

    memory_changes = []
    for event in profiler.events():
        memory_change = event.self_cpu_memory_usage
        if memory_change:
            timestamp = event.time_range.start if memory_change > 0 else event.time_range.end
            memory_changes.append((timestamp, memory_change))

    current = 0
    peak = 0
    for _, memory_change in sorted(memory_changes, key=lambda change: change[0]):
        current += memory_change
        peak = max(peak, current)

    del result
    return peak


def profile_mps_peak(function, repeats=20):
    torch.mps.synchronize()
    gc.collect()
    torch.mps.empty_cache()
    baseline = torch.mps.current_allocated_memory()
    peak = [baseline]
    ready = threading.Event()
    stop = threading.Event()

    def sample_allocator():
        ready.set()
        while not stop.is_set():
            peak[0] = max(peak[0], torch.mps.current_allocated_memory())

    sampler = threading.Thread(target=sample_allocator, daemon=True)
    sampler.start()
    ready.wait()
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True
        ):
            with torch.profiler.record_function("mps_recur_memory"):
                for _ in range(repeats):
                    result = function()
                    torch.mps.synchronize()
                    del result
    finally:
        stop.set()
        sampler.join()

    return peak[0] - baseline


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the scalar linear recurrence on MPS (lti_recur and scan) "
            "against the CPU kernels: runtime and peak memory over a grid of "
            "batch sizes and lengths."
        )
    )
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=tuple(DTYPES),
        default=list(DTYPES),
        help="dtypes to run (default: all)",
    )
    parser.add_argument(
        "--layout",
        nargs="+",
        choices=LAYOUTS,
        default=list(LAYOUTS),
        help="coefficient layouts to run, as in the module docstring (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU threads: {NUM_THREADS}")
    print()
    print(
        "| dtype | B | T | Coefficients | CPU median ms (IQR) | "
        "MPS median ms (IQR) | Speedup | CPU peak MiB | "
        "MPS peak MiB | Memory ratio |"
    )
    print("| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")

    torch.manual_seed(5)
    for dtype_name in args.dtype:
        dtype = DTYPES[dtype_name]
        for layout in args.layout:
            for batch, steps in CASES:
                if layout == "batched" and batch == 1:
                    continue
                if layout == "shared":
                    a_shape = (1,)
                elif layout == "batched":
                    a_shape = (batch,)
                else:
                    a_shape = (batch, steps)
                a_cpu = random_decay(a_shape, dtype)
                zi_cpu = torch.randn(batch, dtype=dtype)
                x_cpu = torch.randn(batch, steps, dtype=dtype)
                cpu_inputs = (a_cpu, zi_cpu, x_cpu)
                mps_inputs = tuple(value.to("mps") for value in cpu_inputs)
                mps_function = make_function(layout, *mps_inputs)

                # The CPU op in float32 (complex64) on the same inputs.
                reference_dtype = torch.promote_types(dtype, torch.float32)
                expected = make_function(
                    layout, *(value.to(reference_dtype) for value in cpu_inputs)
                )()
                actual = mps_function().cpu()
                rtol, atol = TOLERANCES[dtype]
                torch.testing.assert_close(
                    actual.to(reference_dtype), expected, rtol=rtol, atol=atol
                )
                del actual, expected

                mps_ms, mps_iqr_ms = measure_runtime(mps_function, timer=synchronized_mps_timer)
                mps_mib = profile_mps_peak(mps_function) / 2**20
                if cpu_supports(layout, dtype):
                    cpu_function = make_function(layout, *cpu_inputs)
                    cpu_ms, cpu_iqr_ms = measure_runtime(cpu_function)
                    cpu_mib = profile_cpu_peak(cpu_function) / 2**20
                    cpu_cells = (
                        f"{cpu_ms:.4f} ({cpu_iqr_ms:.4f})",
                        f"{cpu_ms / mps_ms:.2f}x",
                        f"{cpu_mib:.3f}",
                    )
                    memory_ratio = f"{cpu_mib / mps_mib:.2f}x"
                else:
                    cpu_cells = ("n/a", "n/a", "n/a")
                    memory_ratio = "n/a"
                print(
                    f"| {dtype_name} | {batch} | {steps} | {layout} | "
                    f"{cpu_cells[0]} | {mps_ms:.4f} ({mps_iqr_ms:.4f}) | "
                    f"{cpu_cells[1]} | {cpu_cells[2]} | {mps_mib:.3f} | "
                    f"{memory_ratio} |"
                )


if __name__ == "__main__":
    main()
