import gc
import platform
import threading
import time

import torch
import torch.utils.benchmark as benchmark

import philtorch  # noqa: F401 - loads the native operator registrations

CASES = (
    (1, 4_096, False),
    (1, 262_144, False),
    (16, 4_096, True),
    (64, 65_536, True),
    (256, 4_096, True),
)
NUM_THREADS = torch.get_num_threads()


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
            timestamp = (
                event.time_range.start if memory_change > 0 else event.time_range.end
            )
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
            with torch.profiler.record_function("mps_lti_recur_memory"):
                for _ in range(repeats):
                    result = function()
                    torch.mps.synchronize()
                    del result
    finally:
        stop.set()
        sampler.join()

    return peak[0] - baseline


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU threads: {NUM_THREADS}")
    print()
    print(
        "| B | T | Coefficients | CPU median ms (IQR) | "
        "MPS median ms (IQR) | Speedup | CPU peak MiB | "
        "MPS peak MiB | Memory ratio |"
    )
    print("| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")

    torch.manual_seed(5)
    for batch, steps, batched_decay in CASES:
        coefficient_count = batch if batched_decay else 1
        a_cpu = (torch.rand(coefficient_count) - 0.5) * 0.9
        zi_cpu = torch.randn(batch)
        x_cpu = torch.randn(batch, steps)
        mps_inputs = tuple(value.to("mps") for value in (a_cpu, zi_cpu, x_cpu))

        cpu_function = lambda: torch.ops.philtorch.lti_recur(a_cpu, zi_cpu, x_cpu)
        mps_function = lambda: torch.ops.philtorch.lti_recur(*mps_inputs)

        expected = cpu_function()
        actual = mps_function().cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        del actual, expected

        cpu_ms, cpu_iqr_ms = measure_runtime(cpu_function)
        mps_ms, mps_iqr_ms = measure_runtime(mps_function, timer=synchronized_mps_timer)
        cpu_mib = profile_cpu_peak(cpu_function) / 2**20
        mps_mib = profile_mps_peak(mps_function) / 2**20
        coefficients = "batched" if batched_decay else "shared"
        print(
            f"| {batch} | {steps} | {coefficients} | "
            f"{cpu_ms:.4f} ({cpu_iqr_ms:.4f}) | "
            f"{mps_ms:.4f} ({mps_iqr_ms:.4f}) | "
            f"{cpu_ms / mps_ms:.2f}x | {cpu_mib:.3f} | {mps_mib:.3f} | "
            f"{cpu_mib / mps_mib:.2f}x |"
        )


if __name__ == "__main__":
    main()
