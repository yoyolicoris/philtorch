"""Benchmark the linear recurrence on MPS against the CPU kernels.

Times the native ops, on MPS and, where its kernel takes the dtype, on the CPU,
over batch sizes B, lengths T and state sizes M: ``lti_recur`` and ``scan`` for
M = 1, ``lti_recurN`` and ``recurN`` above. The coefficient layouts are

* ``shared``: time-invariant, one coefficient or M x M matrix for all items;
* ``batched``: time-invariant, one per item (from B = 2);
* ``per_step``: time-varying, one per step of each item;
* ``per_step_shared``: time-varying, one per step shared over the batch
  (M > 1, from B = 2).

Cases of more than 2^26 coefficients or inputs above 1 GiB (which the CPU and
MPS copies and the outputs make 4 GiB) are skipped, and complex64 runs up to
M = 64, the kernels' limit. The recurrences are stable: ``|a| < 0.9``, with a
random phase for complex64, and matrices of spectral norm 0.9; int64's
coefficients, states and inputs are random 62-bit integers, whose products
wrap around. Each case
reports the median runtime (and IQR) of
``torch.utils.benchmark.Timer.blocked_autorange``, synchronizing the MPS device
at each measurement boundary, and the speed-up of MPS over CPU.

Run it with no other GPU work going on::

    pixi run python benchmarks/benchmark_mps_recur.py
    pixi run python benchmarks/benchmark_mps_recur.py --dtype float16 --layout per_step
    pixi run python benchmarks/benchmark_mps_recur.py --states 4 16 --dtype float32
"""

import argparse
import platform

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
    "int64": torch.int64,
}
LAYOUTS = ("shared", "batched", "per_step", "per_step_shared")
STATES = (1, 2, 4, 8, 16, 32, 64, 128)
MAX_COEFFICIENTS = 2**26
MAX_INPUT_BYTES = 2**30
# The Metal kernels' largest M, half that for complex64.
MAX_STATES = 128
MAX_DECAY = 0.9
NUM_THREADS = torch.get_num_threads()


def random_integers(shape):
    return torch.randint(-(2**62), 2**62, shape)


def random_values(shape, dtype):
    """Random states and inputs."""
    if dtype == torch.int64:
        return random_integers(shape)
    return torch.randn(shape, dtype=dtype)


def random_decay(shape, dtype):
    """Stable coefficients: ``|a| < MAX_DECAY``, with a random phase if complex."""
    if dtype == torch.int64:
        return random_integers(shape)
    if dtype.is_complex:
        magnitude = torch.rand(shape) * MAX_DECAY
        return torch.polar(magnitude, torch.rand(shape) * 2 * torch.pi).to(dtype)
    return ((torch.rand(shape) * 2 - 1) * MAX_DECAY).to(dtype)


def random_matrices(shape, dtype):
    """Stable matrices: each of spectral norm ``MAX_DECAY``."""
    if dtype == torch.int64:
        return random_integers(shape)
    wide = torch.complex128 if dtype.is_complex else torch.float64
    A = torch.randn(shape, dtype=wide)
    return (A / torch.linalg.matrix_norm(A, ord=2, keepdim=True) * MAX_DECAY).to(dtype)


def coefficient_shape(layout, batch, steps, states):
    """The coefficients' shape, or None where the layout is skipped."""
    if layout in ("batched", "per_step_shared") and batch == 1:
        return None
    if states == 1:
        return {"shared": (1,), "batched": (batch,), "per_step": (batch, steps)}.get(layout)
    matrix = (states, states)
    return {
        "shared": matrix,
        "batched": (batch, *matrix),
        "per_step": (batch, steps, *matrix),
        "per_step_shared": (steps, *matrix),
    }[layout]


def make_function(layout, states, a, zi, x):
    """The op for a layout, bound to its inputs, as a function of no arguments."""
    ops = torch.ops.philtorch
    if states > 1:
        op = ops.lti_recurN if layout in ("shared", "batched") else ops.recurN
        return lambda: op(a, zi, x)
    if layout == "per_step":
        return lambda: ops.scan(x, a, zi)
    return lambda: ops.lti_recur(a, zi, x)


def measure_runtime(function):
    """The median and IQR, in ms, of ``function()`` by ``Timer.blocked_autorange``,
    whose clock synchronizes the accelerator, MPS included, at each boundary."""
    # Called once first, untimed: the first call of a state size and dtype
    # compiles its Metal library.
    function()
    measurement = benchmark.Timer(
        stmt="function()", globals={"function": function}, num_threads=NUM_THREADS
    ).blocked_autorange(min_run_time=0.5)
    return measurement.median * 1_000, measurement.iqr * 1_000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the linear recurrence on MPS against the CPU kernels, over "
            "a grid of state sizes, batch sizes and lengths."
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
    parser.add_argument(
        "--states",
        nargs="+",
        type=int,
        default=list(STATES),
        help="state sizes M to run (default: %(default)s)",
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
        "| M | dtype | B | T | Coefficients | CPU median ms (IQR) | MPS median ms (IQR) | Speedup |"
    )
    print("| ---: | :--- | ---: | ---: | :--- | ---: | ---: | ---: |")

    torch.manual_seed(5)
    for states in args.states:
        for dtype_name in args.dtype:
            dtype = DTYPES[dtype_name]
            for layout in args.layout:
                for batch, steps in CASES:
                    shape = coefficient_shape(layout, batch, steps, states)
                    if shape is None or torch.Size(shape).numel() > MAX_COEFFICIENTS:
                        continue
                    if states > (MAX_STATES // 2 if dtype.is_complex else MAX_STATES):
                        continue
                    if batch * steps * states * dtype.itemsize > MAX_INPUT_BYTES:
                        continue
                    if states == 1:
                        a_cpu = random_decay(shape, dtype)
                        zi_cpu = random_values((batch,), dtype)
                        x_cpu = random_values((batch, steps), dtype)
                    else:
                        a_cpu = random_matrices(shape, dtype)
                        zi_cpu = random_values((batch, states), dtype)
                        x_cpu = random_values((batch, steps, states), dtype)
                    cpu_inputs = (a_cpu, zi_cpu, x_cpu)
                    mps_inputs = tuple(value.to("mps") for value in cpu_inputs)
                    mps_function = make_function(layout, states, *mps_inputs)

                    mps_ms, mps_iqr_ms = measure_runtime(mps_function)
                    cpu_function = make_function(layout, states, *cpu_inputs)
                    cpu_ms, cpu_iqr_ms = measure_runtime(cpu_function)
                    print(
                        f"| {states} | {dtype_name} | {batch} | {steps} | {layout} | "
                        f"{cpu_ms:.4f} ({cpu_iqr_ms:.4f}) | {mps_ms:.4f} ({mps_iqr_ms:.4f}) | "
                        f"{cpu_ms / mps_ms:.2f}x |",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
