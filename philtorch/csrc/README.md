# Metal scalar recurrence implementation

This directory contains philtorch's native CPU, CUDA, and Metal sources.
The Metal implementation of the scalar linear recurrence, time-invariant and time-varying, lives in [`scalar_recur_mps.mm`](scalar_recur_mps.mm) (host code) and [`scalar_recur_mps.metal`](scalar_recur_mps.metal) (kernels).
It implements

```text
h[b, t] = a * (t == 0 ? zi[b] : h[b, t - 1]) + x[b, t]
```

with `a` one coefficient per step (`philtorch::scan`, time-varying) or per sequence, shared or not (`philtorch::lti_recur`, time-invariant), as an associative affine scan.
It takes float32, float16, bfloat16 and complex64, accumulating float16 and bfloat16 in float32.
One three-pass design serves every input: reduce each tile, scan the tile totals of each sequence, and replay each tile from the state before it.

## Source map

| File | Role |
| --- | --- |
| [`ops.cpp`](ops.cpp) | Creates the `philtorch._C` module and defines the dispatcher schemas of the recurrence ops, `philtorch::lti_recur` among them. |
| [`host_recur.cpp`](host_recur.cpp) | CPU kernels of the recurrence ops, `philtorch::scan` included, split over time when a batch has fewer sequences than threads, long enough for the split to pay. |
| [`host_dot.h`](host_dot.h) | The dot product of the CPU kernels of the recurrence ops and of `lpc`, vectorized for float and double from size 8. |
| [`torchlpc_shim.cpp`](torchlpc_shim.cpp) | Defines the schemas of `philtorch::scan` and `philtorch::lpc`, and the CPU kernel of `lpc`, one sequence per thread. |
| [`scalar_recur_mps.mm`](scalar_recur_mps.mm) | MPS host code of `lti_recur` and `scan`, on torch's stable ABI and Metal C shim. |
| [`scalar_recur_mps.metal`](scalar_recur_mps.metal) | The Metal kernels, embedded in the extension at build time. |
| [`../_ops.py`](../_ops.py) | Registers fake implementations and autograd for `philtorch::lti_recur` and `philtorch::scan`. |
| [`../_recur.py`](../_recur.py) | Sends M = 1 recurrences on MPS to `lti_recur` or `scan`. |
| [`../../tests/test_recur_ext.py`](../../tests/test_recur_ext.py) | Checks CPU/MPS parity, boundary lengths, long scans, and gradients. |
| [`../../benchmarks/benchmark_mps_recur.py`](../../benchmarks/benchmark_mps_recur.py) | Reproduces CPU/MPS runtime and working-memory results. |

## Turning a recurrence into a scan

A `Pair { m, v }` represents the affine map

```text
f(h) = m * h + v.
```

`compose(left, right)` applies two adjacent maps in sequence:

```text
compose(left, right)
    = (left.m * right.m, left.v * right.m + right.v)
    = right(left(h)).
```

Composition is associative, so the maps can be evaluated with a parallel prefix scan.
It is not commutative; changing the operand order changes the recurrence.
For complex64, `m` and `v` are complex numbers (`float2`), multiplied as such.

Step `t` of sequence `b` is the map

```text
p[0] = (0, a * zi[b] + x[b, 0])
p[t] = (a, x[b, t])            for t > 0
identity = (1, 0)              past the end
```

The first map folds the initial state in, and its zero multiplier makes every map composed from the start of a sequence a constant: its `m` is zero and its `v` is the state.
So the inclusive prefix at step `t` holds `h[b, t]` in `v`, and a tile needs only the state at the end of the tile before it, while the first tile needs none.
Steps past the end are identities, so partial tiles need no padding pass.

## Execution

```mermaid
flowchart TB
    opcall["lti_recur(a, zi, x) or scan(x, decay, zi)<br/>MPS dispatcher"]
    host["scalar_recur_mps<br/>validate, make inputs contiguous, allocate scratch"]

    subgraph command["PyTorch's MPS stream, through torch's Metal shim"]
        direction TB

        subgraph stage1["1. reduce_{dtype}_{lti,lpv}, if more than one tile"]
            direction LR
            grid1["Grid: n_tiles x B<br/>256 threads per group"]
            load["Each thread loads<br/>4 consecutive steps, one vector"]
            simd["8 SIMD32 scans<br/>of the thread totals"]
            totals["totals<br/>B x n_tiles maps"]
            grid1 --> load --> simd --> totals
        end

        subgraph stage2["2. scan_{float,cfloat}, if more than one tile"]
            direction LR
            grid2["Grid: B<br/>one 256-thread group per sequence"]
            chunks["Scan 512 tile totals per chunk<br/>carry the map between chunks"]
            states["states<br/>B x n_tiles end states"]
            grid2 --> chunks --> states
        end

        subgraph stage3["3. replay_{dtype}_{lti,lpv}"]
            direction LR
            grid3["Grid: n_tiles x B<br/>256 threads per group"]
            reload["Reload 4 steps<br/>per thread"]
            simd2["Repeat the tile's<br/>SIMD scan"]
            combine["Start from the previous<br/>tile's end state"]
            output["output<br/>B x T, one vector store per thread"]
            grid3 --> reload --> simd2 --> combine --> output
        end

        totals --> grid2
        states --> combine
    end

    opcall --> host --> grid1
    output --> result["h[t] = a h[t - 1] + x[t]"]
```

Let

```text
TILE    = 256 threads * 4 steps = 1024 steps
n_tiles = ceil(T / TILE)
```

The kernels are encoded in order on PyTorch's current MPS stream, which orders them without a CPU round trip.
A sequence of one tile runs the replay alone: one dispatch, which reads its input once.

### Stage 1: reduce tiles

`reduce` runs one 256-thread threadgroup for every `(tile, sequence)` pair.
Thread `lid` loads steps `4 * lid` to `4 * lid + 3` of its tile with one vector load (`packed_float4` and the like, which need only the element's alignment), so a SIMD group reads 128 consecutive steps.
Time-varying coefficients are loaded the same way; a time-invariant one is read once per thread.

1. Each thread composes its four maps into one total.
2. `simd_inclusive` scans those totals with `simd_shuffle_up` at offsets 1, 2, 4, 8, and 16.
3. Lane 31 of each SIMD group writes its total to threadgroup memory, and SIMD group 0 scans the eight totals.
4. The last thread writes the tile's map to `totals`.

### Stage 2: scan tile totals

`scan`, one kernel per accumulation type (float for float32, float16 and bfloat16, and float2 for complex64), runs one 256-thread threadgroup per sequence, which scans its tile totals two per thread, 512 at a time, carrying the map so far between chunks of 512.
It writes `v`, the state, of each inclusive prefix to `states`: the state at the end of each tile.

### Stage 3: replay tiles

`replay` runs the same grid as stage 1, reloads the steps, and repeats the tile's scan, which gives each thread the map from the tile's start to its first step.
Applied to the end state of the previous tile (zero for the first tile, whose first map is constant), it gives the thread's starting state, from which the thread runs its four steps in sequence and writes them with one vector store.
Replaying the scan costs SIMD arithmetic, but rereads the input instead of writing and rereading per-step prefixes.

## Buffers and launch geometry

| Buffer | Shape | Purpose |
| --- | --- | --- |
| `decays` | `(1,)`, `(B,)` or `(B, T)` | Shared, per-sequence or per-step coefficient. |
| `zi` | `(B,)` | Initial state. |
| `x` | `(B, T)` | Input sequence. |
| `totals` | `(B, n_tiles)` maps | One map per tile, two values of the accumulation type. |
| `states` | `(B, n_tiles)` | The state at the end of each tile. |
| `output` | `(B, T)` | Recurrence result. |

Scratch, in float32 (two floats per value for complex64), is `3 * B * n_tiles` values, and none for sequences of one tile.
The dominant traffic is reading `x` (and per-step coefficients) twice and writing `output` once.

| Property | Value |
| --- | --- |
| Tile size | 1024 steps |
| Threads per tile | 256 |
| Steps per thread | 4, one vector load and store |
| SIMD width | 32 lanes |
| SIMD groups per tile | 8 |
| Metal dispatches | 3, or 1 for sequences of one tile |

The tile, threadgroup, SIMD and vector constants are coupled: `THREADS`, `K` and `TOTALS` in the Metal source, `THREADS` and `STEPS_PER_THREAD` in the host code.
Every kernel declares 256 threads as its maximum per threadgroup (`max_total_threads_per_threadgroup`), so the compiler guarantees that size; every Apple GPU has 32-wide SIMD groups, which the kernels assume, as the shim gives no way to query the width.
Each kernel indexes the steps of a sequence in 32 bits, as 64-bit integer arithmetic is emulated on the GPU: per-sequence pointers are computed once per thread, and the host rejects sequences of 2^32 - 1024 steps or more.

## PyTorch and Metal integration

The host code uses only torch's stable ABI, `torch/csrc/stable` and `torch/headeronly`, and torch's Metal C shim, [`shim_mps.h`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/inductor/aoti_torch/c/shim_mps.h), so the compiled code calls torch through C functions alone.
It is held to the API of torch 2.10, the oldest torch that builds it, by `TORCH_TARGET_VERSION`, which `setup.py` generates from `build_support.MIN_TORCH_MACOS`, also the floor of macOS source builds.
No Objective-C remains, and the extension links no Apple framework.

1. `setup.py` turns `scalar_recur_mps.metal` into `scalar_recur_mps.metal.h`, a raw string, and writes `torch_target_version.h`, both in `build/metal` (`build_support.write_metal_headers`), which the host code includes.
2. On first use, `aoti_torch_mps_create_shader_library` compiles that source, and `aoti_torch_mps_get_kernel_function` looks up a dtype's kernels, which stay cached for the process.
   The bfloat kernels exist only from Metal 3.1 on, so only bfloat16 inputs would fail without them.
3. Scratch tensors are ordinary MPS tensors from the stable `new_empty`, so PyTorch manages their memory.
4. Inputs are made contiguous; `aoti_torch_mps_set_arg_tensor` binds a tensor's buffer at its storage offset.
5. `aoti_torch_mps_run_command_block` encodes each dispatch on PyTorch's MPS stream.
   Its callback records the first error code, which the host raises after the block returns, so that no C++ exception crosses the shim's C frames.

PyTorch's stable-ABI notes promise its C shim, `shim.h`, and the generated `c_shim_*.h` for at least two years, but don't name `shim_mps.h`, which AOTInductor's MPS models depend on.

## Dispatcher and autograd

`ops.cpp` and `torchlpc_shim.cpp` define the schemas.
The MPS registration at the end of `scalar_recur_mps.mm` maps

```text
philtorch::lti_recur -> lti_recur_mps
philtorch::scan      -> scan_mps
```

Both schemas have a custom autograd formula in `philtorch/_ops.py`.
Their backward passes express the reverse recurrence with the same dispatcher operations, so the MPS forward and backward both run these kernels.

## Supported inputs and edge cases

The MPS kernels require:

- `a`, `zi`, and `x` on an MPS device with the same dtype;
- float32, float16, bfloat16 or complex64 values;
- for `lti_recur`, a scalar `a`, or a vector of one value or one per sequence; for `scan`, a `(B, T)` decay;
- one-dimensional `zi` with length `B`, and two-dimensional `x` with shape `(B, T)`;
- `T` below 2^32 - 1024.

`lti_recur` rejects a zero-length sequence, because its autograd formula needs a final output state; `scan` returns an empty result.
An empty batch returns `empty_like(x)` without a Metal dispatch.
Parallel reassociation can change floating-point rounding relative to a strict serial loop, so tests use explicit per-dtype tolerances.

## CPU and MPS benchmark

The following results were measured on an Apple M1 Pro (14-core GPU) with macOS 26.7.1, PyTorch 2.13.0 and six CPU threads, by

```bash
pixi run python benchmarks/benchmark_mps_recur.py
```

Input transfers are excluded.
Runtime uses `torch.utils.benchmark.Timer.blocked_autorange` for at least 500 ms and reports median latency with interquartile range in parentheses; the MPS timer synchronizes PyTorch's MPS stream at each measurement boundary.
CPU peak memory is reconstructed from allocation events captured by `torch.profiler.profile(profile_memory=True)`; MPS peak memory is sampled with `torch.mps.current_allocated_memory()` during repeated calls.
Memory figures exclude inputs and allocator cache but include output and native temporary tensors.
The CPU kernel of `scan` takes no float16 or bfloat16 (`n/a`).

| dtype | B | T | Coefficients | CPU median ms (IQR) | MPS median ms (IQR) | Speedup | CPU peak MiB | MPS peak MiB | Memory ratio |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 1 | 4096 | shared | 0.0677 (0.0068) | 0.0234 (0.0006) | 2.90x | 0.016 | 0.016 | 0.97x |
| float32 | 1 | 262144 | shared | 0.2632 (0.0291) | 0.0375 (0.0038) | 7.02x | 1.000 | 1.003 | 1.00x |
| float32 | 16 | 4096 | shared | 0.0725 (0.0052) | 0.0505 (0.0042) | 1.44x | 0.250 | 0.251 | 1.00x |
| float32 | 64 | 65536 | shared | 1.6243 (0.2290) | 0.2427 (0.0123) | 6.69x | 16.000 | 16.047 | 1.00x |
| float32 | 256 | 4096 | shared | 0.4801 (0.0573) | 0.0959 (0.0141) | 5.00x | 4.000 | 4.012 | 1.00x |
| float32 | 16 | 4096 | batched | 0.0760 (0.0048) | 0.0461 (0.0164) | 1.65x | 0.250 | 0.251 | 1.00x |
| float32 | 64 | 65536 | batched | 1.9368 (0.3711) | 0.2485 (0.0144) | 7.79x | 16.000 | 16.047 | 1.00x |
| float32 | 256 | 4096 | batched | 0.4384 (0.0418) | 0.0968 (0.0473) | 4.53x | 4.000 | 4.012 | 1.00x |
| float32 | 1 | 4096 | per_step | 0.0674 (0.0094) | 0.0232 (0.0007) | 2.90x | 0.016 | 0.016 | 0.97x |
| float32 | 1 | 262144 | per_step | 0.2332 (0.0143) | 0.0432 (0.0109) | 5.40x | 1.000 | 1.003 | 1.00x |
| float32 | 16 | 4096 | per_step | 0.0710 (0.0058) | 0.0269 (0.0012) | 2.64x | 0.250 | 0.251 | 1.00x |
| float32 | 64 | 65536 | per_step | 1.6967 (0.1577) | 0.5039 (0.0120) | 3.37x | 16.000 | 16.047 | 1.00x |
| float32 | 256 | 4096 | per_step | 0.4406 (0.0324) | 0.0853 (0.0015) | 5.16x | 4.000 | 4.012 | 1.00x |
| float16 | 1 | 4096 | shared | 0.0681 (0.0180) | 0.0233 (0.0006) | 2.93x | 0.008 | 0.008 | 0.94x |
| float16 | 1 | 262144 | shared | 0.2395 (0.0072) | 0.0358 (0.0014) | 6.69x | 0.500 | 0.503 | 0.99x |
| float16 | 16 | 4096 | shared | 0.0732 (0.0048) | 0.0266 (0.0007) | 2.75x | 0.125 | 0.126 | 0.99x |
| float16 | 64 | 65536 | shared | 1.3633 (0.0760) | 0.2243 (0.0177) | 6.08x | 8.000 | 8.047 | 0.99x |
| float16 | 256 | 4096 | shared | 0.3940 (0.0540) | 0.1385 (0.0590) | 2.85x | 2.000 | 2.012 | 0.99x |
| float16 | 16 | 4096 | batched | 0.0727 (0.0013) | 0.0264 (0.0008) | 2.76x | 0.125 | 0.126 | 0.99x |
| float16 | 64 | 65536 | batched | 1.5489 (0.3382) | 0.2185 (0.0074) | 7.09x | 8.000 | 8.047 | 0.99x |
| float16 | 256 | 4096 | batched | 0.3869 (0.0212) | 0.0729 (0.0010) | 5.31x | 2.000 | 2.012 | 0.99x |
| float16 | 1 | 4096 | per_step | n/a | 0.0233 (0.0017) | n/a | n/a | 0.008 | n/a |
| float16 | 1 | 262144 | per_step | n/a | 0.0413 (0.0135) | n/a | n/a | 0.503 | n/a |
| float16 | 16 | 4096 | per_step | n/a | 0.0261 (0.0009) | n/a | n/a | 0.126 | n/a |
| float16 | 64 | 65536 | per_step | n/a | 0.2291 (0.0124) | n/a | n/a | 8.047 | n/a |
| float16 | 256 | 4096 | per_step | n/a | 0.0831 (0.0129) | n/a | n/a | 2.012 | n/a |
| bfloat16 | 1 | 4096 | shared | 0.0714 (0.0113) | 0.0239 (0.0013) | 2.99x | 0.008 | 0.008 | 0.94x |
| bfloat16 | 1 | 262144 | shared | 0.2416 (0.0186) | 0.0574 (0.0189) | 4.21x | 0.500 | 0.503 | 0.99x |
| bfloat16 | 16 | 4096 | shared | 0.0725 (0.0044) | 0.0463 (0.0209) | 1.57x | 0.125 | 0.126 | 0.99x |
| bfloat16 | 64 | 65536 | shared | 1.3421 (0.2254) | 0.2366 (0.0031) | 5.67x | 8.000 | 8.047 | 0.99x |
| bfloat16 | 256 | 4096 | shared | 0.3770 (0.0269) | 0.0786 (0.0026) | 4.80x | 2.000 | 2.012 | 0.99x |
| bfloat16 | 16 | 4096 | batched | 0.0720 (0.0034) | 0.0271 (0.0007) | 2.66x | 0.125 | 0.126 | 0.99x |
| bfloat16 | 64 | 65536 | batched | 1.3255 (0.1143) | 0.2438 (0.0113) | 5.44x | 8.000 | 8.047 | 0.99x |
| bfloat16 | 256 | 4096 | batched | 0.4648 (0.1461) | 0.1009 (0.0203) | 4.61x | 2.000 | 2.012 | 0.99x |
| bfloat16 | 1 | 4096 | per_step | n/a | 0.0561 (0.0049) | n/a | n/a | 0.008 | n/a |
| bfloat16 | 1 | 262144 | per_step | n/a | 0.0373 (0.0014) | n/a | n/a | 0.503 | n/a |
| bfloat16 | 16 | 4096 | per_step | n/a | 0.0467 (0.0033) | n/a | n/a | 0.126 | n/a |
| bfloat16 | 64 | 65536 | per_step | n/a | 0.2409 (0.0071) | n/a | n/a | 8.047 | n/a |
| bfloat16 | 256 | 4096 | per_step | n/a | 0.1040 (0.0646) | n/a | n/a | 2.012 | n/a |
| complex64 | 1 | 4096 | shared | 0.0885 (0.0128) | 0.0264 (0.0014) | 3.36x | 0.031 | 0.032 | 0.98x |
| complex64 | 1 | 262144 | shared | 0.6272 (0.0284) | 0.0543 (0.0173) | 11.55x | 2.000 | 2.006 | 1.00x |
| complex64 | 16 | 4096 | shared | 0.1260 (0.0127) | 0.0339 (0.0016) | 3.71x | 0.500 | 0.501 | 1.00x |
| complex64 | 64 | 65536 | shared | 5.8414 (1.8812) | 0.6170 (0.0150) | 9.47x | 32.000 | 32.094 | 1.00x |
| complex64 | 256 | 4096 | shared | 1.2396 (0.0650) | 0.1834 (0.0182) | 6.76x | 8.000 | 8.023 | 1.00x |
| complex64 | 16 | 4096 | batched | 0.1242 (0.0036) | 0.0335 (0.0010) | 3.71x | 0.500 | 0.501 | 1.00x |
| complex64 | 64 | 65536 | batched | 5.3326 (0.8157) | 0.6030 (0.0114) | 8.84x | 32.000 | 32.094 | 1.00x |
| complex64 | 256 | 4096 | batched | 1.5673 (0.2158) | 0.1517 (0.0215) | 10.33x | 8.000 | 8.023 | 1.00x |
| complex64 | 1 | 4096 | per_step | 0.0844 (0.0074) | 0.0443 (0.0165) | 1.91x | 0.031 | 0.032 | 0.98x |
| complex64 | 1 | 262144 | per_step | 0.6468 (0.0235) | 0.0585 (0.0014) | 11.07x | 2.000 | 2.006 | 1.00x |
| complex64 | 16 | 4096 | per_step | 0.1234 (0.0030) | 0.0340 (0.0012) | 3.63x | 0.500 | 0.501 | 1.00x |
| complex64 | 64 | 65536 | per_step | 4.9784 (0.1493) | 0.9920 (0.0146) | 5.02x | 32.000 | 32.094 | 1.00x |
| complex64 | 256 | 4096 | per_step | 1.2541 (0.0686) | 0.1849 (0.0034) | 6.78x | 8.000 | 8.023 | 1.00x |

## Build and verification

On macOS, `setup.py` checks that torch is 2.10 or newer, adds every `.mm` source to the C++ extension and generates its headers in `build/metal`.
It locates Homebrew LLVM and libomp with `brew --prefix`, selects Homebrew Clang unless `CXX` is already set, and uses the Homebrew OpenMP headers.
The extension links `@rpath/libomp.dylib` through PyTorch's library directory so PyTorch and extension modules share one OpenMP runtime.

Install the required system packages before creating or rebuilding the environment:

```bash
brew install llvm libomp
```

Rebuild the editable extension after changing the host code or the Metal source:

```bash
pixi reinstall philtorch
```

Run focused native recurrence tests with:

```bash
pixi run pytest tests/test_recur_ext.py -k mps -q
```

The MPS tests cover every dtype and coefficient layout at lengths around the vector width and the 1024-step tile, sliced inputs, gradients, and the public paths, against CPU.
The long scan test uses `T = 1024 * 512 + 5`, so that stage 2 carries a map across more than one chunk of 512 tile totals.

## Invariants to preserve

When modifying this implementation, keep these details synchronized:

- `compose(left, right)` means `right(left(h))`; do not reverse it.
- The first step folds in `zi` with a zero multiplier, so maps from a sequence's start are constants and `states` holds plain states.
- Steps past the end must be identities, `(1, 0)`.
- `states` holds each tile's end state; replay reads the previous tile's, and the first tile starts from zero.
- The 1024-step tile assumes 256 threads, four steps per thread, and eight 32-lane SIMD groups; `THREADS` and `K` in the Metal source and `THREADS` and `STEPS_PER_THREAD` in the host code must agree.
- The dispatches must stay ordered: tile reduction, tile-total scan, then tile replay.
- The MPS registrations are the public `philtorch::lti_recur` and `philtorch::scan` schemas.
