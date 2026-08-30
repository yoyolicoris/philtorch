# PhilTorch $\Huge \overset{🔥}{\Phi}$

[![PyPI version](https://img.shields.io/pypi/v/philtorch.svg)](https://pypi.org/project/philtorch/)
[![Python versions](https://img.shields.io/pypi/pyversions/philtorch.svg)](https://pypi.org/project/philtorch/)
[![Build CPU wheels](https://github.com/yoyolicoris/philtorch/actions/workflows/build-wheels.yml/badge.svg?branch=dev)](https://github.com/yoyolicoris/philtorch/actions/workflows/build-wheels.yml)
[![codecov](https://codecov.io/gh/yoyolicoris/philtorch/branch/dev/graph/badge.svg?token=288BR3PYIX)](https://codecov.io/gh/yoyolicoris/philtorch)
[![arXiv](https://img.shields.io/badge/arXiv-2511.14390-b31b1b.svg)](https://arxiv.org/abs/2511.14390)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PhilTorch provides differentiable, time-domain linear time-invariant (LTI) and linear parameter-varying (LPV) digital filters and recurrences for PyTorch.

- Differentiate through filter coefficients, inputs, and initial states with PyTorch autograd.
- Process batched signals and filters whose coefficients vary at every time step.
- Use native CPU kernels, the supported macOS MPS scalar-LTI kernel, and CUDA recurrence kernels where the selected shape, dtype, backend, and build support them.
- Compile vendored torchlpc paths exercised by `philtorch.lpv.linear_recurrence` and `philtorch.lpv.allpole`, plus supported CUDA ParaRNN 2×2 and 3×3 state-space paths, with `torch.compile(..., fullgraph=True)` for forward evaluation and ordinary reverse-mode backward.

## Installation

PhilTorch v0.5 requires its compiled `philtorch._C` extension and does not silently fall back when that extension is missing.
Choose a pre-built CPU wheel only for a supported platform and PyTorch minor; otherwise build the extension against the PyTorch installation you intend to use.

| Route | Linux | macOS | Windows | Acceleration | Python | PyTorch compatibility |
| --- | --- | --- | --- | --- | --- | --- |
| PyPI wheel | manylinux_2_28 x86_64 | 14+ arm64 | AMD64 | Native CPU kernels, plus the float32 scalar-LTI MPS kernel on macOS. | 3.10–3.13 | Built and release-tested against the exact latest stable PyTorch patch selected when v0.5 is released. |
| Source build | Toolchain-dependent. | Toolchain-dependent. | Toolchain-dependent. | CPU, supported macOS MPS, or CUDA. | 3.10+ | Builds against the installed PyTorch 2.0 or newer. |

### Pre-built PyPI wheels

```bash
python -m pip install philtorch
```

The v0.5 PyPI wheels contain PhilTorch's native CPU kernels and, on macOS, its float32 scalar-LTI MPS kernel, but they do not contain CUDA kernels.
Because `philtorch._C` uses the PyTorch C++ API and a wheel filename cannot encode a PyTorch release, v0.5 does not claim wheel ABI compatibility across PyTorch minor releases.
Use a source build when your PyTorch minor differs from the one used to build the v0.5 wheel.

### CUDA and other source builds

Install the desired PyTorch build first by following the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then build PhilTorch without build isolation so `setup.py` can inspect that exact installation:

```bash
python -m pip install "setuptools>=77.0.3" "setuptools_scm>=8" wheel
python -m pip install --no-binary=philtorch --no-build-isolation philtorch
```

A source build requires a platform C++ toolchain.
When the installed PyTorch reports OpenMP support, macOS builds additionally require Homebrew LLVM and `libomp` unless `PHILTORCH_DISABLE_OPENMP=1` is set.
`setup.py` selects a CUDA extension only when `torch.cuda.is_available()` is true and `CUDA_HOME` is set; otherwise it builds the C++ extension.
The source distribution includes the vendored [torchlpc](https://github.com/DiffAPF/torchlpc) and [ParaRNN](https://github.com/apple/ml-pararnn) sources used by `philtorch._C`.

For an editable v0.5 build from Git, initialize the pinned submodules before installing:

```bash
git clone --branch v0.5 --recurse-submodules https://github.com/yoyolicoris/philtorch.git
cd philtorch
python -m pip install torch  # Choose the correct CPU or CUDA build first.
python -m pip install "setuptools>=77.0.3" "setuptools_scm>=8" wheel
python -m pip install --editable . --no-build-isolation
```

PhilTorch uses [`setuptools_scm`](https://setuptools-scm.readthedocs.io/) to derive package versions from Git tags and write `philtorch/_version.py` during the build.
The `v0.5` tag produces package version `0.5`, untagged commits use the `guess-next-dev` scheme without a local `+...` component, and builds without usable SCM metadata use the configured `0.4` fallback.
A v0.5 release therefore does not require a manual edit to `philtorch/VERSION.txt`, but direct builds from a checkout should retain the Git tag metadata.

## Quickstart

This example applies a first-order LTI filter on CPU and differentiates a scalar loss with respect to the signal and both coefficient tensors:

```python
import torch

from philtorch.lti import lfilter

device = torch.device("cpu")
dtype = torch.float64

x = torch.linspace(0.0, 1.0, steps=16, device=device, dtype=dtype, requires_grad=True)
b = torch.tensor([0.5], device=device, dtype=dtype, requires_grad=True)
a = torch.tensor([-0.5], device=device, dtype=dtype, requires_grad=True)
zi = torch.zeros(1, device=device, dtype=dtype)

y, zf = lfilter(b, a, x, zi=zi)
loss = y.square().mean()
loss.backward()

print(y.shape)  # torch.Size([16])
print(zf.shape)  # torch.Size([1])
print(x.grad is not None, b.grad is not None, a.grad is not None)  # True True True
```

For `philtorch.lti.lfilter`, the time axis is last, so `x` may have shape `(N,)` or `(B, N)`.
All inputs in this example use the same `torch.float64` dtype and CPU device, and the returned `y` and `zf` stay on that dtype and device.
PhilTorch expects `b = [b0, ..., bM]` and denominator coefficients `a = [a1, ..., aN]` after normalizing the implicit `a0` to one.
The `lfilter` call order is `lfilter(b, a, x, ...)`.
With the default transposed direct-form II path, providing `zi` makes `lfilter` return `(y, zf)`, and both state tensors have length `max(len(b) - 1, len(a))` for an unbatched signal.
The helper for steady-state initial conditions uses `lfilter_zi(a, b)`, so its coefficient argument order differs from `lfilter`.

Move the tensors to a CUDA device only after installing a CUDA source build.

## PyTorch 2 transforms

Full-graph [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) forward and ordinary reverse-mode backward parity is tested for `philtorch.lpv.linear_recurrence` and `philtorch.lpv.allpole` through the vendored torchlpc operators.
The same parity is tested conditionally for CUDA ParaRNN-backed `philtorch.lpv.state_space_recursion` with state dimensions two and three when those kernels are available.
The corresponding differentiable kernels retain eager JVP, `vmap`, `gradcheck`, and `gradgradcheck` support.
PhilTorch v0.5 does not make a blanket compile guarantee for every public API, backend, device, dtype, optional Helion path, dynamic shape, export workflow, or compile-time performance, and it does not claim compiled JVP, `vmap`, or `jacfwd` support.

## Choose the right API

The supported high-level interfaces are exported from [`philtorch.lti`](philtorch/lti/__init__.py) for fixed coefficients and [`philtorch.lpv`](philtorch/lpv/__init__.py) for coefficients that vary over time.

| Intent | API | Notes |
| --- | --- | --- |
| Apply a fixed-coefficient causal IIR or FIR filter. | `philtorch.lti.lfilter` or `philtorch.lti.fir` | `lfilter` supports `df2`, `tdf2`, `df1`, and `tdf1` forms. |
| Apply a fixed-coefficient zero-phase filter. | `philtorch.lti.filtfilt` | Runs the LTI filter forward and backward; edge padding is enabled by default and can be disabled with `padmode=None`. |
| Construct or recover IIR filter state. | `philtorch.lti.lfilter_zi` or `philtorch.lti.lfiltic` | These helpers follow PhilTorch's implicit-`a0` denominator convention. |
| Apply a time-varying filter. | `philtorch.lpv.lfilter`, `philtorch.lpv.fir`, or `philtorch.lpv.allpole` | Coefficient tensors include a time dimension aligned with the input. |
| Evaluate a scalar linear recurrence. | `philtorch.lti.linear_recurrence` or `philtorch.lpv.linear_recurrence` | Choose the namespace according to whether the recurrence coefficient is fixed or time-varying. |
| Evaluate a state-space system. | `philtorch.lti.state_space` or `philtorch.lpv.state_space` | Use `state_space_recursion` directly when only the internal state sequence is needed. |
| Evaluate an LTI state-space system through eigendecomposition. | `philtorch.lti.diag_state_space` | Requires a diagonalisable `A`, or explicitly supplied `L`, `V`, and/or `Vinv`; diagonalisation failures propagate. |
| Apply an LTI comb filter or cubic-spline interpolation. | `philtorch.lti.comb_filter` or `philtorch.lti.cubic_spline` | These utilities are also part of the public LTI exports. |

`philtorch.lti` exports `lfilter`, `lfilter_zi`, `lfiltic`, `filtfilt`, `state_space_recursion`, `diag_state_space`, `state_space`, `fir`, `linear_recurrence`, `comb_filter`, and `cubic_spline`.
`philtorch.lpv` exports `lfilter`, `linear_recurrence`, `state_space`, `state_space_recursion`, `allpole`, and `fir`.
See each function's docstring for its current signature and shape notes.

## Performance and backend selection

Digital filters are recursive, so the best implementation depends on device, dtype, state dimension, sequence length, and whether coefficients vary over time.
The required `philtorch._C` extension registers the native operators, but individual calls may still select a PyTorch recurrence path when a native kernel does not support that configuration.

### Runtime dispatch

- Leave `unroll_factor=1` to use a native recurrence kernel whenever the current state-space path supports the input.
- On CPU, the LTI and LPV state-space paths can use compiled recurrence operators for every state dimension.
- On CUDA, the LTI state-space path uses native recurrence operators for state dimensions one and two.
- On macOS, native MPS support is limited to float32 scalar LTI recurrence; use CPU or a non-default `unroll_factor` for other MPS recurrence configurations.
- On CUDA, the LPV state-space path uses the vendored [torchlpc](https://github.com/DiffAPF/torchlpc) all-pole operator for scalar recurrences and vendored [ParaRNN](https://github.com/apple/ml-pararnn) operators for floating-point state dimensions two and three; dimension two otherwise uses `philtorch::recur2` when that operator supports the input.
- `philtorch.lpv.linear_recurrence` and `philtorch.lpv.allpole` also use the vendored [torchlpc](https://github.com/DiffAPF/torchlpc) operators at `unroll_factor=1`.
- `philtorch.lpv.lfilter` defaults to `backend="ssm"` and `form="tdf2"`; `backend="torchlpc"` selects the vendored [torchlpc](https://github.com/DiffAPF/torchlpc) denominator-recursion path, defaults to `form="df2"`, and does not implement `form="tdf2"`.
- Any `unroll_factor != 1` bypasses direct native-kernel dispatch; values satisfying `1 < unroll_factor < sequence_length` use the block-unrolled formulation, while values at least as large as the sequence use the direct PyTorch recurrence loop.
- Configurations that the dispatcher does not select for native execution use the PyTorch recurrence path, but a selected native path can still raise when the installed extension lacks the requested device or dtype implementation.

Prefer cascades or parallel banks of first-order and second-order sections when the application permits that factorization, especially for CUDA workloads that should stay on native low-order kernels.
Treat `unroll_factor=8` on CPU and `unroll_factor=16` or `32` on CUDA as starting points rather than universal defaults, and benchmark the actual batch size, sequence length, state dimension, dtype, and device.
Keep `unroll_factor` smaller than the sequence length when evaluating block unrolling.

For background on the block formulation, see [Unrolling State Space Models](https://iamycy.github.io/posts/2025/06/28/unroll-ssm/).

### Reproducible benchmark template

The following command records the PhilTorch and PyTorch versions, device, dtype, shape, filter order, warm-up count, and `torch.utils.benchmark` measurement without publishing a machine-independent headline number:

```bash
python - <<'PY'
import platform

import philtorch
import torch
from philtorch.lti import lfilter
from torch.utils.benchmark import Timer

device = torch.device("cpu")  # Use "cuda" only with a CUDA-enabled PhilTorch source build.
dtype = torch.float32
batch_size = 32
num_samples = 65536
warmup = 10

x = torch.randn(batch_size, num_samples, device=device, dtype=dtype)
b = torch.tensor([0.5], device=device, dtype=dtype).repeat(batch_size, 1)
a = torch.tensor([-0.5], device=device, dtype=dtype).repeat(batch_size, 1)

def run():
    return lfilter(b, a, x)

for _ in range(warmup):
    run()

if device.type == "cuda":
    torch.cuda.synchronize()

measurement = Timer(stmt="run()", globals={"run": run}).blocked_autorange(min_run_time=1.0)
device_name = torch.cuda.get_device_name() if device.type == "cuda" else platform.platform()
print({"philtorch": philtorch.__version__, "torch": torch.__version__, "device": device_name, "dtype": str(dtype), "batch_size": batch_size, "num_samples": num_samples, "filter_order": a.shape[-1], "warmup": warmup})
print(measurement)
PY
```

Report the command, environment dictionary, and complete measurement together when comparing backends or hardware.

## Examples and project links

- The [low-pass estimation notebook](examples/estimate_lowpass.ipynb) demonstrates learning filter parameters.
- The [contribution guide](CONTRIBUTING.md) documents the development and pull-request workflow.
- The [issue tracker](https://github.com/yoyolicoris/philtorch/issues) is the place for bug reports and feature requests.
- PhilTorch is distributed under the [MIT License](LICENSE), with third-party notices in [`LICENSES`](LICENSES/README.md).

## Paper and citation

PhilTorch's LTI direct-form filtering work is described in [Accelerating Automatic Differentiation of Direct Form Digital Filters](https://arxiv.org/abs/2511.14390) by Chin-Yun Yu and György Fazekas.

```bibtex
@article{yu2025accelerating,
  title={Accelerating Automatic Differentiation of Direct Form Digital Filters},
  author={Yu, Chin-Yun and Fazekas, György},
  journal={arXiv preprint arXiv:2511.14390},
  year={2025}
}
```

<details>
<summary>News</summary>

- **2025-12-06:** The paper was presented at the [Differentiable Systems and Scientific Machine Learning Workshop](https://differentiable-systems.github.io/workshop-eurips-2025/) at EurIPS 2025, with the [poster available here](https://github.com/yoyolicoris/presentations/blob/main/posters/2025/DiffSys_Eurips.pdf).
- **2025-11-10:** PhilTorch was presented at the [Audio Developer Conference 2025](https://conference.audio.dev/session/2025/philtorch/), with the [slides available here](https://github.com/yoyolicoris/presentations/blob/main/slides/2025/adc25.pdf).
- **2025-10-31:** The PhilTorch paper was accepted by the Differentiable Systems and Scientific Machine Learning Workshop at EurIPS 2025.

</details>
