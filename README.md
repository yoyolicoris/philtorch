# PhilTorch $\Huge \overset{🔥}{\Phi}$

[![PyPI version](https://img.shields.io/pypi/v/philtorch.svg)](https://pypi.org/project/philtorch/)
[![Python versions](https://img.shields.io/pypi/pyversions/philtorch.svg)](https://pypi.org/project/philtorch/)
[![Build CPU wheels](https://github.com/yoyolicoris/philtorch/actions/workflows/build-wheels.yml/badge.svg?branch=dev)](https://github.com/yoyolicoris/philtorch/actions/workflows/build-wheels.yml)
[![codecov](https://codecov.io/gh/yoyolicoris/philtorch/branch/dev/graph/badge.svg?token=288BR3PYIX)](https://codecov.io/gh/yoyolicoris/philtorch)
[![OpenReview](https://img.shields.io/badge/OpenReview-ZhwIyvtBNB-8c1b13.svg)](https://openreview.net/forum?id=ZhwIyvtBNB)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PhilTorch provides differentiable, time-domain linear digital filters for PyTorch.

- Differentiate through filter coefficients, inputs, and initial states with PyTorch autograd.
- Process batched signals and filters whose coefficients vary at every time step.
- Compute in the time domain without FFTs, through a pure functional API with no stateful objects.
- Run on native CPU, CUDA, and MPS kernels.

## News

- **2025-12-06:** We presented our paper, [Accelerating Automatic Differentiation of Direct Form Digital Filters](https://openreview.net/forum?id=ZhwIyvtBNB), at the [Differentiable Systems and Scientific Machine Learning Workshop](https://differentiable-systems.github.io/workshop-eurips-2025/) at EurIPS 2025.
  The [poster is available here](https://github.com/yoyolicoris/presentations/blob/main/posters/2025/DiffSys_Eurips.pdf).
- **2025-11-10:** PhilTorch was first presented at the [Audio Developer Conference 2025](https://conference.audio.dev/session/2025/philtorch/).
  The [presentation slides are available here](https://github.com/yoyolicoris/presentations/blob/main/slides/2025/adc25.pdf).
- **2025-10-31:** Our short paper describing the LTI filter implementation in PhilTorch was accepted by the [Differentiable Systems and Scientific Machine Learning Workshop at EurIPS 2025](https://differentiable-systems.github.io/workshop-eurips-2025/).
  The [preprint is available here](https://arxiv.org/abs/2511.14390).

<!-- docs-install-start -->
## Installation

PhilTorch requires its compiled `philtorch._C` extension; there is no pure-PyTorch fallback when the extension is missing.

| Route | Platforms | Kernels | Python | PyTorch |
| --- | --- | --- | --- | --- |
| PyPI wheel | Linux x86_64 (manylinux_2_28), macOS 14+ arm64, Windows AMD64 | CPU, plus float32 scalar-LTI MPS on macOS | 3.10–3.13 | The latest stable minor at publication, which each wheel requires (e.g. `torch == 2.14.*`); listed in the [release notes](https://github.com/yoyolicoris/philtorch/releases) |
| CUDA wheel, from the [PhilTorch index](https://yoyolicoris.github.io/philtorch/) | Linux x86_64 (manylinux_2_28) | CPU and CUDA | 3.10–3.13 | One build per PyTorch minor and CUDA version; see [CUDA wheels](#cuda-wheels) |
| Source build | Any platform with a C++ toolchain | CPU, MPS, or CUDA | 3.10+ | 2.4 or newer |

### PyPI wheels

```bash
python -m pip install philtorch
```

Wheels contain no CUDA kernels.
Each wheel is built against one PyTorch minor release and requires it (e.g. `torch == 2.14.*`), because the compiled extension does not load with other minor releases.
pip therefore installs that PyTorch minor with PhilTorch, replacing a different one if needed.
If a different minor is installed anyway, for example with `--no-deps`, `import philtorch` raises an `ImportError` naming the minor it needs.
For CUDA on Linux, use the [CUDA wheels](#cuda-wheels); for another PyTorch minor release, build from source.

### CUDA wheels

Linux wheels with CUDA kernels are published on a separate package index rather than PyPI.
Wheel tags don't record which PyTorch and CUDA build a wheel is for, and PyPI rejects the local version labels that do (e.g. `+torch2.14.1.cu132`).
There is one index per PyTorch minor and CUDA version that PyTorch publishes wheels for.
Install PyTorch first, then install PhilTorch from the index that matches it:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132
python -m pip install philtorch --index-url https://yoyolicoris.github.io/philtorch/whl/torch2.14-cu132/
```

To find the matching index for an installed PyTorch, run `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
For example, `2.14.1+cu132 13.2` matches `whl/torch2.14-cu132/`.
On Linux, `pip install torch` from PyPI installs the CUDA 13.0 build, which matches `whl/torch2.14-cu130/`.

| PyTorch | CUDA 12.9 | CUDA 13.0 | CUDA 13.2 |
| --- | --- | --- | --- |
| 2.14 | — | [`whl/torch2.14-cu130/`](https://yoyolicoris.github.io/philtorch/whl/torch2.14-cu130/) | [`whl/torch2.14-cu132/`](https://yoyolicoris.github.io/philtorch/whl/torch2.14-cu132/) |
| 2.13 | [`whl/torch2.13-cu129/`](https://yoyolicoris.github.io/philtorch/whl/torch2.13-cu129/) | [`whl/torch2.13-cu130/`](https://yoyolicoris.github.io/philtorch/whl/torch2.13-cu130/) | [`whl/torch2.13-cu132/`](https://yoyolicoris.github.io/philtorch/whl/torch2.13-cu132/) |
| 2.12 | [`whl/torch2.12-cu129/`](https://yoyolicoris.github.io/philtorch/whl/torch2.12-cu129/) | [`whl/torch2.12-cu130/`](https://yoyolicoris.github.io/philtorch/whl/torch2.12-cu130/) | [`whl/torch2.12-cu132/`](https://yoyolicoris.github.io/philtorch/whl/torch2.12-cu132/) |

The index URLs are relative to `https://yoyolicoris.github.io/philtorch/`, which lists every index, including those for earlier releases.

- **Requirements:** Python 3.10–3.13, an NVIDIA GPU with compute capability 7.5, 8.x, 9.0, 10.x or 12.x, and a driver that supports the wheel's CUDA version.
- **PyTorch:** each wheel requires its PyTorch minor (e.g. `torch == 2.14.*`), so pip keeps the PyTorch you installed.
- **Mismatches:** `import philtorch` raises an `ImportError` if the installed PyTorch is a different minor release, has no CUDA support, or uses a different CUDA major version. It only warns if the CUDA minor version differs.
- **Use `--index-url`, not `--extra-index-url`:** with PyPI also searched, pip installs PyPI's CPU-only wheel whenever it is a newer release than the CUDA build in the index, without warning.

### Source builds

Install PyTorch first with the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then build PhilTorch against it without build isolation:

```bash
python -m pip install "setuptools>=77.0.3" "setuptools_scm>=8" wheel
python -m pip install --no-binary=philtorch --no-build-isolation philtorch
```

The build compiles a CUDA extension when `torch.cuda.is_available()` is true and `CUDA_HOME` is set, and a CPU-only C++ extension otherwise.
On a headless build server with no visible GPU, set `PHILTORCH_FORCE_CUDA=1` to compile the CUDA extension anyway.
In that case the installed PyTorch must be a CUDA build, and `TORCH_CUDA_ARCH_LIST` must also be set to the target architectures (e.g. `TORCH_CUDA_ARCH_LIST="8.0 8.6"`), because there is no GPU to infer them from.
On macOS, when the installed PyTorch supports OpenMP, the build also needs Homebrew `llvm` and `libomp`, unless `PHILTORCH_DISABLE_OPENMP=1` is set.

For an editable install from Git:

```bash
git clone --branch main https://github.com/yoyolicoris/philtorch.git
cd philtorch
python -m pip install torch  # Choose the correct CPU or CUDA build first.
python -m pip install "setuptools>=77.0.3" "setuptools_scm>=8" wheel
python -m pip install --editable . --no-build-isolation
```

Package versions come from Git tags through [`setuptools_scm`](https://setuptools-scm.readthedocs.io/), so keep the tag metadata when building from a checkout.

### Development version

Development builds from `dev` are published to TestPyPI:

```bash
python -m pip install -i https://test.pypi.org/simple/ philtorch
```

<!-- docs-install-end -->

## Quickstart

### Filtering with SciPy-designed coefficients

```python
import torch
from scipy.signal import butter

from philtorch.lti import filtfilt, lfilter, lfilter_zi

x = torch.randn(201, dtype=torch.float64)

b_np, a_np = butter(3, 0.05)
# Normalize so that a0 = 1, then drop a0 from the denominator.
b = torch.from_numpy(b_np / a_np[0])
a = torch.from_numpy(a_np[1:] / a_np[0])

# lfilter_zi takes (a, b), the reverse of SciPy's lfilter_zi(b, a).
zi = lfilter_zi(a, b)

y, _ = lfilter(b, a, x, zi=zi * x[0])
y_zero_phase = filtfilt(b, a, x)
```

## Module overview

- `philtorch`: Root module.
    - `lpv`: Functions under it are for linear parameter-varying filters.
        - `fir`:
            - Finite Impulse Response filters.
        - `allpole`:
            - All-pole filters.
        - `lfilter`:
            - Parameter-varying version of `scipy.signal.lfilter`. It supports not only transposed direct form II but also transposed direct form I, direct form I, and direct form II structures.
        - `state_space`:
            - Parameter-varying state-space models.
        - `state_space_recursion`:
            - The core recursion function for state-space models.
        - `linear_recurrence`:
            - A linear recurrence function with scalar coefficients.
    - `lti`: Functions under it are for linear time-invariant filters.
        - `fir`:
            - Finite Impulse Response filters.
        - `lfilter`:
            - A differentiable version of `scipy.signal.lfilter`. It supports not only transposed direct form II but also transposed direct form I, direct form I, and direct form II structures.
        - `filtfilt`:
            - A differentiable version of `scipy.signal.filtfilt`.
        - `lfilter_zi`:
            - A differentiable version of `scipy.signal.lfilter_zi`.
        - `lfiltic`:
            - A differentiable version of `scipy.signal.lfiltic`.
        - `state_space`:
            - State-space models.
        - `diag_state_space`:
            - State-space models with diagonalisable state matrix.
        - `delay_state_space`:
            - State-space models whose states are delay lines of arbitrary integer length, such as feedback delay networks.
        - `state_space_recursion`:
            - The core recursion function for state-space models.
        - `linear_recurrence`:
            - A linear recurrence function with scalar coefficients.
        - `comb_filter`:
            - Delayed all-pole comb filters.
        - `cubic_spline`:
            - Cubic-spline interpolation for integer upsampling.
    - `utils`: Utility functions.
    - `mat`: Matrix operations.
    - `poly`: Polynomial operations.

For detailed API reference, please refer to the docstring of each function.

## Choose the right API

The supported high-level interfaces are exported from [`philtorch.lti`](philtorch/lti/__init__.py) for fixed coefficients and [`philtorch.lpv`](philtorch/lpv/__init__.py) for coefficients that vary over time.

| Intent | API | Notes |
| --- | --- | --- |
| Apply a fixed-coefficient causal IIR or FIR filter. | `philtorch.lti.lfilter` or `philtorch.lti.fir` | `lfilter` supports `df2`, `tdf2`, `df1`, and `tdf1` forms. |
| Apply a fixed-coefficient zero-phase filter. | `philtorch.lti.filtfilt` | Runs the LTI filter forward and backward; edge padding is enabled by default and can be disabled with `padmode=None`. |
| Construct or recover IIR filter state. | `philtorch.lti.lfilter_zi` or `philtorch.lti.lfiltic` | These helpers use the same normalized `a0 = 1` coefficients as `lfilter`. |
| Apply a time-varying filter. | `philtorch.lpv.lfilter`, `philtorch.lpv.fir`, or `philtorch.lpv.allpole` | Coefficient tensors include a time dimension aligned with the input. `lpv.lfilter` defaults to `backend="ssm"`; `backend="torchlpc"` does not support `form="tdf2"`. |
| Evaluate a scalar linear recurrence. | `philtorch.lti.linear_recurrence` or `philtorch.lpv.linear_recurrence` | Choose the namespace according to whether the recurrence coefficient is fixed or time-varying. |
| Evaluate a state-space system. | `philtorch.lti.state_space` or `philtorch.lpv.state_space` | Use `state_space_recursion` directly when only the internal state sequence is needed. |
| Evaluate an LTI state-space system through eigendecomposition. | `philtorch.lti.diag_state_space` | Requires a diagonalisable `A`, or explicitly supplied `L`, `V`, and/or `Vinv`; diagonalisation failures propagate. |
| Evaluate an LTI system whose states are delay lines, such as a feedback delay network. | `philtorch.lti.delay_state_space` | `delays` are positive integers, one per line; `B`, `C`, `D`, and `out_idx` follow `state_space`. `zi` holds one initial queue per line, and `(y, zf)` is returned only when it is given. |
| Apply an LTI comb filter or cubic-spline interpolation. | `philtorch.lti.comb_filter` or `philtorch.lti.cubic_spline` | These utilities are also part of the public LTI exports. |

## Comparison with `scipy.signal`

PhilTorch functions take PyTorch tensors, support autograd and batching, and keep the input dtype and device.
Coefficients are normalized so that `a0 = 1`: divide SciPy's `b` and `a` by `a[0]`, then pass `a[1:]` as the denominator, as in the [Quickstart](#quickstart) example.

| SciPy | PhilTorch | Differences |
| --- | --- | --- |
| `lfilter` | `philtorch.lti.lfilter`, `philtorch.lti.fir` | Filters the last axis of 1-D or batched 2-D tensors, batches coefficients, and supports `df2`, `tdf2`, `df1`, and `tdf1`. `fir` is the batched FIR-only path. |
| `lfiltic` | `philtorch.lti.lfiltic` | Builds the filter state from past inputs and outputs. |
| `lfilter_zi` | `philtorch.lti.lfilter_zi` | Takes `(a, b)`, the reverse of SciPy's `(b, a)`. |
| `filtfilt` | `philtorch.lti.filtfilt` | Defaults to `padmode="replicate"` and disables padding with `padmode=None`; `method="gust"` is not implemented and `irlen` is unused. |
| `cspline1d`, `cspline1d_eval` | `philtorch.lti.cubic_spline` | Upsamples batched 2-D signals by an integer factor only; no arbitrary evaluation points or smoothing (`lamb` must be zero); uses PyTorch reflect padding unless `scipy_padding=True`. |
| `dlsim` | `philtorch.lti.state_space` | Takes `A`, `B`, `C`, `D` tensors instead of a system object and returns the output, plus the final state when `zi` is given; no time vector or full state trajectory. `dimpulse` and `dstep` can be reproduced with explicit impulse or step inputs. |
| — | `philtorch.lpv.lfilter`, `philtorch.lpv.fir`, `philtorch.lpv.allpole`, `philtorch.lpv.state_space` | Time-varying filters and state-space models whose coefficients carry a time dimension. |
| — | `philtorch.lti.linear_recurrence`, `philtorch.lpv.linear_recurrence`, `state_space_recursion` | Differentiable scalar recurrences and internal state sequences. |
| — | `philtorch.lti.diag_state_space` | State-space evaluation through the eigendecomposition of a diagonalizable `A`. |
| — | `philtorch.lti.comb_filter` | Delayed all-pole comb recurrence with a given coefficient and delay; it does not design filters like `iircomb`. |


## Performance

Recursive filters are hard to parallelize, so PhilTorch implements custom C++ and CUDA kernels.
With the default `unroll_factor=1`, the state-space paths (`lfilter`, `state_space`, and `state_space_recursion`) choose a kernel from the device and the filter order, which is the state size `M`:

| Device | LTI, M = 1 | LTI, M = 2 | LTI, M ≥ 3 | LPV, M = 1 | LPV, M = 2 | LPV, M = 3 | LPV, M ≥ 4 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CUDA (source builds) | ✓ | ✓ | loop | ✓ | ✓ | ✓¹ | loop |
| MPS | ✓² | ✗ | loop | ✗ | ✗ | loop | loop |

✓ native kernel · loop: PyTorch recurrence loop · ✗ raises an error because no kernel exists for the device; use CPU or `unroll_factor > 1` instead.
¹ Real dtypes only; complex inputs use the loop. ² float32 only.

Setting `1 < unroll_factor < sequence length` switches to a block-unrolled PyTorch recursion, which computes blocks in parallel with matrix multiplications ([background](https://iamycy.github.io/posts/2025/06/28/unroll-ssm/)); larger values fall back to a plain PyTorch loop.
It is slower than the native kernels but much faster than a naive loop.
Good starting points are 8 on CPU and 16–32 on CUDA, but benchmark your own batch size, sequence length, state size, dtype, and device, for example with `torch.utils.benchmark`.
Native kernels are fastest for first- and second-order filters, so prefer cascades or parallel banks of such sections, especially on CUDA.

## Examples

### Fibonacci numbers with `state_space`

`philtorch.lti.state_space` computes

```math
\begin{aligned}
\mathbf{h}_{n+1} &= \mathbf{A} \mathbf{h}_n + \mathbf{B} \mathbf{x}_n, \\
\mathbf{y}_n &= \mathbf{C} \mathbf{h}_n + \mathbf{D} \mathbf{x}_n.
\end{aligned}
```

Setting the following, with $\mathbf{B} = \mathbf{D} = 0$ and no input, yields the Fibonacci numbers:

```math
\mathbf{A} = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}, \qquad
\mathbf{C} = \begin{bmatrix} 1 & 0 \end{bmatrix}, \qquad
\mathbf{h}_0 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}.
```

```python
import torch

from philtorch.lti import state_space

A = torch.tensor([[1, 1], [1, 0]])
C = torch.tensor([1, 0])
x = torch.zeros(1, 10).long()
h0 = torch.tensor([1, 0])
y, _ = state_space(A, x, C=C, zi=h0)
print(y)
```

```text
tensor([[ 1,  1,  2,  3,  5,  8, 13, 21, 34, 55]])
```

This produces the first ten Fibonacci numbers, where $F_n = F_{n-1} + F_{n-2}$ with $F_0 = F_1 = 1$.

### Learning filter parameters

The [low-pass estimation notebook](examples/estimate_lowpass.ipynb) demonstrates learning filter parameters by gradient descent.

## Project links

- The [contribution guide](CONTRIBUTING.md) documents the development and pull-request workflow, and the [PhilTorch Roadmap](https://github.com/users/yoyolicoris/projects/5) lists current priorities.
- The [issue tracker](https://github.com/yoyolicoris/philtorch/issues) is the place for bug reports and feature requests.
- PhilTorch is distributed under the [MIT License](LICENSE); third-party notices are in [`LICENSES`](LICENSES/README.md).

## Paper and citation

PhilTorch's LTI direct-form filtering work is described in [Accelerating Automatic Differentiation of Direct Form Digital Filters](https://openreview.net/forum?id=ZhwIyvtBNB) by Chin-Yun Yu and György Fazekas.

```bibtex
@inproceedings{yu2025accelerating,
  title={Accelerating Automatic Differentiation of Direct Form Digital Filters},
  author={Yu, Chin-Yun and Fazekas, György},
  booktitle={Differentiable Systems and Scientific Machine Learning Workshop at EurIPS 2025},
  year={2025},
  url={https://openreview.net/forum?id=ZhwIyvtBNB}
}
```
