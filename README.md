# PhilTorch $\Huge \overset{🔥}{\Phi}$
[![PyPI version](https://badge.fury.io/py/philtorch.svg)](https://badge.fury.io/py/philtorch)
[![codecov](https://codecov.io/gh/yoyolicoris/philtorch/branch/dev/graph/badge.svg?token=288BR3PYIX)](https://codecov.io/gh/yoyolicoris/philtorch)
[![arXiv](https://img.shields.io/badge/arXiv-2511.14390-b31b1b.svg)](https://arxiv.org/abs/2511.14390)

A PyTorch package for fast automatic differentiation of discrete time linear filters.

Our principle design goals are:

- Provide fast and differentiable version of `scipy.signal.*` functions.
- Focus on time-domain implementation without using FFT.
- Support batch processing, parameter-varying filters, and GPU acceleration.
- Pure functional implementation and no stateful objects.

## News

- **2025-12-06**: We presented our paper, [Accelerating Automatic Differentiation of Direct Form Digital Filters](https://openreview.net/forum?id=ZhwIyvtBNB), at the [Differentiable Systems and Scientific Machine Learning Workshop](https://differentiable-systems.github.io/workshop-eurips-2025/) @ EurIPS 2025! You can check the poster [here](https://github.com/yoyolicoris/presentations/blob/main/posters/2025/DiffSys_Eurips.pdf).
- **2025-11-10**: PhilTorch was first presented at the [Audio Developer Conference (ADC) 2025](https://conference.audio.dev/session/2025/philtorch/)! The presentation slides are available [here](https://github.com/yoyolicoris/presentations/blob/main/slides/2025/adc25.pdf).
- **2025-10-31**: Our short paper describing the LTI filter implementation in PhilTorch has been accepted by the [DiffSys Workshop @ EurIPSs 2025](https://differentiable-systems.github.io/workshop-eurips-2025/)! The preprint is available [here](https://arxiv.org/abs/2511.14390).

## Installation

### Stable release
```bash
pip install philtorch
```

### Development version
```bash
pip install -i https://test.pypi.org/simple/ philtorch
# or
pip install git+https://github.com/yoyolicoris/philtorch.git
``` 
> **_Note:_**
> - The installation process compiles C++/CUDA extensions, so make sure you have a working C++ compiler and CUDA toolkit (if you want to use GPU acceleration) installed.
> - We recommend using `--no-build-isolation` flag to avoid potential issues with building the package in an isolated environment, especially when installing with CUDA support.

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
        - `state_space_recursion`: 
            - The core recursion function for state-space models.
        - `linear_recurrence`: 
            - A linear recurrence function with scalar coefficients.
    - `utils`: Utility functions.
    - `mat`: Matrix operations.
    - `poly`: Polynomial operations.

For detailed API reference, please refer to the docstring of each function.

## Performance Guide

Digital filters like IIR filters are recursively defined and thus hard to parallelise in PyTorch.
PhilTorch implements custom C++/CUDA extensions to achieve high performance.
Currently, we have full support for first and second order filters, and we plan to have fast kernel for higher order filters in the future.
Thus, we recommend composing first and second order sections (SOS), either cascaded or parallel form, if possible.

In the worst case when the extension is not compiled, we also provide a fallback implementation using PyTorch operations.
This implementation computes parallel associative scans using just matrix multiplications.
It divides the input sequence into blocks recursively, and computes the output for each block in parallel.
For more details, please refer to the this [blog post](https://iamycy.github.io/posts/2025/06/28/unroll-ssm/).

The size of the blocks can greatly affect the performance.
To control the block size, we provide a `unroll_factor` argument in most of the filter functions.
By default, it is set to 1, which means no unrolling.
The optimal value depends on the filter order, input length, and hardware.
In general, we recommend setting it to 8 when the Tensors are on CPU, and 16 to 32 when they are on GPU.
Though it's at least 10 times slower than the custom extension, the fallback implementation is still much faster than naive for-loop.


## Examples

### Recreating `scipy.signal.lfilter` example

```python
import torch
from philtorch.lti import lfilter, lfilter_zi, filtfilt
from scipy.signal import butter

x = torch.randn(201)

b_np, a_np = butter(3, 0.05)
# note that in philtorch a_0 is always 1
b_np /= a_np[0]
a_np = a_np[1:] / a_np[0]
b, a = torch.from_numpy(b_np), torch.from_numpy(a_np)

# note that the position of a and b are swapped compared to scipy
zi = lfilter_zi(a, b)

z, _ = lfilter(b, a, x, zi=zi * x[0])
z2 = filtfilt(b, a, x)
```

If `lfilter` is imported from `philtorch.lpv`, it can also handle parameter-varying filters, where `a` and `b` are at least 2D tensors with an additional time dimension.

### Computing the first 10 Fibonacci numbers using `state_space`

The function `philtorch.lti.state_space` compute the following recursion:

```math
\begin{aligned}
\mathbf{h}_{n+1} &= \mathbf{A} \mathbf{h}_n + \mathbf{B} \mathbf{x}_n \\
\mathbf{y}_n &= \mathbf{C} \mathbf{h}_n + \mathbf{D} \mathbf{x}_n
\end{aligned}
```

We can use it to compute the Fibonacci numbers by setting:

```math
\begin{aligned}
\mathbf{A} = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}, \quad
\mathbf{C} = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad
\mathbf{B} = \mathbf{D} = 0, \\
\mathbf{h}_0 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
\end{aligned}
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
```
tensor([[ 1,  1,  2,  3,  5,  8, 13, 21, 34, 55]])
```
The result is the first 10 Fibonacci numbers, which has the following recursion relation:

```math
F_n = F_{n-1} + F_{n-2}, \quad F_0 = 1, \quad F_1 = 1
```
