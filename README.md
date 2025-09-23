# PhilTorch

A PyTorch package for fast automatic differentiation of discrete time linear filters.

Our principle design goals are:

- Provide fast and differentiable version of `scipy.signal.*` functions.
- Focus on time-domain implementation without using FFT.
- Support batch processing, parameter-varying filters, and GPU acceleration.
- Pure functional implementation and no stateful objects.

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
