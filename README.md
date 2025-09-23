# PhilTorch

A PyTorch package for fast automatic differentiation of discrete time linear filters.

Our priorities are:

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

## Module overview

- `philtorch`: Root module.
    - `lpv`: Functions under it are for linear parameter-varying filters.
        - `fir`: Finite Impulse Response filters.
        - `allpole`: All-pole filters.
        - `lfilter`: Parameter-varying version of `scipy.signal.lfilter`. It supports not only transposed direct form II but also transposed direct form I, direct form I, and direct form II structures.
        - `state_space`: Parameter-varying state-space models.
        - `state_space_recursion`: The core recursion function for state-space models.
        - `linear_recurrence`: A linear recurrence function with scalar coefficients.
    - `lti`: Functions under it are for linear time-invariant filters.
        - `fir`: Finite Impulse Response filters.
        - `lfilter`: A differentiable version of `scipy.signal.lfilter`. It supports not only transposed direct form II but also transposed direct form I, direct form I, and direct form II structures.
        - `filtfilt`: A differentiable version of `scipy.signal.filtfilt`.
        - `lfilter_zi`: A differentiable version of `scipy.signal.lfilter_zi`.
        - `lfiltic`: A differentiable version of `scipy.signal.lfiltic`.
        - `state_space`: State-space models.
        - `diag_state_space`: State-space models with diagonalisable state matrix.
        - `state_space_recursion`: The core recursion function for state-space models.
        - `linear_recurrence`: A linear recurrence function with scalar coefficients.
    - `utils`: Utility functions.
    - `mat`: Matrix operations.
    - `poly`: Polynomial operations.