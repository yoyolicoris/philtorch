# third_party

Third-party CUDA kernels compiled into `philtorch._C`. They used to be a git
submodule. Since upstream no longer takes changes, the files PhilTorch compiles
are now kept here, and PhilTorch maintains them. They keep their upstream
license, which `LICENSES/` mirrors so that wheels carry it too.

## torchlpc

From <https://github.com/DiffAPF/torchlpc> at `1bfde4a` (`v0.7.2-18-g1bfde4a`).
MIT, Copyright (c) 2023 Chin-Yun Yu (`torchlpc/LICENSE`).

| File | Upstream path |
| --- | --- |
| `torchlpc/cuda/lpc.cu` | `torchlpc/csrc/cuda/lpc.cu` |

`philtorch/csrc/torchlpc_cuda.cu` compiles `lpc.cu` and registers its kernel
as `philtorch::lpc`, so it doesn't conflict with an installed torchlpc. The
CPU kernel of `lpc` is in `philtorch/csrc/torchlpc_shim.cpp`, and the autograd
wrappers in `philtorch/_torchlpc.py`. The kernels of `philtorch::scan` are
PhilTorch's own: `philtorch/csrc/host_recur.cpp` (CPU) and
`philtorch/csrc/scalar_recur.cu` (CUDA).

## Local changes

Every kernel runs on PyTorch's current stream. Upstream launched them on the
legacy default stream, which PyTorch's streams don't wait for, so on any other
stream they could read inputs before they were written and return wrong
results.

- `torchlpc/cuda/lpc.cu`:
  - launches on the current stream and checks each launch;
  - checks the order limit with `STD_TORCH_CHECK` instead of `assert`;
  - requires the coefficients, and `zi`, on the input's device;
  - uses only torch's stable C++ API (`torch/csrc/stable`, `torch/headeronly`);
  - reads `zi` and `x` and writes the output directly, instead of filtering a
    padded copy;
  - has one kernel templated on `c10::complex`, in place of the kernels on split
    real and imaginary parts;
  - checks the dtypes and shapes of its inputs as the CPU op does, with the
    shared `check_lpc_inputs` (`philtorch/csrc/recur_checks.h`).
