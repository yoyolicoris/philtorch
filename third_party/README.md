# third_party

Third-party CUDA kernels compiled into `philtorch._C`. They used to be git
submodules. Since neither upstream takes changes any more, the files PhilTorch
compiles are now kept here, and PhilTorch maintains them. Each keeps its
upstream license, which `LICENSES/` mirrors so that wheels carry it too.

## torchlpc

From <https://github.com/DiffAPF/torchlpc> at `1bfde4a` (`v0.7.2-18-g1bfde4a`).
MIT, Copyright (c) 2023 Chin-Yun Yu (`torchlpc/LICENSE`).

| File | Upstream path |
| --- | --- |
| `torchlpc/cuda/lpc.cu` | `torchlpc/csrc/cuda/lpc.cu` |
| `torchlpc/cuda/linear_recurrence.cu` | `torchlpc/csrc/cuda/linear_recurrence.cu` |
| `torchlpc/cuda/LICENSE.txt` | `torchlpc/csrc/cuda/LICENSE.txt` |

`linear_recurrence.cu` comes from Eric Martin's linear recurrence kernels,
MIT, Copyright (c) 2017 Eric Martin (`torchlpc/cuda/LICENSE.txt`).

`philtorch/csrc/torchlpc_cuda.cu` compiles both files and registers the
kernels as `philtorch::lpc` and `philtorch::scan`, so they don't conflict with
an installed torchlpc. The CPU kernels are in `philtorch/csrc/torchlpc_shim.cpp`
and the autograd wrappers in `philtorch/_torchlpc.py`.

## pararnn

From <https://github.com/apple/ml-pararnn> at `513d75d`.
Copyright (C) 2025 Apple Inc. (`pararnn/LICENSE`).

| File | Upstream path |
| --- | --- |
| `pararnn/csrc/parallel_reduce.cu` | `pararnn/csrc/parallel_reduce.cu` |
| `pararnn/csrc/parallel_reduction_kernel.h` | `pararnn/csrc/parallel_reduction_kernel.h` |
| `pararnn/csrc/rnn_cell_impl.h` | `pararnn/csrc/rnn_cell_impl.h` |
| `pararnn/csrc/helpers.h` | `pararnn/csrc/helpers.h` |
| `pararnn/csrc/nonlinearities.h` | `pararnn/csrc/nonlinearities.h` |

`setup.py` compiles `parallel_reduce.cu` for CUDA builds, with
`-DFLOAT64_CHUNK_SIZE_DIAG=4 -DFLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2=1`, and
`philtorch/csrc/pararnn_shim.cpp` registers the block-diagonal kernels as
`parallel_reduce_cuda::parallel_reduce_block_diag_{2x2,3x3}_cuda`.

## Local changes

Every kernel runs on PyTorch's current stream. Upstream launched them on the
legacy default stream, which PyTorch's streams don't wait for, so on any other
stream they could read inputs before they were written and return wrong
results.

- `torchlpc/cuda/lpc.cu`: launch on the current stream, check each launch,
  check the order limit with `TORCH_CHECK` instead of `assert`, and require
  the coefficients on the input's device.
- `torchlpc/cuda/linear_recurrence.cu`: launch on the current stream, check
  each launch, take the working memory from PyTorch's caching allocator
  instead of `cudaMalloc`/`cudaFree` (which synchronized the device and
  ignored errors), size it in `size_t`, make the initial states contiguous,
  require the decays and initial states on the input's device, and raise
  instead of overflowing past `INT_MAX` elements.
- `pararnn/csrc/parallel_reduction_kernel.h`: launch on the current stream,
  check each launch, and drop the `cudaDeviceSynchronize()` calls between the
  kernels, which the stream already orders.
- `pararnn/csrc/parallel_reduce.cu`: guard the device of the inputs.

The ParaRNN kernels can't launch for more than 65535 batch items or more than
`1024 * 1024 * chunk` steps; `philtorch/lpv/ssm.py` routes such inputs to the
other kernels.
