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

None: the files are copied unchanged.
