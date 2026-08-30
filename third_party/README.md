# third_party

Vendored extensions built as part of `philtorch._C` (single shared library).

Both projects are pinned git submodules (see `.gitmodules`). Their full license
texts are mirrored in `LICENSES/` and summarized in the top-level `NOTICE`, so
the required copyright and permission notices ship in source distributions and
wheels even when the submodules are not checked out.

- `torchlpc` — `https://github.com/DiffAPF/torchlpc.git` at `1bfde4a` (`v0.7.2-18-g1bfde4a`, `origin/dev`)
  LICENSE: `LICENSES/torchlpc-LICENSE` (MIT, Copyright (c) 2023 Chin-Yun Yu)
  Sources used: `third_party/torchlpc/torchlpc/csrc/scan_cpu.cpp` (via shim `philtorch/csrc/torchlpc_shim.cpp` to avoid duplicate `PyInit__C`), `third_party/torchlpc/torchlpc/csrc/cuda/{lpc.cu,linear_recurrence.cu}`
  Python wrappers vendored as `philtorch/_torchlpc.py` (mirrors `sample_wise_lpc`).

- `pararnn` — `https://github.com/apple/ml-pararnn.git` at `dc2647b` (`origin/main`)
  LICENSE: `LICENSES/pararnn-LICENSE` (Copyright (C) 2025 Apple Inc.)
  Sources used (CUDA only): `third_party/pararnn/pararnn/csrc/parallel_reduce.cu` (only the
  block-diagonal parallel-reduce kernels philtorch uses; registered via shim
  `philtorch/csrc/pararnn_shim.cpp`) with flags `-DFLOAT64_CHUNK_SIZE_DIAG=4 -DFLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2=1`. The fused GRU/LSTM and diag kernels are not compiled.

Build: `setup.py:get_extensions()` compiles them into `philtorch._C`. The
torchlpc kernels are registered privately as `philtorch::lpc` and
`philtorch::scan`, avoiding conflicts with an independently installed torchlpc
extension. The PararNN kernels remain under
`parallel_reduce_cuda::parallel_reduce_block_diag_{2x2,3x3}_cuda`.
