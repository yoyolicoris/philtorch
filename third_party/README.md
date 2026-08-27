# third_party

Vendored extensions built as part of `philtorch._C` (single shared library).

- `torchlpc` — `git@github.com:DiffAPF/torchlpc.git` at `1bfde4a` (`v0.7.2-18-g1bfde4a`, `origin/dev`)
  LICENSE: `third_party/torchlpc/LICENSE` (MIT, Copyright (c) 2023 Chin-Yun Yu)
  Sources used: `torchlpc/csrc/scan_cpu.cpp` (via shim `philtorch/csrc/torchlpc_shim.cpp` to avoid duplicate `PyInit__C`), `torchlpc/csrc/cuda/{lpc.cu,linear_recurrence.cu}`
  Python wrappers vendored as `philtorch/_torchlpc.py` (mirrors `sample_wise_lpc`).

- `pararnn` — `git@github.com:apple/ml-pararnn.git` at `dc2647b` (`origin/main`)
  LICENSE: `third_party/pararnn/LICENSE`
  Sources used (CUDA only): `pararnn/csrc/{parallel_reduction_bindings.cpp,parallel_reduce.cu,fused_gru_diag.cu,fused_lstm_cifg_diag.cu}` with flags `-DFLOAT64_CHUNK_SIZE_DIAG=4 -DFLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2=1`.

Build: `setup.py:get_extensions()` compiles them into `philtorch._C`; `TORCH_LIBRARY(torchlpc, ...)` and `TORCH_LIBRARY(parallel_reduce_cuda, ...)` remain as separate namespaces but linked in one `.so`, so `torch.ops.torchlpc.allpole` and `torch.ops.parallel_reduce_cuda.*` keep working without external pip deps.
