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

- `torchlpc/cuda/lpc.cu`: launch on the current stream, check each launch,
  check the order limit with `TORCH_CHECK` instead of `assert`, require
  the coefficients on the input's device, and include ATen in place of
  `torch/script.h` and `torch/torch.h`, which pull in Python's full API.
- `torchlpc/cuda/lpc.cu`: use only torch's stable C++ API (`torch/csrc/stable`,
  `torch/headeronly`), as the other native sources do: `STD_TORCH_CHECK`,
  `STD_CUDA_KERNEL_LAUNCH_CHECK`, the stable device guard, `THO_DISPATCH_V2`
  and `STABLE_TORCH_LIBRARY_IMPL`. The launches moved into two helper
  templates, because the launches did not compile inside the dispatch macro;
  `cat`, `flip`, `real`, `imag`, `stack` and `view_as_complex`,
  which have no stable wrapper, go through `philtorch::call_aten`
  (`philtorch/csrc/stable_support.h`), which includes it first.
