#include <torch/extension.h>

// Pararnn shim - vendored kernel registrations without duplicate PyInit.
// Original file third_party/pararnn/pararnn/csrc/parallel_reduction_bindings.cpp
// contains PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...) which defines PyInit__C
// colliding with host_recur2.cpp's PyInit__C. This shim keeps only the
// TORCH_LIBRARY registrations for the kernels philtorch actually uses,
// omitting the pybind module and the unused GRU/LSTM/diag kernels.

// Forward declaration - defined in the other pararnn CUDA sources
template <int N>
at::Tensor parallel_reduce_block_diag_cuda(const at::Tensor &a, const at::Tensor &b);

// Only the block-diagonal parallel-reduce kernels are used by philtorch
// (see philtorch/lpv/ssm.py MatrixRecurrence for M == 2 and M == 3).
TORCH_LIBRARY(parallel_reduce_cuda, m)
{
  m.def("parallel_reduce_block_diag_2x2_cuda(Tensor a, Tensor b) -> Tensor");
  m.def("parallel_reduce_block_diag_3x3_cuda(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(parallel_reduce_cuda, CUDA, m)
{
  m.impl("parallel_reduce_block_diag_2x2_cuda", &parallel_reduce_block_diag_cuda<2>);
  m.impl("parallel_reduce_block_diag_3x3_cuda", &parallel_reduce_block_diag_cuda<3>);
}

// NOTE: No PYBIND11_MODULE here - that's the whole point of this shim.
// The original file's PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...) would define
// PyInit__C and collide with host_recur2.cpp's PyInit__C when linked into
// the same philtorch._C extension.
