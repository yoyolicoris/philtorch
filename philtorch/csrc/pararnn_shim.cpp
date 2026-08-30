#include <torch/extension.h>

// Register the vendored CUDA kernels without upstream's Python module.
template <int N>
at::Tensor parallel_reduce_block_diag_cuda(at::Tensor jac, at::Tensor rhs);

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
