#include <torch/extension.h>
#include "helpers.h"

// Pararnn shim - vendored kernel registrations without duplicate PyInit
// Original file third_party/pararnn/pararnn/csrc/parallel_reduction_bindings.cpp
// contains PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...) which defines PyInit__C
// colliding with host_recur2.cpp's PyInit__C. This shim keeps only the
// TORCH_LIBRARY definitions needed for the ops, omitting the pybind module.

// Forward declarations - these are defined in the other pararnn CUDA sources
at::Tensor parallel_reduce_diag_cuda(const at::Tensor &a, const at::Tensor &b);
template<int N>
at::Tensor parallel_reduce_block_diag_cuda(const at::Tensor &a, const at::Tensor &b);
at::Tensor fused_fwd_gru_diag_mh(const at::Tensor &a, const at::Tensor &b);
at::Tensor fused_bwd_gru_diag_mh(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c, const at::Tensor &d);
at::Tensor fused_fwd_lstm_cifg_diag_mh(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c);
at::Tensor fused_bwd_lstm_cifg_diag_mh(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c, const at::Tensor &d, const at::Tensor &e);

// Helpers for chunk sizes - copied from upstream bindings
unsigned int get_diag_chunk_size(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Double: return dtype2chunkSizeDiag<double>;
    case at::ScalarType::Float: return dtype2chunkSizeDiag<float>;
    case at::ScalarType::Half: return dtype2chunkSizeDiag<at::Half>;
    case at::ScalarType::BFloat16: return dtype2chunkSizeDiag<at::BFloat16>;
    default: throw std::invalid_argument("Unsupported dtype in get_diag_chunk_size()");
  }
}
unsigned int get_block_diag_2x2_chunk_size(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Double: return dtype2chunkSizeBlockDiag<2,double>;
    case at::ScalarType::Float: return dtype2chunkSizeBlockDiag<2,float>;
    case at::ScalarType::Half: return dtype2chunkSizeBlockDiag<2,at::Half>;
    case at::ScalarType::BFloat16: return dtype2chunkSizeBlockDiag<2,at::BFloat16>;
    default: throw std::invalid_argument("Unsupported dtype in get_block_diag_2x2_chunk_size()");
  }
}
unsigned int get_block_diag_3x3_chunk_size(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Double: return dtype2chunkSizeBlockDiag<3,double>;
    case at::ScalarType::Float: return dtype2chunkSizeBlockDiag<3,float>;
    case at::ScalarType::Half: return dtype2chunkSizeBlockDiag<3,at::Half>;
    case at::ScalarType::BFloat16: return dtype2chunkSizeBlockDiag<3,at::BFloat16>;
    default: throw std::invalid_argument("Unsupported dtype in get_block_diag_3x3_chunk_size()");
  }
}
unsigned int get_fused_gru_chunk_size(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Double: return dtype2chunkSizeGRU<double>;
    case at::ScalarType::Float: return dtype2chunkSizeGRU<float>;
    case at::ScalarType::Half: return dtype2chunkSizeGRU<at::Half>;
    case at::ScalarType::BFloat16: return dtype2chunkSizeGRU<at::BFloat16>;
    default: throw std::invalid_argument("Unsupported dtype in get_fused_gru_chunk_size()");
  }
}
unsigned int get_fused_lstm_cifg_chunk_size(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Double: return dtype2chunkSizeLSTMCIFG<double>;
    case at::ScalarType::Float: return dtype2chunkSizeLSTMCIFG<float>;
    case at::ScalarType::Half: return dtype2chunkSizeLSTMCIFG<at::Half>;
    case at::ScalarType::BFloat16: return dtype2chunkSizeLSTMCIFG<at::BFloat16>;
    default: throw std::invalid_argument("Unsupported dtype in get_fused_lstm_cifg_chunk_size()");
  }
}
int64_t get_threads_per_block(){ return static_cast<int64_t>(THREADS_PER_BLOCK); }
int64_t get_threads_per_warp(){ return static_cast<int64_t>(THREADS_PER_WARP); }

// TORCH_LIBRARY definitions - same as upstream, but without PYBIND11_MODULE
TORCH_LIBRARY(parallel_reduce_cuda, m) {
  m.def("parallel_reduce_diag_cuda(Tensor a, Tensor b) -> Tensor");
  m.def("parallel_reduce_block_diag_2x2_cuda(Tensor a, Tensor b) -> Tensor");
  m.def("parallel_reduce_block_diag_3x3_cuda(Tensor a, Tensor b) -> Tensor");
  m.def("fused_fwd_gru_diag_mh(Tensor a, Tensor b) -> Tensor");
  m.def("fused_bwd_gru_diag_mh(Tensor a, Tensor b, Tensor c, Tensor d) -> Tensor");
  m.def("fused_fwd_lstm_cifg_diag_mh(Tensor a, Tensor b, Tensor c) -> Tensor");
  m.def("fused_bwd_lstm_cifg_diag_mh(Tensor a, Tensor b, Tensor c, Tensor d, Tensor e) -> Tensor");
  // Utility ops for chunk sizes - expose as torch ops for completeness
  m.def("get_diag_chunk_size(int dtype) -> int");
  m.def("get_block_diag_2x2_chunk_size(int dtype) -> int");
  m.def("get_block_diag_3x3_chunk_size(int dtype) -> int");
  m.def("get_fused_gru_chunk_size(int dtype) -> int");
  m.def("get_fused_lstm_cifg_chunk_size(int dtype) -> int");
  m.def("get_threads_per_block() -> int");
  m.def("get_threads_per_warp() -> int");
}

TORCH_LIBRARY_IMPL(parallel_reduce_cuda, CUDA, m) {
  m.impl("parallel_reduce_diag_cuda", &parallel_reduce_diag_cuda);
  m.impl("parallel_reduce_block_diag_2x2_cuda", &parallel_reduce_block_diag_cuda<2>);
  m.impl("parallel_reduce_block_diag_3x3_cuda", &parallel_reduce_block_diag_cuda<3>);
  m.impl("fused_fwd_gru_diag_mh", &fused_fwd_gru_diag_mh);
  m.impl("fused_bwd_gru_diag_mh", &fused_bwd_gru_diag_mh);
  m.impl("fused_fwd_lstm_cifg_diag_mh", &fused_fwd_lstm_cifg_diag_mh);
  m.impl("fused_bwd_lstm_cifg_diag_mh", &fused_bwd_lstm_cifg_diag_mh);
  m.impl("get_diag_chunk_size", &get_diag_chunk_size);
  m.impl("get_block_diag_2x2_chunk_size", &get_block_diag_2x2_chunk_size);
  m.impl("get_block_diag_3x3_chunk_size", &get_block_diag_3x3_chunk_size);
  m.impl("get_fused_gru_chunk_size", &get_fused_gru_chunk_size);
  m.impl("get_fused_lstm_cifg_chunk_size", &get_fused_lstm_cifg_chunk_size);
  m.impl("get_threads_per_block", &get_threads_per_block);
  m.impl("get_threads_per_warp", &get_threads_per_warp);
}

// NOTE: No PYBIND11_MODULE here - that's the whole point of this shim.
// The original file's PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...) would define
// PyInit__C and collide with host_recur2.cpp's PyInit__C when linked into
// the same philtorch._C extension.
