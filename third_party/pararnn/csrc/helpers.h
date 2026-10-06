//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

#pragma once

// This is actually fixed
#ifndef THREADS_PER_WARP
#define THREADS_PER_WARP 32
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

#ifndef MAX_SEQUENTIAL_STEPS_DIAG
#define MAX_SEQUENTIAL_STEPS_DIAG 16
#endif

#ifndef MAX_SEQUENTIAL_STEPS_BLOCK_DIAG
#define MAX_SEQUENTIAL_STEPS_BLOCK_DIAG 8
#endif

#ifndef MAX_NEWTON_ITS
#define MAX_NEWTON_ITS 3
#endif


#ifndef FLOAT64_CHUNK_SIZE_DIAG
//#define FLOAT64_CHUNK_SIZE_DIAG 4
#error "FLOAT64_CHUNK_SIZE_DIAG must be defined via compiler flag (-DFLOAT64_CHUNK_SIZE_DIAG=...) both for cxx and nvcc"
#endif

#ifndef FLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2
//#define FLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2 1
#error "FLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2 must be defined via compiler flag (-FLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2=...) both for cxx and nvcc"
#endif


#include <torch/extension.h>
#include <ATen/Dispatch.h>              // Required for AT_DISPATCH macros
#include <c10/util/Half.h>              // at::Half (PyTorch half-precision type)
#include <c10/util/BFloat16.h>          // at::BFloat16 (PyTorch bfloat16 type)
#include <cuda_fp16.h>                  // Required for __half
#include <cuda_bf16.h>                  // Required for __nv_bfloat16


// Torch doesn't support a single dispatcher for floating + reduced types?
#ifndef AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF_AND_BFLOAT16
#define AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(...) \
    AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#endif

#ifndef AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16
#define AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(__VA_ARGS__))
#endif


// *********************************************************************************************************************
// Prescribe chunk sizes for Thomas reduction for various kernels / datatypes
// *********************************************************************************************************************
//// To sweep, this is pretty handy (comment the specifications below)
//template< const int num_hidden_vars, typename scalar_t> constexpr unsigned int dtype2chunkSizeBlockDiag = FLOAT64_CHUNK_SIZE_BLOCK_DIAG_2x2;
//template<typename scalar_t> constexpr unsigned int dtype2chunkSizeDiag       = FLOAT64_CHUNK_SIZE_DIAG;
//template<typename scalar_t> constexpr unsigned int dtype2chunkSizeGRU        = FLOAT64_CHUNK_SIZE_DIAG;
//template<typename scalar_t> constexpr unsigned int dtype2chunkSizeGRUmba     = FLOAT64_CHUNK_SIZE_DIAG;
//template<typename scalar_t> constexpr unsigned int dtype2chunkSizeLSTM       = FLOAT64_CHUNK_SIZE_DIAG;
//template<typename scalar_t> constexpr unsigned int dtype2chunkSizeLSTMCIFG   = FLOAT64_CHUNK_SIZE_DIAG;
//template<typename scalar_t> constexpr unsigned int dtype2chunkSizeLSTMbaCIFG = FLOAT64_CHUNK_SIZE_DIAG;

///*
// Diagonal Jacobian ***************************************************************************************************
template<typename scalar_t>
constexpr unsigned int dtype2chunkSizeDiag = 0;         // constexpr *must* have an init. I'm putting a bad number, so that hopefully it will fail immediately if something goes wrong - although good luck debugging!
template<> constexpr unsigned int dtype2chunkSizeDiag<double>        = 4;      // >= 8  spills
template<> constexpr unsigned int dtype2chunkSizeDiag<float>         = 2;      // >= 32 spills
template<> constexpr unsigned int dtype2chunkSizeDiag<at::Half>      = 4;      // >= 32 spills
template<> constexpr unsigned int dtype2chunkSizeDiag<at::BFloat16>  = 4;      // >= 32 spills
template<> constexpr unsigned int dtype2chunkSizeDiag<__half>        = dtype2chunkSizeDiag<at::Half>;
template<> constexpr unsigned int dtype2chunkSizeDiag<__nv_bfloat16> = dtype2chunkSizeDiag<at::BFloat16>;

// Block-diagonal Jacobian *********************************************************************************************
template< const int num_hidden_vars, typename scalar_t>
constexpr unsigned int dtype2chunkSizeBlockDiag = 0;    // constexpr *must* have an init. I'm putting a bad number, so that hopefully it will fail immediately if something goes wrong - although good luck debugging!
// 2x2 block diag
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<2,double>        = 2;   // >= 4  spills
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<2,float>         = 2;   // >= 8  spills
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<2,at::Half>      = 2;   // >= 16 spills
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<2,at::BFloat16>  = 2;   // >= 16 spills
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<2,__half>        = dtype2chunkSizeBlockDiag<2,at::Half>;
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<2,__nv_bfloat16> = dtype2chunkSizeBlockDiag<2,at::BFloat16>;
// 3x3 block diag
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<3,double>        = 1;    // >= 2 spills
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<3,float>         = 2;    // >= 4 spills
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<3,at::Half>      = 2;    // >= 8 spills
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<3,at::BFloat16>  = 2;    // >= 4 spills
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<3,__half>        = dtype2chunkSizeBlockDiag<3,at::Half>;
template<> constexpr unsigned int dtype2chunkSizeBlockDiag<3,__nv_bfloat16> = dtype2chunkSizeBlockDiag<3,at::BFloat16>;

// GRU *****************************************************************************************************************
template<typename scalar_t>
constexpr unsigned int dtype2chunkSizeGRU = 0;         // constexpr *must* have an init. I'm putting a bad number, so that hopefully it will fail immediately if something goes wrong - although good luck debugging!
template<> constexpr unsigned int dtype2chunkSizeGRU<double>        = 1;            // >= 1 spills!!!
template<> constexpr unsigned int dtype2chunkSizeGRU<float>         = 4;            // >= 8 spills
template<> constexpr unsigned int dtype2chunkSizeGRU<at::Half>      = 4;            // >= 8 spills
template<> constexpr unsigned int dtype2chunkSizeGRU<at::BFloat16>  = 4;            // >= 8 spills
template<> constexpr unsigned int dtype2chunkSizeGRU<__half>        = dtype2chunkSizeGRU<at::Half>;
template<> constexpr unsigned int dtype2chunkSizeGRU<__nv_bfloat16> = dtype2chunkSizeGRU<at::BFloat16>;

// LSTM_CIFG ***********************************************************************************************************
template< typename scalar_t >
constexpr unsigned int dtype2chunkSizeLSTMCIFG = 0;         // constexpr *must* have an init. I'm putting a bad number, so that hopefully it will fail immediately if something goes wrong - although good luck debugging!
template<> constexpr unsigned int dtype2chunkSizeLSTMCIFG<double>        = 1;       // >= 2  spills
template<> constexpr unsigned int dtype2chunkSizeLSTMCIFG<float>         = 1;       // >= 8  spills
template<> constexpr unsigned int dtype2chunkSizeLSTMCIFG<at::Half>      = 4;       // >= 8  spills
template<> constexpr unsigned int dtype2chunkSizeLSTMCIFG<at::BFloat16>  = 8;       // >= 16 spills
template<> constexpr unsigned int dtype2chunkSizeLSTMCIFG<__half>        = dtype2chunkSizeLSTMCIFG<at::Half>;
template<> constexpr unsigned int dtype2chunkSizeLSTMCIFG<__nv_bfloat16> = dtype2chunkSizeLSTMCIFG<at::BFloat16>;

//*/

// *********************************************************************************************************************
// Common functions declarations
// *********************************************************************************************************************
template< typename pimpl_t,
          int max_sequential_steps >
void parallelReduceLauncher(
        typename pimpl_t::scalar_t *jac,
        typename pimpl_t::scalar_t *rhs,
        typename pimpl_t::scalar_t *jacTemp,
        typename pimpl_t::scalar_t *rhsTemp,
        typename pimpl_t::idx_t seqLength,
        typename pimpl_t::idx_t hiddenDim,
        typename pimpl_t::idx_t batchSize
);


template< typename pimpl_t >
__device__ void parallelReductionSharedInner(
    typename pimpl_t::jac_t (&myRegJac)[pimpl_t::chunk_size],
    typename pimpl_t::rhs_t (&myRegRhs)[pimpl_t::chunk_size],
    typename pimpl_t::jac_t *sharedJac,
    typename pimpl_t::rhs_t *sharedRhs,
    int numMyEqs,
    typename pimpl_t::idx_t seqLength
);


torch::Tensor parallel_reduce_diag_cuda(
        torch::Tensor jac,    // (B,) N, T, K, K
        torch::Tensor rhs     // (B,) N, T, K
);


template< int num_hidden_vars >
torch::Tensor parallel_reduce_block_diag_cuda(
        torch::Tensor jac,    // (B,) N, T, K, K
        torch::Tensor rhs     // (B,) N, T, K
);



torch::Tensor fused_fwd_gru_diag_mh(
    torch::Tensor A,
    torch::Tensor B
);
torch::Tensor fused_bwd_gru_diag_mh(
    torch::Tensor grad,
    torch::Tensor h,
    torch::Tensor A,
    torch::Tensor B
);


torch::Tensor fused_fwd_lstm_cifg_diag_mh(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
);
torch::Tensor fused_bwd_lstm_cifg_diag_mh(
    torch::Tensor grad,
    torch::Tensor h,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
);
