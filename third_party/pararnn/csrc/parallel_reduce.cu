//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#include <torch/extension.h>
#include <cuda_fp16.h>          // Required for __half
#include <cuda_bf16.h>          // Required for __nv_bfloat16
#include <c10/util/Half.h>      // Defines c10::Half
#include <c10/util/BFloat16.h>  // Defines c10::BFloat16

#include "helpers.h"
#include "rnn_cell_impl.h"
#include "parallel_reduction_kernel.h"




// Implementation for block diagonal ***********************************************************************************
template< const int num_hidden_vars >
torch::Tensor parallel_reduce_block_diag_cuda(
        torch::Tensor jac,    // (B,) N, T, K, K
        torch::Tensor rhs     // (B,) N, T, K
    ) {
    TORCH_CHECK(jac.scalar_type() == rhs.scalar_type(), "Input tensors must have the same type");

    // Tweak to make this func work with both batched and non-batched inputs
    const bool batched = rhs.dim() > 3;
    unsigned int batchSize = batched? jac.size(0) : 1;
    unsigned int hiddenDim = jac.size(batched+0);
    unsigned int seqLength = jac.size(batched+1);
    unsigned int numBlocks = jac.size(batched+2);

    // Be efficient about extra memory allocation
//    at::Tensor jacTemp = jac.clone().to(torch::kCUDA).contiguous(); // This ensure tensors in signature are non-modifiable
//    at::Tensor rhsTemp = rhs.clone().to(torch::kCUDA).contiguous(); // and allows us to use this with torch.compile()
    const bool mustCopyRhs = !rhs.is_contiguous() || !rhs.is_cuda();
    torch::Tensor rhsTemp = torch::empty_like(rhs);          // output must be allocated regardless
    if( mustCopyRhs ){                                       // if rhs is not contiguous/on device, I need to copy it
        rhsTemp = rhs.to(torch::kCUDA).contiguous();
    }

    const bool mustCopyJac = !jac.is_contiguous() || !jac.is_cuda();
    torch::Tensor jacTemp;
    if( mustCopyJac ){                                       // if jac is not contiguous/on device, I need to copy it
        jacTemp = jac.to(torch::kCUDA).contiguous();
    }
//    else if( seqLength > THREADS_PER_BLOCK * dtype2chunkSizeBlockDiag2x2<jac.scalar_type()> ){ // if tensor is very long, I need to store temporary data
//        jacTemp = torch::empty_like(jac);
//    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(jac.scalar_type(), "Launching block-diag parallel solver", ([&] {
        using cuda_scalar_t = std::conditional_t<
            std::is_same_v<scalar_t, at::Half>, __half,                                            // to cast at::Half to cuda native __half
            std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, __nv_bfloat16, scalar_t>>;  // to cast at::BFloat16 to cuda native __nv_bfloat16

        if( !mustCopyJac && seqLength > THREADS_PER_BLOCK * dtype2chunkSizeBlockDiag<num_hidden_vars, scalar_t> ){ // if tensor is very long, I need to store temporary data
            jacTemp = torch::empty_like(jac);
        }

        parallelReduceLauncher<
            RNNCellBlockDiagImpl< cuda_scalar_t, num_hidden_vars >, MAX_SEQUENTIAL_STEPS_BLOCK_DIAG
        >(
            reinterpret_cast<cuda_scalar_t*>(mustCopyJac? jacTemp.data_ptr<scalar_t>() : jac.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(mustCopyRhs? rhsTemp.data_ptr<scalar_t>() : rhs.data_ptr<scalar_t>()),
            jacTemp.defined() ? reinterpret_cast<cuda_scalar_t*>(jacTemp.data_ptr<scalar_t>()) : nullptr,
            reinterpret_cast<cuda_scalar_t*>(rhsTemp.data_ptr<scalar_t>()),
            seqLength, hiddenDim, batchSize
        );
    }));

    return rhsTemp;
}


// Explicit instantiation of the template function for supported block sizes
template torch::Tensor parallel_reduce_block_diag_cuda< 2 >(torch::Tensor, torch::Tensor);  //2x2 blocks
template torch::Tensor parallel_reduce_block_diag_cuda< 3 >(torch::Tensor, torch::Tensor);  //3x3 blocks




// Implementation for diagonal *****************************************************************************************
torch::Tensor parallel_reduce_diag_cuda(
        torch::Tensor jac,    // (B,) N, T
        torch::Tensor rhs     // (B,) N, T
    ) {
    TORCH_CHECK(jac.scalar_type() == rhs.scalar_type(), "Input tensors must have the same type");

    // Tweak to make this func work with both batched and non-batched inputs
    const bool batched = rhs.dim() > 2;
    unsigned int batchSize = batched? jac.size(0) : 1;
    unsigned int hiddenDim = jac.size(batched+0);
    unsigned int seqLength = jac.size(batched+1);

    // Be efficient about extra memory allocation
    const bool mustCopyRhs = !rhs.is_contiguous() || !rhs.is_cuda();
    torch::Tensor rhsTemp = torch::empty_like(rhs);          // output must be allocated regardless
    if( mustCopyRhs ){                                       // if rhs is not contiguous/on device, I need to copy it
        rhsTemp = rhs.to(torch::kCUDA).contiguous();
    }

    const bool mustCopyJac = !jac.is_contiguous() || !jac.is_cuda();
    torch::Tensor jacTemp;
    if( mustCopyJac ){                                       // if jac is not contiguous/on device, I need to copy it
        jacTemp = jac.to(torch::kCUDA).contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(jac.scalar_type(), "Launching diag parallel solver", ([&] {
        using cuda_scalar_t = std::conditional_t<
            std::is_same_v<scalar_t, at::Half>, __half,                                            // to cast at::Half to cuda native __half
            std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, __nv_bfloat16, scalar_t>>;  // to cast at::BFloat16 to cuda native __nv_bfloat16

        if( !mustCopyJac && seqLength > THREADS_PER_BLOCK * dtype2chunkSizeDiag<scalar_t> ){ // if tensor is very long, I need to store temporary data
            jacTemp = torch::empty_like(jac);
        }

        parallelReduceLauncher<
            RNNCellDiagImpl< cuda_scalar_t >, MAX_SEQUENTIAL_STEPS_DIAG
        >(
            reinterpret_cast<cuda_scalar_t*>(mustCopyJac? jacTemp.data_ptr<scalar_t>() : jac.data_ptr<scalar_t>()),
            reinterpret_cast<cuda_scalar_t*>(mustCopyRhs? rhsTemp.data_ptr<scalar_t>() : rhs.data_ptr<scalar_t>()),
            jacTemp.defined() ? reinterpret_cast<cuda_scalar_t*>(jacTemp.data_ptr<scalar_t>()) : nullptr,
            reinterpret_cast<cuda_scalar_t*>(rhsTemp.data_ptr<scalar_t>()),
            seqLength, hiddenDim, batchSize
        );
    }));

    return rhsTemp;
}



