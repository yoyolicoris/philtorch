//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>          // Required for __half
#include <cuda_bf16.h>          // Required for __nv_bfloat16


template< typename scalar_t >
__forceinline__ __device__ scalar_t myTanh( scalar_t x ){
    return tanh(x);
}
template<>
__forceinline__ __device__ float myTanh( float x ){
    return tanhf(x);
}
template<>
__forceinline__ __device__ __half myTanh( __half x ){
    return __float2half(tanhf(__half2float(x)));
}
template<>
__forceinline__ __device__ __nv_bfloat16 myTanh( __nv_bfloat16 x ){
    return __float2bfloat16(tanhf(__bfloat162float(x)));
}

template< typename scalar_t >
__forceinline__ __device__ scalar_t tanhPrime( scalar_t x ){
    const scalar_t tanhx = myTanh(x);
    const scalar_t one = 1.;
    return one - tanhx * tanhx;
}


template< typename scalar_t >
__forceinline__ __device__ scalar_t myExp( scalar_t x ){
    return exp(x);
}
template<>
__forceinline__ __device__ float myExp( float x ){
    return expf(x);
}
template<>
__forceinline__ __device__ __half myExp( __half x ){
    return __float2half(expf(__half2float(x)));
}
template<>
__forceinline__ __device__ __nv_bfloat16 myExp( __nv_bfloat16 x ){
    return __float2bfloat16(expf(__bfloat162float(x)));
}


template< typename scalar_t >
__forceinline__ __device__ scalar_t mySigmoid( scalar_t x ){
    // fast sigmoid
//    const scalar_t half = 0.5;
//    const scalar_t one = 1.;
//    return half * (myTanh(half * x) + one);
    const scalar_t one = 1.;
    return one / (one + myExp(-x));
}
template< typename scalar_t >
__forceinline__ __device__ scalar_t sigmoidPrime( scalar_t x ){
    const scalar_t sigmax = mySigmoid(x);
    const scalar_t one = 1.;
    return sigmax * (one - sigmax);
}

template< typename scalar_t >
__forceinline__ __device__ scalar_t halfSigmoid( scalar_t x ){
    const scalar_t half = 0.5;
    return half * mySigmoid(x);
}
template< typename scalar_t >
__forceinline__ __device__ scalar_t halfSigmoidPrime( scalar_t x ){
    const scalar_t half = 0.5;
    return half * sigmoidPrime(x);
}
