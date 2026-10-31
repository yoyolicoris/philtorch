#pragma once

// What the compiled ops need beyond torch's stable C++ API, which they use
// alone: torch/csrc/stable/* and torch/headeronly/*. Every converted source
// includes this header first. It sets the version the extension targets here,
// before any torch header is read, rather than with a compiler flag, so that
// it doesn't reach the Metal source (lti_recur_mps_replay.mm), which still
// uses the non-stable API; -DTORCH_TARGET_VERSION=... overrides it.
#ifndef TORCH_TARGET_VERSION
#define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 13) << 48))
#endif

#if defined(__CUDACC__) && !defined(USE_CUDA)
// torch's shim.h declares aoti_torch_get_current_cuda_stream only under it.
#define USE_CUDA
#endif

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/complex.h>

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace philtorch
{
    // The type kernels accumulate in, as at::opmath_type: float for Half and
    // BFloat16, the type itself otherwise.
    template <typename T>
    struct opmath
    {
        using type = T;
    };
    template <>
    struct opmath<c10::Half>
    {
        using type = float;
    };
    template <>
    struct opmath<c10::BFloat16>
    {
        using type = float;
    };
    template <>
    struct opmath<c10::complex<c10::Half>>
    {
        using type = c10::complex<float>;
    };
    template <typename T>
    using opmath_type = typename opmath<T>::type;

    template <typename T>
    constexpr T ceil_div(T a, T b)
    {
        return (a + b - 1) / b;
    }

    inline bool is_floating(torch::headeronly::ScalarType t)
    {
        using S = torch::headeronly::ScalarType;
        return t == S::Double || t == S::Float || t == S::Half ||
               t == S::BFloat16;
    }

    inline bool is_complex(torch::headeronly::ScalarType t)
    {
        using S = torch::headeronly::ScalarType;
        return t == S::ComplexDouble || t == S::ComplexFloat ||
               t == S::ComplexHalf;
    }

#ifdef __CUDACC__
    // The current CUDA stream of the device.
    inline cudaStream_t current_cuda_stream(torch::stable::DeviceIndex device)
    {
        void *stream = nullptr;
        STABLE_TORCH_ERROR_CODE_CHECK(
            aoti_torch_get_current_cuda_stream(device, &stream));
        return static_cast<cudaStream_t>(stream);
    }
#endif
} // namespace philtorch
