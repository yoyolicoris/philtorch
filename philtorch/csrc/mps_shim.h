#pragma once

// What the Metal sources share of torch's Metal C shim (shim_mps.h), which
// compiles shader source, binds tensors and dispatches on PyTorch's MPS
// stream.
#include "stable_support.h"

#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>

#include <array>
#include <initializer_list>
#include <mutex>
#include <string>

// MPS_COMMON_METAL, the text of mps_common.metal, which setup.py generates in
// the build directory.
#include "mps_common.metal.h"

namespace philtorch::mps
{

using torch::headeronly::ScalarType;

// The library of a shader source, after its definitions and mps_common.metal;
// libraries stay for the process.
inline AOTIMetalShaderLibraryHandle compile(const std::string &definitions,
                                            const char *source)
{
    const std::string text = definitions + MPS_COMMON_METAL + source;
    AOTIMetalShaderLibraryHandle created = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_mps_create_shader_library(text.c_str(), &created));
    return created;
}

// The Metal type of the dtypes the kernels take, float2 being complex64;
// `op` names the kernels in the error for the others.
inline const char *metal_type(ScalarType dtype, const char *op)
{
    switch (dtype)
    {
    case ScalarType::Float:
        return "float";
    case ScalarType::Half:
        return "half";
    case ScalarType::BFloat16:
        return "bfloat";
    default:
        STD_TORCH_CHECK(dtype == ScalarType::ComplexFloat, op,
                        " takes float32, float16, bfloat16 and complex64");
        return "float2";
    }
}

// The dtype the kernels accumulate dtype in: float32, or complex64.
inline ScalarType accumulation_type(ScalarType dtype)
{
    return dtype == ScalarType::ComplexFloat ? dtype : ScalarType::Float;
}

inline AOTIMetalKernelFunctionHandle
kernel_function(AOTIMetalShaderLibraryHandle library, const std::string &name)
{
    AOTIMetalKernelFunctionHandle function = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_mps_get_kernel_function(library, name.c_str(), &function));
    return function;
}

// One dispatch's arguments: its tensors, then its integers, bound in that
// order from index 0, over a grid of grid[0] x grid[1] threads in
// threadgroups of group[0] x group[1].
struct Launch
{
    std::initializer_list<AtenTensorHandle> tensors;
    std::initializer_list<int64_t> integers;
    std::array<uint64_t, 2> grid, group;
    AOTITorchError error = AOTI_TORCH_SUCCESS;
};

// Runs inside the shim's command block: it records the first error, which
// launch() raises, rather than throwing through the shim's C frames.
inline void encode(AOTIMetalKernelFunctionHandle function, void *data)
{
    Launch &launch = *static_cast<Launch *>(data);
    const auto ok = [&] { return launch.error == AOTI_TORCH_SUCCESS; };
    launch.error = aoti_torch_mps_start_encoding(function);
    unsigned index = 0;
    for (AtenTensorHandle tensor : launch.tensors)
        if (ok())
            launch.error = aoti_torch_mps_set_arg_tensor(function, index++, tensor);
    for (int64_t integer : launch.integers)
        if (ok())
            launch.error = aoti_torch_mps_set_arg_int(function, index++, integer);
    if (!ok())
        return;
    launch.error = aoti_torch_mps_dispatch_array_with_group_size(
        function, launch.grid.data(), launch.grid.size(), launch.group.data(),
        launch.group.size());
}

inline void launch(AOTIMetalKernelFunctionHandle function,
                   std::initializer_list<AtenTensorHandle> tensors,
                   std::initializer_list<int64_t> integers,
                   std::array<uint64_t, 2> grid, std::array<uint64_t, 2> group)
{
    // A kernel handle holds the encoder of its command block, and the handles
    // are shared, so calls on different MPS streams encode one at a time.
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    Launch arguments{tensors, integers, grid, group};
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_mps_run_command_block(function, encode, &arguments));
    TORCH_ERROR_CODE_CHECK(arguments.error);
}

inline bool is_mps(const torch::stable::Tensor &t)
{
    return t.device().type() == torch::headeronly::DeviceType::MPS;
}

} // namespace philtorch::mps
