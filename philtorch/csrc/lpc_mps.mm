// lpc on MPS, on torch's stable ABI: the stable C++ API and torch's Metal C
// shim (shim_mps.h), as in scalar_recur_mps.mm. Serial over time, as on CPU
// and CUDA: one threadgroup per sequence, one thread per coefficient.
#include "mps_shim.h"
#include "recur_checks.h"

#include <limits>

// LPC_MPS_METAL, the text of lpc_mps.metal, which setup.py generates in the
// build directory.
#include "lpc_mps.metal.h"

namespace
{

using torch::headeronly::ScalarType;
using philtorch::mps::SIMD_WIDTH;
using torch::stable::Tensor;

// The kernels' name in errors.
constexpr const char *OP = "the MPS lpc";

// The most coefficients, one thread each, as on CUDA; the kernels' own limit.
constexpr int64_t MAX_ORDER = 1024;

AOTIMetalKernelFunctionHandle kernel(ScalarType dtype)
{
    static const AOTIMetalShaderLibraryHandle library =
        philtorch::mps::compile("", LPC_MPS_METAL);
    const auto function = [&](const char *name)
    { return philtorch::mps::kernel_function(library, name); };
    // All looked up on the first call, one at a time under the statics'
    // initialization, before any launch.
    static const AOTIMetalKernelFunctionHandle f = function("lpc_float"),
                                               h = function("lpc_half"),
                                               bf = function("lpc_bfloat"),
                                               c = function("lpc_float2"),
                                               ch = function("lpc_half2");
    switch (dtype)
    {
    case ScalarType::Float:
        return f;
    case ScalarType::Half:
        return h;
    case ScalarType::BFloat16:
        return bf;
    case ScalarType::ComplexHalf:
        return ch;
    default:
        return c;
    }
}

// y[b, t] = x[b, t] - sum_i a[b, t, i] y[b, t - 1 - i], from y[b, -1 - k] =
// zi[b, k]; x and y are (B, T), a (B, T, M), zi (B, M).
Tensor lpc_mps(const Tensor &x, const Tensor &a, const Tensor &zi)
{
    check_lpc_inputs(x, a, zi);
    STD_TORCH_CHECK(a.device() == x.device(),
                    "Coefficients must be on the same device as input");
    STD_TORCH_CHECK(zi.device() == x.device(),
                    "Initial conditions must be on the same device as input");
    const ScalarType dtype = x.scalar_type();
    STD_TORCH_CHECK(dtype == ScalarType::Float || dtype == ScalarType::Half ||
                        dtype == ScalarType::BFloat16 ||
                        dtype == ScalarType::ComplexFloat ||
                        dtype == ScalarType::ComplexHalf,
                    OP, " takes float32, float16, bfloat16, complex64 and complex32");
    const int64_t n_batches = x.size(0), steps = x.size(1), order = a.size(2);
    STD_TORCH_CHECK(order <= MAX_ORDER,
                    "LPC order must be less than or equal to 1024");
    // The kernel indexes the steps of a sequence in 32 bits.
    STD_TORCH_CHECK(steps <= std::numeric_limits<int32_t>::max() &&
                        n_batches <= std::numeric_limits<uint32_t>::max(),
                    "the input is too large for ", OP);

    const auto x_contiguous = torch::stable::contiguous(x);
    if (n_batches == 0 || steps == 0)
        return torch::stable::empty_like(x_contiguous);
    // No coefficients: y = x.
    if (order == 0)
        return torch::stable::clone(x_contiguous);

    const auto a_contiguous = torch::stable::contiguous(a);
    const auto zi_contiguous = torch::stable::contiguous(zi);
    auto output = torch::stable::empty_like(x_contiguous);
    // One thread per coefficient, in whole SIMD groups.
    const uint64_t threads = static_cast<uint64_t>(
        philtorch::ceil_div<int64_t>(order, SIMD_WIDTH) * SIMD_WIDTH);
    philtorch::mps::launch(kernel(dtype),
                           {x_contiguous.get(), a_contiguous.get(),
                            zi_contiguous.get(), output.get()},
                           {steps, order},
                           {threads, static_cast<uint64_t>(n_batches)},
                           {threads, 1});
    return output;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(philtorch, MPS, m)
{
    m.impl("lpc", TORCH_BOX(&lpc_mps));
}
