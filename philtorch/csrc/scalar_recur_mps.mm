// The scalar recurrence on MPS, time-invariant and time-varying, on torch's
// stable ABI: the stable C++ API and torch's Metal C shim (shim_mps.h), which
// compiles the shader source, binds tensors and dispatches on PyTorch's MPS
// stream. stable_support.h holds it to the API of the package's floor torch.
#include "mps_shim.h"

#include <array>
#include <limits>
#include <string>

// SCALAR_RECUR_MPS_METAL, the text of scalar_recur_mps.metal, which setup.py
// generates in the build directory.
#include "scalar_recur_mps.metal.h"

namespace
{

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// The kernels' name in errors.
constexpr const char *OP = "the MPS scalar recurrence";

// Threads per threadgroup and steps per thread, as in scalar_recur_mps.metal.
constexpr uint64_t THREADS = 256;
constexpr int64_t STEPS_PER_THREAD = 4;


AOTIMetalShaderLibraryHandle library()
{
    static const AOTIMetalShaderLibraryHandle handle =
        philtorch::mps::compile("", SCALAR_RECUR_MPS_METAL);
    return handle;
}

// A dtype's kernels; reduce and replay for time-invariant coefficients, then
// time-varying ones.
struct Kernels
{
    std::array<AOTIMetalKernelFunctionHandle, 2> reduce, replay;
    AOTIMetalKernelFunctionHandle scan;
};

AOTIMetalKernelFunctionHandle kernel_function(const std::string &name)
{
    return philtorch::mps::kernel_function(library(), name);
}

// The kernels of dtype, whose names end in its Metal type, and the scan of
// tile totals of its accumulation type.
Kernels make_kernels(ScalarType dtype)
{
    using philtorch::mps::metal_type;
    const std::string suffix = metal_type(dtype, OP);
    const std::string accumulation =
        metal_type(philtorch::mps::accumulation_type(dtype), OP);
    return {{kernel_function("scalar_recur_reduce_" + suffix + "_lti"),
             kernel_function("scalar_recur_reduce_" + suffix + "_lpv")},
            {kernel_function("scalar_recur_replay_" + suffix + "_lti"),
             kernel_function("scalar_recur_replay_" + suffix + "_lpv")},
            kernel_function("scalar_recur_scan_" + accumulation)};
}

// The kernels of dtype, all looked up on the first call, one at a time under
// the statics' initialization, before any launch.
const Kernels &kernels(ScalarType dtype)
{
    philtorch::mps::metal_type(dtype, OP);
    static const Kernels f = make_kernels(ScalarType::Float),
                         h = make_kernels(ScalarType::Half),
                         bf = make_kernels(ScalarType::BFloat16),
                         c = make_kernels(ScalarType::ComplexFloat);
    switch (dtype)
    {
    case ScalarType::Float:
        return f;
    case ScalarType::Half:
        return h;
    case ScalarType::BFloat16:
        return bf;
    default:
        return c;
    }
}

// One dispatch over a grid of grid[0] x grid[1] threads in rows of THREADS.
void launch(AOTIMetalKernelFunctionHandle function,
            std::initializer_list<AtenTensorHandle> tensors,
            std::initializer_list<int64_t> integers,
            std::array<uint64_t, 2> grid)
{
    philtorch::mps::launch(function, tensors, integers, grid, {THREADS, 1});
}

// h[b, t] = a h[b, t - 1] + x[b, t], h[b, -1] = zi[b], with a = decays[b, t]
// if varying, with stride_b = T, else decays[b * stride_b], shared (0) or one
// per sequence (1).
Tensor scalar_recur_mps(const Tensor &decays, bool varying, int64_t stride_b,
                        const Tensor &zi, const Tensor &x)
{
    using philtorch::mps::is_mps;
    STD_TORCH_CHECK(is_mps(decays) && is_mps(zi) && is_mps(x),
                    "the coefficients, initial states and inputs must be MPS tensors");
    const ScalarType dtype = x.scalar_type();
    STD_TORCH_CHECK(decays.scalar_type() == dtype && zi.scalar_type() == dtype,
                    "the coefficients, initial states and inputs must have the same dtype");
    const Kernels &k = kernels(dtype);

    const auto decays_contiguous = torch::stable::contiguous(decays);
    const auto zi_contiguous = torch::stable::contiguous(zi);
    const auto x_contiguous = torch::stable::contiguous(x);
    auto output = torch::stable::empty_like(x_contiguous);
    const int64_t n_batches = x.size(0);
    const int64_t steps = x.size(1);
    if (n_batches == 0 || steps == 0)
        return output;

    const int64_t tile = static_cast<int64_t>(THREADS) * STEPS_PER_THREAD;
    const int64_t n_tiles = (steps - 1) / tile + 1;
    constexpr int64_t grid_limit = std::numeric_limits<uint32_t>::max();
    // The kernels index the steps of a sequence, the last tile's past its end
    // included, in 32 bits, which bounds the grid's n_tiles * THREADS too.
    STD_TORCH_CHECK(steps <= grid_limit - tile && n_batches <= grid_limit,
                    "the input is too large for the MPS scalar recurrence");

    const std::array<uint64_t, 2> tile_grid{
        static_cast<uint64_t>(n_tiles) * THREADS,
        static_cast<uint64_t>(n_batches)};
    // A sequence of one tile needs no state from before it: the replay alone.
    Tensor states = output;
    if (n_tiles > 1)
    {
        // Each tile's map, two values of the accumulation type, and the state
        // at its end.
        const bool complex = dtype == ScalarType::ComplexFloat;
        const int64_t values = complex ? 2 : 1;
        auto totals = torch::stable::new_empty(
            x_contiguous, {n_batches, n_tiles, 2 * values}, ScalarType::Float);
        states = torch::stable::new_empty(
            x_contiguous, {n_batches, n_tiles, values}, ScalarType::Float);
        launch(k.reduce[varying],
               {decays_contiguous.get(), zi_contiguous.get(),
                x_contiguous.get(), totals.get()},
               {steps, stride_b, n_tiles}, tile_grid);
        launch(k.scan, {totals.get(), states.get()}, {n_tiles},
               {THREADS, static_cast<uint64_t>(n_batches)});
    }
    launch(k.replay[varying],
           {decays_contiguous.get(), zi_contiguous.get(), x_contiguous.get(),
            states.get(), output.get()},
           {steps, stride_b, n_tiles}, tile_grid);
    return output;
}

Tensor lti_recur_mps(Tensor a, Tensor zi, Tensor x)
{
    STD_TORCH_CHECK(a.dim() <= 1, "A must be a vector or a scalar");
    STD_TORCH_CHECK(x.dim() == 2 && zi.dim() == 1 && zi.size(0) == x.size(0),
                    "x must be (B, T) and zi (B,)");
    STD_TORCH_CHECK(x.size(1) > 0, "x must contain at least one time step");
    const bool per_item = a.dim() == 1 && a.numel() == x.size(0);
    STD_TORCH_CHECK(per_item || a.numel() == 1, "A must be a scalar or (B,)");
    return scalar_recur_mps(a, false, per_item ? 1 : 0, zi, x);
}

// h[t] = decay[t] h[t - 1] + impulse[t]: one coefficient per step.
Tensor scan_mps(Tensor impulse, Tensor decay, Tensor init)
{
    STD_TORCH_CHECK(impulse.dim() == 2 && decay.dim() == 2 &&
                        decay.size(0) == impulse.size(0) &&
                        decay.size(1) == impulse.size(1) && init.dim() == 1 &&
                        init.size(0) == impulse.size(0),
                    "impulse and decay must be (B, T) and init (B,)");
    return scalar_recur_mps(decay, true, impulse.size(1), init, impulse);
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(philtorch, MPS, m)
{
    m.impl("lti_recur", TORCH_BOX(&lti_recur_mps));
    m.impl("scan", TORCH_BOX(&scan_mps));
}
