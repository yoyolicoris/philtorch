// The recurrence of M states on MPS, time-invariant (lti_recurN) and
// time-varying (recurN), on torch's stable ABI and Metal C shim, as
// scalar_recur_mps.mm. The kernels, recur_mps.metal, are compiled once per
// state size and dtype, on first use.
#include "mps_shim.h"

#include <array>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <utility>

// RECUR_MPS_METAL, the text of recur_mps.metal, which setup.py generates in
// the build directory.
#include "recur_mps.metal.h"

namespace
{

using philtorch::mps::launch;
using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// The kernels' name in errors.
constexpr const char *OP = "the MPS recurrence";

// Threads per threadgroup of the chunks' kernels, and per SIMD group.
constexpr uint64_t THREADS = 64;
constexpr int64_t SIMD_WIDTH = 32;
// Steps per chunk, and at most as many consecutive chunks of a sequence per
// run (see sequence_chunk in recur_mps.metal).
constexpr int64_t CHUNK = 64;
constexpr int64_t RUN = 32;
// The largest state size: a vector of states runs on a SIMD group's lanes.
constexpr int64_t MAX_STATES = SIMD_WIDTH;

struct Kernels
{
    AOTIMetalKernelFunctionHandle run, reduce;
};

// The kernels of M states in dtype, compiled on the first call that needs them.
const Kernels &kernels(int64_t M, ScalarType dtype)
{
    static std::mutex mutex;
    static std::map<std::pair<int64_t, ScalarType>, Kernels> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(M, dtype);
    const auto found = cache.find(key);
    if (found != cache.end())
        return found->second;
    const AOTIMetalShaderLibraryHandle library = philtorch::mps::compile(
        "#define STATES " + std::to_string(M) + "\n#define SCALAR " +
            philtorch::mps::metal_type(dtype, OP) + "\n#define THREADS " +
            std::to_string(THREADS) + "\n",
        RECUR_MPS_METAL);
    using philtorch::mps::kernel_function;
    const Kernels compiled{kernel_function(library, "recur_run"),
                           kernel_function(library, "recur_reduce")};
    return cache.emplace(key, compiled).first->second;
}

// The lanes of a vector of M states, LANES in recur_mps.metal: M rounded up
// to a power of 2.
int64_t lanes(int64_t M)
{
    int64_t p = 1;
    while (p < M)
        p *= 2;
    return p;
}

// The threads of n vectors of M states, SIMD_WIDTH / lanes(M) to a SIMD
// group, in whole threadgroups, which the kernels index in 32 bits.
std::array<uint64_t, 2> threads(int64_t n, int64_t M)
{
    const int64_t simd_groups = philtorch::ceil_div(n, SIMD_WIDTH / lanes(M));
    const int64_t groups =
        philtorch::ceil_div<int64_t>(simd_groups * SIMD_WIDTH, THREADS);
    STD_TORCH_CHECK(groups <= std::numeric_limits<uint32_t>::max() / THREADS,
                    "the input is too large for ", OP);
    return {static_cast<uint64_t>(groups) * THREADS, 1};
}

// The chunks per run of a sequence of n chunks, RUN at most, the runs as
// even as they can be, so that the last is padded by fewer chunks than
// there are runs.
int64_t run_length(int64_t n)
{
    return philtorch::ceil_div(n, philtorch::ceil_div(n, RUN));
}

// The vectors of each sequence's runs of n chunks, padding included.
int64_t run_vectors(int64_t n)
{
    return philtorch::ceil_div(n, run_length(n)) * run_length(n);
}

// h[b, t] = A[b, t] h[b, t - 1] + x[b, t], with A[b, t] the M x M matrix at
// a + b * stride_b + t * stride_t, a_batch of them, one per sequence or
// shared (stride_b = 0), and time-invariant if stride_t = 0, from h[b, -1] =
// zi[b] if from_zi, and zero otherwise. The inputs are contiguous.
Tensor recur_mps(const Tensor &a, int64_t a_batch, int64_t stride_b,
                 int64_t stride_t, const Tensor &zi, bool from_zi,
                 const Tensor &x)
{
    auto out = torch::stable::empty_like(x);
    const int64_t batch = x.size(0), steps = x.size(1), M = x.size(2);
    if (out.numel() == 0)
        return out;
    const Kernels &k = kernels(M, x.scalar_type());
    const std::array<uint64_t, 2> group{THREADS, 1};

    // A sequence of one chunk needs no state from before it.
    const int64_t maps_per_item = (steps - 1) / CHUNK;
    if (maps_per_item == 0)
    {
        launch(k.run, {a.get(), zi.get(), x.get(), out.get(), out.get()},
               {batch, steps, CHUNK, stride_b, stride_t, from_zi, 1},
               threads(batch, M), group);
        return out;
    }

    // The maps of every chunk but the last, in the accumulation type: their
    // offsets o, and their matrices P, one per chunk of time-varying A, and
    // one, A^chunk, for time-invariant A.
    const int64_t maps_per_a = stride_t == 0 ? 1 : maps_per_item;
    const ScalarType acc = philtorch::mps::accumulation_type(x.scalar_type());
    auto offsets = torch::stable::new_empty(x, {batch, maps_per_item, M}, acc);
    auto maps = torch::stable::new_empty(x, {a_batch, maps_per_a, M, M}, acc);
    launch(k.reduce, {a.get(), zi.get(), x.get(), offsets.get(), maps.get()},
           {batch, steps, CHUNK, stride_b, stride_t, from_zi, a_batch,
            run_length(maps_per_item)},
           threads(batch * run_vectors(maps_per_item) + a_batch * maps_per_a * M,
                   M),
           group);

    // The state at each chunk's end, from the same recurrence over the
    // chunks' maps, from a zero start, as chunk 0's offset holds zi.
    const Tensor starts = recur_mps(
        maps, a_batch, a_batch == 1 ? 0 : maps_per_a * M * M,
        stride_t == 0 ? 0 : M * M, offsets, false, offsets);
    launch(k.run, {a.get(), zi.get(), x.get(), starts.get(), out.get()},
           {batch, steps, CHUNK, stride_b, stride_t, from_zi,
            run_length(maps_per_item + 1)},
           threads(batch * run_vectors(maps_per_item + 1), M), group);
    return out;
}

void check_inputs(const Tensor &A, const Tensor &zi, const Tensor &x)
{
    using philtorch::mps::is_mps;
    STD_TORCH_CHECK(is_mps(A) && is_mps(zi) && is_mps(x),
                    "the coefficients, initial states and inputs must be MPS tensors");
    STD_TORCH_CHECK(A.scalar_type() == x.scalar_type() &&
                        zi.scalar_type() == x.scalar_type(),
                    "the coefficients, initial states and inputs must have the same dtype");
    philtorch::mps::metal_type(x.scalar_type(), OP);
    STD_TORCH_CHECK(x.dim() == 3 && zi.dim() == 2 && zi.size(0) == x.size(0) &&
                        zi.size(1) == x.size(2),
                    "x must be (B, T, M) and zi (B, M)");
    STD_TORCH_CHECK(A.size(-1) == x.size(2) && A.size(-2) == x.size(2),
                    "the last two dimensions of A must be M, x's last");
    STD_TORCH_CHECK(x.size(2) <= MAX_STATES, "the MPS recurrence takes M up to ",
                    MAX_STATES, ", got ", x.size(2));
    // The kernels index the steps of a sequence, the last chunk's past its
    // end included, in 32 bits, and check their grids' size as they launch.
    STD_TORCH_CHECK(x.size(1) <= std::numeric_limits<int32_t>::max(),
                    "the input is too large for ", OP);
}

Tensor recurN_mps(const Tensor &A, const Tensor &zi, const Tensor &x)
{
    STD_TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    check_inputs(A, zi, x);
    STD_TORCH_CHECK(A.size(-3) == x.size(1), "A must have one matrix per step of x");
    STD_TORCH_CHECK(A.dim() == 3 || A.size(0) == x.size(0), "A must have x's batch size");
    const int64_t M = x.size(2);
    return recur_mps(torch::stable::contiguous(A), A.dim() == 4 ? x.size(0) : 1,
                     A.dim() == 4 ? x.size(1) * M * M : 0, M * M,
                     torch::stable::contiguous(zi), true, torch::stable::contiguous(x));
}

Tensor lti_recurN_mps(const Tensor &A, const Tensor &zi, const Tensor &x)
{
    STD_TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be a 2D or 3D tensor");
    check_inputs(A, zi, x);
    STD_TORCH_CHECK(A.dim() == 2 || A.size(0) == x.size(0), "A must have x's batch size");
    const int64_t M = x.size(2);
    return recur_mps(torch::stable::contiguous(A), A.dim() == 3 ? x.size(0) : 1,
                     A.dim() == 3 ? M * M : 0, 0, torch::stable::contiguous(zi), true,
                     torch::stable::contiguous(x));
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(philtorch, MPS, m)
{
    m.impl("recurN", TORCH_BOX(&recurN_mps));
    m.impl("lti_recurN", TORCH_BOX(&lti_recurN_mps));
}
