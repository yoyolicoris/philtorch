// The recurrence of M states on MPS, time-invariant (lti_recurN) and
// time-varying (recurN), on torch's stable ABI and Metal C shim, as
// scalar_recur_mps.mm. The kernels, recur_mps.metal, are compiled once per
// state size and dtype, on first use.
#include "mps_shim.h"

#include <algorithm>
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
// The largest state size, half that for complex, as the Triton kernels':
// a lane holds up to MAX_STATES / SIMD_WIDTH states of a vector, and a SIMD
// group a block of 8 columns of a map, in registers.
constexpr int64_t MAX_STATES = 128;

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

// The SIMD groups of n vectors of M states, SIMD_WIDTH / LANES to a group,
// LANES in recur_mps.metal: M rounded up to a power of 2, SIMD_WIDTH at most.
int64_t vector_groups(int64_t n, int64_t M)
{
    int64_t lanes = 1;
    while (lanes < std::min(M, SIMD_WIDTH))
        lanes *= 2;
    return philtorch::ceil_div(n, SIMD_WIDTH / lanes);
}

// The threads of n SIMD groups, in whole threadgroups, which the kernels
// index in 32 bits.
std::array<uint64_t, 2> threads(int64_t simd_groups)
{
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
               threads(vector_groups(batch, M)), group);
        return out;
    }

    // The maps of every chunk but the last, in the accumulation type: their
    // offsets o, and their matrices P, one per chunk of time-varying A, and
    // one, A^chunk, for time-invariant A.
    const int64_t maps_per_a = stride_t == 0 ? 1 : maps_per_item;
    const ScalarType acc = philtorch::mps::accumulation_type(x.scalar_type());
    auto offsets = torch::stable::new_empty(x, {batch, maps_per_item, M}, acc);
    auto maps = torch::stable::new_empty(x, {a_batch, maps_per_a, M, M}, acc);
    // The maps' SIMD groups, each of 8 x 8 tiles: ceil(M / 8) per map, or
    // 8 / M maps per group up to 4 states.
    const int64_t packed = M <= 4 ? 8 / M : 1;
    const int64_t map_groups = philtorch::ceil_div(a_batch * maps_per_a, packed) *
                               philtorch::ceil_div<int64_t>(M, 8);
    launch(k.reduce, {a.get(), zi.get(), x.get(), offsets.get(), maps.get()},
           {batch, steps, CHUNK, stride_b, stride_t, from_zi, a_batch,
            run_length(maps_per_item), packed},
           threads(vector_groups(batch * run_vectors(maps_per_item), M) +
                   map_groups),
           group);

    // The state at each chunk's end, from the same recurrence over the
    // chunks' maps, from a zero start, as chunk 0's offset holds zi.
    const Tensor starts = recur_mps(
        maps, a_batch, a_batch == 1 ? 0 : maps_per_a * M * M,
        stride_t == 0 ? 0 : M * M, offsets, false, offsets);
    launch(k.run, {a.get(), zi.get(), x.get(), starts.get(), out.get()},
           {batch, steps, CHUNK, stride_b, stride_t, from_zi,
            run_length(maps_per_item + 1)},
           threads(vector_groups(batch * run_vectors(maps_per_item + 1), M)),
           group);
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
    const int64_t most =
        x.scalar_type() == ScalarType::ComplexFloat ? MAX_STATES / 2 : MAX_STATES;
    STD_TORCH_CHECK(x.size(2) <= most, "the MPS recurrence takes M up to ",
                    MAX_STATES, " (", MAX_STATES / 2, " complex), got ",
                    x.size(2));
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

// recur2's, lti_recur2's, recur3's and lti_recur3's MPS kernels: those of M
// states, taking the order of their name only, as their CUDA kernels and
// fakes do.
Tensor of_order(int64_t order,
                Tensor (*run)(const Tensor &, const Tensor &, const Tensor &),
                const Tensor &A, const Tensor &zi, const Tensor &x)
{
    STD_TORCH_CHECK(x.dim() == 3 && x.size(2) == order,
                    "x must be of shape (B, T, ", order, ")");
    return run(A, zi, x);
}

Tensor recur2_mps(const Tensor &A, const Tensor &zi, const Tensor &x)
{
    return of_order(2, recurN_mps, A, zi, x);
}

Tensor lti_recur2_mps(const Tensor &A, const Tensor &zi, const Tensor &x)
{
    return of_order(2, lti_recurN_mps, A, zi, x);
}

Tensor recur3_mps(const Tensor &A, const Tensor &zi, const Tensor &x)
{
    return of_order(3, recurN_mps, A, zi, x);
}

Tensor lti_recur3_mps(const Tensor &A, const Tensor &zi, const Tensor &x)
{
    return of_order(3, lti_recurN_mps, A, zi, x);
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(philtorch, MPS, m)
{
    m.impl("recurN", TORCH_BOX(&recurN_mps));
    m.impl("lti_recurN", TORCH_BOX(&lti_recurN_mps));
    m.impl("recur2", TORCH_BOX(&recur2_mps));
    m.impl("lti_recur2", TORCH_BOX(&lti_recur2_mps));
    m.impl("recur3", TORCH_BOX(&recur3_mps));
    m.impl("lti_recur3", TORCH_BOX(&lti_recur3_mps));
}
