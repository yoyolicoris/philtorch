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
// The largest state size, half that for complex, as the CUDA kernels':
// a lane holds up to MAX_STATES / SIMD_WIDTH states of a vector, and a SIMD
// group a block of 8 columns of a map, in registers.
constexpr int64_t MAX_STATES = 128;
// The fewest states of vectors that run in blocks (see blocked).
constexpr int64_t BLOCK_STATES = 16;

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
            std::to_string(THREADS) + "\n#define CHUNK " +
            std::to_string(CHUNK) + "\n",
        RECUR_MPS_METAL);
    using philtorch::mps::kernel_function;
    const Kernels compiled{kernel_function(library, "recur_run"),
                           kernel_function(library, "recur_reduce")};
    return cache.emplace(key, compiled).first->second;
}

// The chunks per run of a sequence of n chunks, RUN at most, the runs as
// even as they can be, so that the last is padded by fewer chunks than
// there are runs.
int64_t run_length(int64_t n)
{
    return philtorch::ceil_div(n, philtorch::ceil_div(n, RUN));
}

// Whether vectors run in blocks of 8, by 8 x 8 matrix products: vectors of
// BLOCK_STATES or more that share their matrices, time-invariant or shared
// over the batch; smaller ones would leave the tiles mostly idle.
bool blocked(int64_t M, int64_t stride_t, int64_t a_batch)
{
    return M >= BLOCK_STATES && (stride_t == 0 || a_batch == 1);
}

// The SIMD groups of the vectors of n_batch sequences' chunks, as
// recur_mps.metal lays them out: blocks of 8 if blocked, a sequence's
// chunks or a chunk's sequences padded to a multiple of 8 where a block may
// not span two; and otherwise SIMD_WIDTH / LANES a group in runs (see
// run_length), LANES being M rounded up to a power of 2, SIMD_WIDTH at most.
int64_t vector_groups(int64_t batch, int64_t chunks, int64_t M,
                      int64_t stride_t, int64_t a_batch, bool block)
{
    if (block)
    {
        const int64_t inner = stride_t != 0 ? batch : chunks;
        const int64_t outer = stride_t != 0 ? chunks : batch;
        const int64_t padded = stride_t != 0 || a_batch > 1
            ? philtorch::ceil_div<int64_t>(inner, 8) * 8
            : inner;
        return philtorch::ceil_div<int64_t>(outer * padded, 8);
    }
    int64_t lanes = 1;
    while (lanes < std::min(M, SIMD_WIDTH))
        lanes *= 2;
    const int64_t run = run_length(chunks);
    return philtorch::ceil_div(batch * philtorch::ceil_div(chunks, run) * run,
                               SIMD_WIDTH / lanes);
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
    const bool block = blocked(M, stride_t, a_batch);
    const int64_t chunks = philtorch::ceil_div(steps, CHUNK);

    // The state at each chunk's end but the last, from the same recurrence
    // over the maps of those chunks, from a zero start, as chunk 0's offset
    // holds zi; a sequence of one chunk needs none.
    Tensor starts = out;
    if (chunks > 1)
    {
        // The maps, in the accumulation type: their offsets o, and their
        // matrices P, one per chunk of time-varying A, and one, A^CHUNK, for
        // time-invariant A, in SIMD groups of 8 x 8 tiles: ceil(M / 8) per
        // map, or 8 / M maps per group up to 4 states.
        const int64_t maps_per_item = chunks - 1;
        const int64_t maps_per_a = stride_t == 0 ? 1 : maps_per_item;
        const ScalarType acc = philtorch::mps::accumulation_type(x.scalar_type());
        auto offsets = torch::stable::new_empty(x, {batch, maps_per_item, M}, acc);
        auto maps = torch::stable::new_empty(x, {a_batch, maps_per_a, M, M}, acc);
        const int64_t packed = M <= 4 ? 8 / M : 1;
        const int64_t offset_groups =
            vector_groups(batch, maps_per_item, M, stride_t, a_batch, block);
        const int64_t map_groups =
            philtorch::ceil_div(a_batch * maps_per_a, packed) *
            philtorch::ceil_div<int64_t>(M, 8);
        launch(k.reduce, {a.get(), zi.get(), x.get(), offsets.get(), maps.get()},
               {batch, steps, stride_b, stride_t, a_batch, from_zi,
                run_length(maps_per_item), block, packed, offset_groups},
               threads(offset_groups + map_groups), group);
        starts = recur_mps(maps, a_batch, a_batch == 1 ? 0 : maps_per_a * M * M,
                           stride_t == 0 ? 0 : M * M, offsets, false, offsets);
    }
    launch(k.run, {a.get(), zi.get(), x.get(), starts.get(), out.get()},
           {batch, steps, stride_b, stride_t, a_batch, from_zi,
            run_length(chunks), block},
           threads(vector_groups(batch, chunks, M, stride_t, a_batch, block)),
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

// recur2's, recur3's, lti_recur2's and lti_recur3's MPS kernels: those of
// M states, taking the order of their name only, as their CUDA kernels and
// fakes do.
template <int64_t order>
Tensor recur_of_order(const Tensor &A, const Tensor &zi, const Tensor &x)
{
    STD_TORCH_CHECK(x.dim() == 3 && x.size(2) == order,
                    "x must be of shape (B, T, ", order, ")");
    return recurN_mps(A, zi, x);
}

template <int64_t order>
Tensor lti_recur_of_order(const Tensor &A, const Tensor &zi, const Tensor &x)
{
    STD_TORCH_CHECK(x.dim() == 3 && x.size(2) == order,
                    "x must be of shape (B, T, ", order, ")");
    return lti_recurN_mps(A, zi, x);
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(philtorch, MPS, m)
{
    m.impl("recurN", TORCH_BOX(&recurN_mps));
    m.impl("lti_recurN", TORCH_BOX(&lti_recurN_mps));
    m.impl("recur2", TORCH_BOX(&recur_of_order<2>));
    m.impl("lti_recur2", TORCH_BOX(&lti_recur_of_order<2>));
    m.impl("recur3", TORCH_BOX(&recur_of_order<3>));
    m.impl("lti_recur3", TORCH_BOX(&lti_recur_of_order<3>));
}
