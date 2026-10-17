#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <bit>
#include <cstdint>

// philtorch::resample(x, table, first, table_t, first_t, q, out_len): the
// CUDA kernel of band-limited resampling by a rational factor, applying a
// precomputed polyphase table.  With p = table.size(0) phases and
// taps = table.size(1),
//
//   out[b, f p + i] = sum_{k < taps} table[i, k] x[b, q f + first[i] + k],
//
// for 0 <= f p + i < out_len, and x[b, n] = 0 outside [0, T).  `first` (p,
// int64) holds the first input of each phase in frame 0 and may be negative.
// table_t and first_t, the table and offsets of the transposed operator, are
// not read: they are inputs so that the backward pass, which Python
// registers, can call this op again with them.  There is no CPU kernel.
//
// The table is real and the weights are accumulated in at::opmath_type of x's
// dtype (double for float64, float for float32, float16 and bfloat16).
//
// tile_kernel is a register tiling: a group of kGroupLanes lanes computes R
// consecutive outputs for BB batch rows.  The lanes stride over the inputs the
// tile reads, each x value is loaded once for the R outputs that read it and
// each weight once for the BB rows, and a butterfly reduction over the group
// totals the A = R BB accumulators.  The zero padding of the table rows lets
// every lane run the same loop whichever outputs read an input.
//
// The padded table and the output come from ATen ops and the kernel arguments
// are plain values: no host-to-device copy or synchronization, so a call can be
// captured in a CUDA graph.

namespace {

// The largest p, q and taps.  It keeps the int arithmetic of tile_kernel (the
// offsets within a tile, p, q, taps) from overflowing, whatever T is: the zero
// padding of a table row is at most (R - 1) q / p + 1 <= 7 kMaxTable + 1
// entries on either side, so the padded row length, the offsets d_j and the
// range of u all fit in 32 bits.
constexpr int64_t kMaxTable = int64_t(1) << 20;

constexpr int kWarp = C10_WARP_SIZE;
// The shuffle code (full masks, groups of lanes that divide a warp) is written
// for 32.
static_assert(kWarp == 32);
constexpr int kWarpsPerBlock = 4;
constexpr int kBlock = kWarp * kWarpsPerBlock;  // 128 threads, any architecture
// Lanes per group (a tile): the reduction costs A - A / kGroupLanes shuffles
// for A accumulators, and a load of one group covers kGroupLanes consecutive
// elements (a 32-byte sector in float), so fewer lanes cut the reduction at
// the price of wider loads, which are then split among the groups of a warp.
constexpr int kGroupLanes = 8;
constexpr int kGroupLaneLevels = std::bit_width(static_cast<unsigned>(kGroupLanes)) - 1;  // log2
// Accumulators (R BB) of a lane: with R + BB loads per R BB multiply-adds, 32
// gives a load per 2.7 multiply-adds at a register count (about 128) that
// still lets several blocks share a multiprocessor.
constexpr int kTileAcc = 32;
// Batch rows per lane: more rows share the weight loads, but a tile of
// kTileAcc accumulators then has fewer outputs to share the x loads.
constexpr int kMaxTileRows = 8;
// Outputs per tile, at most: the u range of a tile exceeds the taps by the span
// of its outputs, so long tiles of few rows spend lanes on inputs that few
// outputs read and need many registers for their pointers.
constexpr int kMaxTileOutputs = 8;

// The largest power of two that is at most max and at most B >= 1.
inline int pow2_rows(int64_t B, int max) {
    return static_cast<int>(
        std::min<int64_t>(std::bit_floor(static_cast<uint64_t>(B)), max));
}

// Calls launch(chunk, grid) for the chunks of the batch that fit the y extent
// of a grid, a chunk being the kernel arguments (fields x, out, B, out_len, T)
// for its rows.  A block covers rows_per_block rows and outputs_per_block
// outputs.
template <typename Args, typename Launch>
void launch_chunked(const Args &a, int64_t outputs_per_block,
                    int rows_per_block, Launch &&launch) {
    const auto *prop = at::cuda::getCurrentDeviceProperties();
    const int64_t grid_x = at::ceil_div<int64_t>(a.out_len, outputs_per_block);
    TORCH_CHECK(grid_x <= prop->maxGridSize[0], "out_len ", a.out_len,
                " exceeds the CUDA grid dimension limit");
    const int64_t rows = int64_t{prop->maxGridSize[1]} * rows_per_block;
    for (int64_t b_off = 0; b_off < a.B; b_off += rows) {
        Args chunk = a;
        chunk.B = std::min<int64_t>(a.B - b_off, rows);
        chunk.x = a.x + b_off * a.T;
        chunk.out = a.out + b_off * a.out_len;
        launch(chunk, dim3(static_cast<unsigned int>(grid_x),
                           static_cast<unsigned int>(
                               at::ceil_div<int64_t>(chunk.B, rows_per_block))));
    }
}

template <typename scalar_t>
struct TileArgs {
    using acc_t = at::opmath_type<scalar_t>;
    const scalar_t *x;         // (B, T), contiguous, B of this launch
    scalar_t *out;             // (B, out_len)
    int64_t B, T, out_len;
    int64_t p, q, taps;
    int64_t ts, pad;           // table row stride, zero padding on either side
    const acc_t *table;        // (p, ts)
    const int64_t *first;      // (p)
};

// Butterfly reduction over the lanes of a group.  A accumulators per lane,
// LV levels: at each level a lane swaps the half of its array it does not keep
// with its partner (lane ^ off) and adds what it gets, so that after LV levels
// it holds the totals of A >> LV accumulators, from index idx.
template <int H, int OFF, int LV, typename AccT, int A>
__device__ __forceinline__ void scatter_levels(AccT (&a)[A], int lg, int &idx) {
    if constexpr (LV > 0) {
        const bool up = (lg & OFF) != 0;
#pragma unroll
        for (int i = 0; i < H; ++i) {
            const AccT send = up ? a[i] : a[i + H];
            const AccT keep = up ? a[i + H] : a[i];
            a[i] = keep + WARP_SHFL_XOR(send, OFF, kWarp);
        }
        idx += up ? H : 0;
        scatter_levels<H / 2, OFF / 2, LV - 1>(a, lg, idx);
    }
}

// A group of kGroupLanes lanes computes the tile of R consecutive outputs
// m0 .. m0 + R - 1 for the BB rows of blockIdx.y.
template <typename scalar_t, int R, int BB>
__global__ void __launch_bounds__(kBlock) tile_kernel(const TileArgs<scalar_t> a) {
    using acc_t = at::opmath_type<scalar_t>;
    constexpr int G = kGroupLanes;
    constexpr int A = R * BB;
    constexpr int TPW = kWarp / G;  // tiles per warp
    // Every lane ends with A / G totals to write.
    static_assert(A >= G && (A & (A - 1)) == 0);
    const int lane = threadIdx.x & (kWarp - 1);
    const int lg = lane & (G - 1);
    const int64_t tile0 =
        (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp) * TPW;
    if (tile0 * R >= a.out_len) {
        return;  // whole warp
    }
    // The tiles of a warp past out_len still take part in the reduction.
    const int64_t m0 = (tile0 + lane / G) * R;
    const int64_t m0c = min(m0, a.out_len - 1);
    const int64_t b0 = static_cast<int64_t>(blockIdx.y) * BB;

    // The phases of the tile's outputs (the last one repeated past out_len).
    const int64_t f0 = m0c / a.p;
    int i = static_cast<int>(m0c - f0 * a.p);
    const int p = static_cast<int>(a.p);
    const int64_t nlo_i0 = a.first[i];
    const int64_t nlo0 = a.q * f0 + nlo_i0;
    int d[R];
    const acc_t *wp[R];  // wp[j][u] is the weight of output j at input nlo0 + u
    {
        const int q = static_cast<int>(a.q);
        int df = 0;
#pragma unroll
        for (int j = 0; j < R; ++j) {
            if (j > 0 && m0c + j < a.out_len) {
                if (++i == p) {
                    i = 0;
                    ++df;
                }
            }
            d[j] = df * q + static_cast<int>(a.first[i] - nlo_i0);
            wp[j] = a.table + static_cast<int64_t>(i) * a.ts + a.pad - d[j];
        }
    }
    const int dlast = d[R - 1];
    const int taps = static_cast<int>(a.taps);
    // The u range, with inputs in [0, T), that any output of the tile reads.
    const int64_t ulo64 = max(int64_t{0}, -nlo0);
    const int64_t uhi64 = min(static_cast<int64_t>(dlast) + taps, a.T - nlo0);
    acc_t acc[A];
#pragma unroll
    for (int e = 0; e < A; ++e) {
        acc[e] = acc_t(0);
    }
    if (ulo64 < uhi64) {
        const int ulo = static_cast<int>(ulo64), uhi = static_cast<int>(uhi64);
        const scalar_t *xp[BB];
#pragma unroll
        for (int bb = 0; bb < BB; ++bb) {
            xp[bb] = a.x + min(b0 + bb, a.B - 1) * a.T + nlo0;
        }
        // Unrolled four times, so that the loads of the next iterations are
        // issued while the multiply-adds of this one run (measured: 1 to 3% in
        // float).
#pragma unroll 4
        for (int u = ulo + lg; u < uhi; u += G) {
            acc_t xv[BB];
#pragma unroll
            for (int bb = 0; bb < BB; ++bb) {
                xv[bb] = static_cast<acc_t>(xp[bb][u]);
            }
#pragma unroll
            for (int j = 0; j < R; ++j) {
                const acc_t w = wp[j][u];
#pragma unroll
                for (int bb = 0; bb < BB; ++bb) {
                    acc[j * BB + bb] = fma(w, xv[bb], acc[j * BB + bb]);
                }
            }
        }
    }

    // After the scatter over the log2 G levels each lane holds the totals of
    // A / G accumulators, from index idx.
    int idx = 0;
    scatter_levels<(A >> 1), G / 2, kGroupLaneLevels>(acc, lg, idx);
#pragma unroll
    for (int e = 0; e < A / G; ++e) {
        const int ai = idx + e;
        const int j = ai / BB, bb = ai - j * BB;
        const int64_t m = m0 + j, b = b0 + bb;
        if (m < a.out_len && b < a.B) {
            a.out[b * a.out_len + m] = static_cast<scalar_t>(acc[e]);
        }
    }
}

// The shape of a tile for B rows: BB rows (pow2_rows), and the R outputs that
// make kTileAcc accumulators, at most kMaxTileOutputs.
constexpr int tile_outputs(int bb) {
    return std::min(kTileAcc / bb, kMaxTileOutputs);
}

template <typename scalar_t, int BB>
void launch_tile(const TileArgs<scalar_t> &a, cudaStream_t stream) {
    constexpr int R = tile_outputs(BB);
    constexpr int tiles_per_block = kWarpsPerBlock * (kWarp / kGroupLanes);
    launch_chunked(a, int64_t{tiles_per_block} * R, BB,
                   [&](const TileArgs<scalar_t> &chunk, dim3 grid) {
                       tile_kernel<scalar_t, R, BB><<<grid, kBlock, 0, stream>>>(chunk);
                       C10_CUDA_KERNEL_LAUNCH_CHECK();
                   });
}

template <typename scalar_t>
void launch_tile_rows(const TileArgs<scalar_t> &a, int BB, cudaStream_t stream) {
    switch (BB) {
        case 8: launch_tile<scalar_t, 8>(a, stream); break;
        case 4: launch_tile<scalar_t, 4>(a, stream); break;
        case 2: launch_tile<scalar_t, 2>(a, stream); break;
        default: launch_tile<scalar_t, 1>(a, stream); break;
    }
}

// table: (p, taps), in at::opmath_type<scalar_t>.
template <typename scalar_t>
at::Tensor resample_run(const at::Tensor &x, const at::Tensor &table,
                        const at::Tensor &first, int64_t q, int64_t out_len) {
    using acc_t = at::opmath_type<scalar_t>;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t B = x.size(0), T = x.size(1);
    const int64_t p = table.size(0), taps = table.size(1);

    // The tiles' batch rows BB and outputs R, and the zero padding pad of a
    // table row on either side: the first inputs of two outputs of a tile are
    // at most ceil((R - 1) q / p) apart, and the + 1 covers the rounding in
    // first.
    const int BB = pow2_rows(B, kMaxTileRows);
    const int R = tile_outputs(BB);
    const int64_t pad = at::ceil_div<int64_t>((R - 1) * q, p) + 1;
    const int64_t ts = taps + 2 * pad;

    const at::Tensor xc = x.contiguous();
    const at::Tensor padded =
        at::constant_pad_nd(table, {pad, pad}).contiguous();
    const at::Tensor fa = first.contiguous();
    at::Tensor out = at::empty({B, out_len}, x.options());

    TileArgs<scalar_t> a;
    a.x = xc.const_data_ptr<scalar_t>();
    a.out = out.mutable_data_ptr<scalar_t>();
    a.B = B;
    a.T = T;
    a.out_len = out_len;
    a.p = p;
    a.q = q;
    a.taps = taps;
    a.ts = ts;
    a.pad = pad;
    a.table = padded.const_data_ptr<acc_t>();
    a.first = fa.const_data_ptr<int64_t>();
    launch_tile_rows<scalar_t>(a, BB, stream);
    return out;
}

}  // namespace

at::Tensor sinc_resample_cuda_impl(const at::Tensor &x, const at::Tensor &table,
                                   const at::Tensor &first,
                                   const at::Tensor & /*table_t*/,
                                   const at::Tensor & /*first_t*/, int64_t q,
                                   int64_t out_len) {
    TORCH_CHECK(x.dim() == 2 && x.is_floating_point(),
                "resample: x must be a 2-D floating tensor");
    TORCH_CHECK(table.dim() == 2 && table.is_floating_point(),
                "resample: table must be a 2-D floating tensor");
    const int64_t p = table.size(0), taps = table.size(1);
    TORCH_CHECK(first.dim() == 1 && first.scalar_type() == at::kLong &&
                    first.size(0) == p,
                "resample: first must be an int64 tensor of length ", p);
    TORCH_CHECK(x.is_cuda() && table.device() == x.device() &&
                    first.device() == x.device(),
                "resample: x, table and first must be on the same CUDA "
                "device");
    TORCH_CHECK(q >= 1 && out_len >= 0,
                "resample: q must be at least 1 and out_len at least 0");
    TORCH_CHECK(p >= 1 && p <= kMaxTable && q <= kMaxTable &&
                    taps <= kMaxTable,
                "resample: the number of phases (", p, "), taps (", taps,
                ") and q (", q, ") must be at most ", kMaxTable);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    if (x.numel() == 0 || table.numel() == 0 || out_len == 0) {
        return at::zeros({x.size(0), out_len}, x.options());
    }
    at::Tensor out;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "resample_cuda", [&] {
            const at::Tensor tab =
                table.to(at::toOpMathType(x.scalar_type()));
            out = resample_run<scalar_t>(x, tab, first, q, out_len);
        });
    return out;
}

TORCH_LIBRARY_FRAGMENT(philtorch, m) {
    m.def(
        "resample(Tensor x, Tensor table, Tensor first, Tensor table_t, "
        "Tensor first_t, int q, int out_len) -> Tensor",
        {at::Tag::pt2_compliant_tag});
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m) {
    m.impl("resample", &sinc_resample_cuda_impl);
}
