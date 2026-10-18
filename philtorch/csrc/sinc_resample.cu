#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <limits>

// philtorch::resample(x, table, q, out_len): resampling with a polyphase table
// of p = table.size(0) phases and taps = table.size(1) = 2 K + 2,
//
//   out[b, f p + i] = sum_k table[i, k] x[b, q f + (i q) / p - K + k],
//
// with x zero outside [0, T), accumulated in at::opmath_type.  CUDA only.
//
// tile_kernel: a group of kGroupLanes lanes computes R outputs for BB rows,
// loading each x value once for the R outputs and each weight once for the BB
// rows, and sums the R BB accumulators by a butterfly reduction.  The zero
// padding of the table rows lets every lane run the same loop.  There is no
// host-to-device copy or sync, so calls can be captured in CUDA graphs.

namespace {

// The largest p, q and taps (Python bounds p (q + taps) by 2^22): padded table
// rows, under 16 kMaxTable, and the tile offsets then fit in int.
constexpr int64_t kMaxTable = int64_t(1) << 22;

constexpr int kWarp = 32;  // the shuffle masks assume 32 lanes
constexpr int kWarpsPerBlock = 4;
constexpr int kBlock = kWarp * kWarpsPerBlock;
// Lanes per tile: fewer cut the reduction's shuffles but widen each load.
constexpr int kGroupLaneLevels = 3;
constexpr int kGroupLanes = 1 << kGroupLaneLevels;  // 8
// Accumulators (R BB) per lane: a load per 2.7 multiply-adds, ~128 registers.
constexpr int kTileAcc = 32;
// Rows per tile, at most: more rows share the weights, fewer outputs share x.
constexpr int kMaxTileRows = 8;
// Outputs per tile, at most: longer tiles waste lanes and registers.
constexpr int kMaxTileOutputs = 8;

template <typename scalar_t>
struct TileArgs {
    using acc_t = at::opmath_type<scalar_t>;
    const scalar_t *x;         // (B, T), contiguous
    scalar_t *out;             // (B, out_len)
    int64_t B, T, out_len;
    int64_t p, q, taps;
    int64_t ts, pad;           // table row stride, zero padding on either side
    const acc_t *table;        // (p, ts)
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
// m0 .. m0 + R - 1 for BB rows; block (output block, row block) of a grid of
// out_blocks output blocks per row block.
template <typename scalar_t, int R, int BB>
__global__ void __launch_bounds__(kBlock)
    tile_kernel(const TileArgs<scalar_t> a, unsigned int out_blocks) {
    using acc_t = at::opmath_type<scalar_t>;
    constexpr int G = kGroupLanes;
    constexpr int A = R * BB;
    constexpr int TPW = kWarp / G;  // tiles per warp
    // Every lane ends with A / G totals to write.
    static_assert(A >= G && (A & (A - 1)) == 0);
    const int lane = threadIdx.x & (kWarp - 1);
    const int lg = lane & (G - 1);
    const unsigned int block_out = blockIdx.x % out_blocks, block_row = blockIdx.x / out_blocks;
    const int64_t tile0 =
        (static_cast<int64_t>(block_out) * kWarpsPerBlock + threadIdx.x / kWarp) * TPW;
    if (tile0 * R >= a.out_len) {
        return;  // whole warp
    }
    // The tiles of a warp past out_len still take part in the reduction.
    const int64_t m0 = (tile0 + lane / G) * R;
    const int64_t m0c = min(m0, a.out_len - 1);
    const int64_t b0 = static_cast<int64_t>(block_row) * BB;

    // The phases of the tile's outputs (the last one repeated past out_len).
    const int64_t f0 = m0c / a.p;
    int i = static_cast<int>(m0c - f0 * a.p);
    const int p = static_cast<int>(a.p);
    // The first input of phase i in frame 0 is (i q) / p - K, for taps = 2 K + 2.
    const int64_t K = (a.taps - 2) / 2;
    const int64_t first_i0 = static_cast<int64_t>(i) * a.q / a.p;
    const int64_t nlo0 = a.q * f0 + first_i0 - K;
    int d[R];
    const acc_t *wp[R];  // wp[j][u] is the weight of output j at input nlo0 + u
    {
        int64_t df = 0;
#pragma unroll
        for (int j = 0; j < R; ++j) {
            if (j > 0 && m0c + j < a.out_len) {
                if (++i == p) {
                    i = 0;
                    ++df;
                }
            }
            // The input of output m0c + j is floor(q (m0c + j) / p) - K, so d_j
            // is the difference of two floors, in [0, ceil((R - 1) q / p)]:
            // within the padding.
            const int64_t dj = df * a.q + static_cast<int64_t>(i) * a.q / a.p - first_i0;
            d[j] = static_cast<int>(dj);
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

// Launches tile_kernel on a 1-D grid of output blocks by row blocks, whose x
// extent holds 2^31 - 1 blocks; a block covers BB rows and R outputs per tile.
// table: (p, taps), in at::opmath_type<scalar_t>.
template <typename scalar_t, int BB>
void launch_tile(const at::Tensor &x, const at::Tensor &table, at::Tensor &out,
                 int64_t q, cudaStream_t stream) {
    using acc_t = at::opmath_type<scalar_t>;
    // The outputs that make kTileAcc accumulators, at most kMaxTileOutputs.
    constexpr int R = std::min(kTileAcc / BB, kMaxTileOutputs);
    constexpr int tiles_per_block = kWarpsPerBlock * (kWarp / kGroupLanes);
    const int64_t B = x.size(0), T = x.size(1), out_len = out.size(1);
    const int64_t p = table.size(0), taps = table.size(1);
    // The zero padding of a table row on either side: the first inputs of two
    // outputs of a tile are at most ceil((R - 1) q / p) apart.
    const int64_t pad = at::ceil_div<int64_t>((R - 1) * q, p);
    const at::Tensor padded = at::constant_pad_nd(table, {pad, pad}).contiguous();
    const TileArgs<scalar_t> a{x.const_data_ptr<scalar_t>(),
                               out.mutable_data_ptr<scalar_t>(),
                               B, T, out_len, p, q, taps, taps + 2 * pad, pad,
                               padded.const_data_ptr<acc_t>()};

    const int64_t out_blocks = at::ceil_div<int64_t>(out_len, int64_t{tiles_per_block} * R);
    const int64_t blocks = out_blocks * at::ceil_div<int64_t>(B, BB);
    TORCH_CHECK(blocks <= std::numeric_limits<int>::max(),
                "resample: the output exceeds the CUDA grid limit");
    tile_kernel<scalar_t, R, BB>
        <<<static_cast<unsigned int>(blocks), kBlock, 0, stream>>>(
            a, static_cast<unsigned int>(out_blocks));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// table: (p, taps), in at::opmath_type<scalar_t>.
template <typename scalar_t>
at::Tensor resample_run(const at::Tensor &x, const at::Tensor &table,
                        int64_t q, int64_t out_len) {
    const auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const at::Tensor xc = x.contiguous();
    at::Tensor out = at::empty({x.size(0), out_len}, x.options());
    // The tiles' batch rows: the largest power of two up to B and kMaxTileRows.
    int BB = 1;
    while (BB < kMaxTileRows && 2 * BB <= x.size(0)) {
        BB *= 2;
    }
    switch (BB) {
        case 8: launch_tile<scalar_t, 8>(xc, table, out, q, stream); break;
        case 4: launch_tile<scalar_t, 4>(xc, table, out, q, stream); break;
        case 2: launch_tile<scalar_t, 2>(xc, table, out, q, stream); break;
        default: launch_tile<scalar_t, 1>(xc, table, out, q, stream); break;
    }
    return out;
}

}  // namespace

at::Tensor sinc_resample_cuda_impl(const at::Tensor &x, const at::Tensor &table,
                                   int64_t q, int64_t out_len) {
    TORCH_CHECK(x.dim() == 2 && x.is_floating_point(),
                "resample: x must be a 2-D floating tensor");
    TORCH_CHECK(table.dim() == 2 && table.is_floating_point(),
                "resample: table must be a 2-D floating tensor");
    const int64_t p = table.size(0), taps = table.size(1);
    TORCH_CHECK(taps >= 2 && taps % 2 == 0,
                "resample: the number of taps must be even and at least 2, got ",
                taps);
    TORCH_CHECK(x.is_cuda() && table.device() == x.device(),
                "resample: x and table must be on the same CUDA device");
    TORCH_CHECK(q >= 1 && out_len >= 0,
                "resample: q must be at least 1 and out_len at least 0");
    TORCH_CHECK(p >= 1 && p <= kMaxTable && q <= kMaxTable &&
                    taps <= kMaxTable,
                "resample: the number of phases (", p, "), taps (", taps,
                ") and q (", q, ") must be at most ", kMaxTable);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    if (x.numel() == 0 || out_len == 0) {
        return at::zeros({x.size(0), out_len}, x.options());
    }
    at::Tensor out;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "resample_cuda", [&] {
            const at::Tensor tab =
                table.to(at::toOpMathType(x.scalar_type()));
            out = resample_run<scalar_t>(x, tab, q, out_len);
        });
    return out;
}

TORCH_LIBRARY_FRAGMENT(philtorch, m) {
    m.def(
        "resample(Tensor x, Tensor table, int q, int out_len) -> Tensor",
        {at::Tag::pt2_compliant_tag});
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m) {
    m.impl("resample", &sinc_resample_cuda_impl);
}
