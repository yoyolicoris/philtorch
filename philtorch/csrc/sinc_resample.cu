#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <bit>
#include <cstdint>

#include "sinc_resample.h"

// CUDA kernels of philtorch::resample (see sinc_resample.h for the definition,
// the time grid and the plan, and host_sinc_resample.cpp for the reference
// semantics, which these follow).
//
// Rates reduced to p / q: output p f + i reads the inputs q f + (i q / p) +
// (-S, S) and its weights depend on the phase i alone.  There are two paths.
//
// Table path (the plan's table: p <= out_len and p * taps <= kMaxTable).  Two
// kernels build the weights, in double, rounded to AccT once: prep_kernel the
// angle addition tables and the phases, and table_kernel the table itself, p
// rows of ts = taps + 2 pad entries, one thread per entry, padded with zeros.
// The main kernel (tile_kernel) is a register tiling.  Neighbouring outputs
// share most of their taps, so a group of kGroupLanes lanes computes a tile of
// R consecutive outputs for BB batch rows at once: the lanes stride over the
// inputs n = nlo_0 + u that the tile reads, and each lane loads x[b][n] once
// for the R outputs that read it and the weight of output j at n once for the
// BB rows, which makes R + BB loads for R BB multiply-adds (a warp per output
// would need one load per multiply-add).  The weights of output j at n are
// table[i_j][u - d_j] with d_j = nlo_j - nlo_0; the zero padding of the rows
// lets every lane run the same loop over u whichever outputs read n.  Every
// lane sums its inputs in R BB accumulators and a butterfly reduction over the
// group (A - A / kGroupLanes shuffles for A accumulators, where a plain tree
// would take A log2 kGroupLanes) gives each its total.  A warp holds
// 32 / kGroupLanes such groups, on neighbouring tiles.
//
// General path (every other case).  One group of lanes per output: the lanes
// stride over the taps n = nlo + k, each takes the weight of its tap once and
// accumulates BB batch rows with it in registers, and a reduction over the
// group sums every accumulator.  The setup of an output (its phase and
// sin/cos(pi c dc), in double) runs once per group, so a warp of narrow groups
// spends less on it per output than a warp per output: the group has
// kGroupLanes lanes, or kNarrowLanes when the taps are not more than that, so
// that no lane idles.  The weights are evaluated by the lanes, from sin/cos
// (pi c k), tabulated once per call, by angle addition (z_k = z_0 - c k),
// exactly as the CPU kernel does: that gives sin(pi z_k) to full AccT accuracy
// from a z_k that has only AccT accuracy (|z_k| is up to num_zeros, so float
// sincospif(z_k) would be off by about 1e-5), and spends two table loads and a
// few multiplies per tap instead of a sine.
//
// The window coefficients are made on the host and handed to the kernels by
// value, in a kernel argument of up to kMaxKaiserTerms entries (a __grid_constant__
// parameter, read through the constant cache, where all lanes of a warp read
// the same entry at once): no device allocation, copy or kernel is spent on
// them, and a call can be captured in a CUDA graph.
//
// Half and bfloat16 inputs are converted to float once, before the kernels, as
// on the CPU: the conversion is one cheap pass, it halves the number of kernel
// instantiations, and the results equal the CPU's.

// sin and cos of pi (hi + lo) in float: the reduced argument is split in two
// floats, and sincospif of the leading one is corrected to first order by the
// other.  (Double sincos is too slow on most GPUs to call once per output.)
namespace philtorch::detail {
template <>
struct SincosPi<float> {
    C10_HOST_DEVICE C10_ALWAYS_INLINE static void apply(double hi, double lo, float &s,
                                            float &c) {
#if defined(__CUDA_ARCH__)
        const double r = hi - 2.0 * std::rint(hi * 0.5) + lo;
        const float rh = static_cast<float>(r);
        const float prl = std::numbers::pi_v<float> *
                          static_cast<float>(r - static_cast<double>(rh));
        float sh, ch;
        sincospif(rh, &sh, &ch);
        s = fmaf(prl, ch, sh);
        c = fmaf(-prl, sh, ch);
#else
        double sd, cd;
        sincos_pi_double(hi, lo, sd, cd);
        s = static_cast<float>(sd);
        c = static_cast<float>(cd);
#endif
    }
};
}  // namespace philtorch::detail

namespace {

using philtorch::ResamplePlan;

constexpr int kWarp = C10_WARP_SIZE;
// The shuffle code (full masks, groups of lanes that divide a warp) is written for 32.
static_assert(kWarp == 32);
constexpr int kWarpsPerBlock = 4;
constexpr int kBlock = kWarp * kWarpsPerBlock;  // 128 threads, any architecture
// Lanes per group (a tile of the table path, an output of the general path): the
// reduction costs A - A / kGroupLanes shuffles for A accumulators, and a load of
// one group covers kGroupLanes consecutive elements (a 32-byte sector in float),
// so fewer lanes cut the reduction at the price of wider loads, which are then
// split among the groups of a warp.
constexpr int kGroupLanes = 8;
constexpr int kGroupLaneLevels = std::bit_width(static_cast<unsigned>(kGroupLanes)) - 1;  // log2
// Lanes per output of the general path when its taps are not more than that:
// a group of kGroupLanes lanes would idle some.
constexpr int kNarrowLanes = 4;
// Accumulators (R BB) of a lane of the table path: with R + BB loads per R BB
// multiply-adds, 32 gives a load per 2.7 multiply-adds at a register count
// (about 128) that still lets several blocks share a multiprocessor.
constexpr int kTileAcc = 32;
// Batch rows per lane in the table path: more rows share the weight loads, but
// a tile of kTileAcc accumulators then has fewer outputs to share the x loads.
constexpr int kMaxTileRows = 8;
// Outputs per tile, at most: the u range of a tile exceeds the taps by the span
// of its outputs, so long tiles of few rows spend lanes on inputs that few
// outputs read and need many registers for their pointers.
constexpr int kMaxTileOutputs = 8;
// Batch rows per lane in the general path: its lanes hold one accumulator and
// one pointer per row and nothing else of the tile, so it takes more rows.
constexpr int kMaxGeneralRows = 16;

// The largest power of two that is at most max and at most B >= 1.
inline int pow2_rows(int64_t B, int max) {
    return static_cast<int>(
        std::min<int64_t>(std::bit_floor(static_cast<uint64_t>(B)), max));
}

// Calls launch(args, grid) for the chunks of the batch of at most gridDim.y
// blocks along y (the batch rows of a block: rows_per_block), with `args` the
// kernel arguments for that chunk, which has fields x, out (B, out_len), B, T
// and out_len.  Outputs per block: outputs_per_block, along x.
template <typename Args, typename Launch>
void launch_chunked(const Args &a, int64_t outputs_per_block,
                    int rows_per_block, Launch &&launch) {
    const auto *prop = at::cuda::getCurrentDeviceProperties();
    const int64_t grid_x = at::ceil_div<int64_t>(a.out_len, outputs_per_block);
    TORCH_CHECK_VALUE(grid_x <= prop->maxGridSize[0], "out_len ", a.out_len,
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

// The Kaiser series coefficients (philtorch::make_kaiser_coeffs) as a kernel
// argument.
template <typename T>
struct Coeffs {
    T v[philtorch::kMaxKaiserTerms];
    int terms;
};

template <typename T>
Coeffs<T> kaiser_args(double beta, double scale) {
    const auto coef = philtorch::make_kaiser_coeffs<T>(beta, scale);
    Coeffs<T> c;
    std::copy(coef.begin(), coef.end(), c.v);
    c.terms = static_cast<int>(coef.size());
    return c;
}

// The phases of the table path: for i < p, (nlo_i, dc_i) = output_phase(i) and
// sin / cos (pi c dc_i).  p = 0 for none.
struct PhaseOut {
    int64_t p, q;
    double S;
    int64_t *nlo;
    double *s0, *c0, *dc;
};

// sin / cos (pi c k) as T for k < taps (the tables of the angle addition), and
// the phases.  One thread per k and per i.
template <typename T>
__global__ void prep_kernel(T *__restrict__ sk, T *__restrict__ ck, int64_t taps,
                            double c, PhaseOut ph) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t < taps) {
        philtorch::sincos_pi_prod(c, static_cast<double>(t), sk[t], ck[t]);
    }
    if (t < ph.p) {
        const auto out = philtorch::output_phase(t, ph.p, ph.q, ph.S);
        ph.nlo[t] = out.nlo;
        ph.dc[t] = out.dc;
        philtorch::sincos_pi_prod(c, out.dc, ph.s0[t], ph.c0[t]);
    }
}

// The weights of the table path, padded with zeros: row i, of ts = taps + 2 pad
// entries, holds scale * g(c (dc_i - k)) at k + pad, k < taps, and 0 around it.
// Evaluated in double, with sin(pi c (dc_i - k)) by angle addition from the
// tables of prep_kernel, and rounded to AccT once when stored.
template <typename AccT>
__global__ void table_kernel(AccT *__restrict__ table,
                             const double *__restrict__ sk,
                             const double *__restrict__ ck, PhaseOut ph,
                             int64_t taps, int64_t pad, double c, double inv_n,
                             const __grid_constant__ Coeffs<double> coef) {
    const int64_t ts = taps + 2 * pad;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= ph.p * ts) {
        return;
    }
    const int64_t i = idx / ts, k = idx - i * ts - pad;
    if (k < 0 || k >= taps) {
        table[idx] = static_cast<AccT>(0);
        return;
    }
    const auto grid = philtorch::make_tap_grid<double>(c, ph.dc[i]);
    const double s = ph.s0[i] * ck[k] - ph.c0[i] * sk[k];
    table[idx] = static_cast<AccT>(philtorch::kaiser_sinc<double>(
        grid.z(static_cast<int32_t>(k)), s, inv_n, coef.v, coef.terms));
}

// ---------------------------------------------------------------------------
// Table path.

template <typename AccT>
struct TileArgs {
    const AccT *x;             // (B, T), contiguous, B of this launch
    AccT *out;                 // (B, out_len)
    int64_t B, T, out_len;
    int64_t p, q, taps;
    int64_t ts, pad;           // table row stride, zero padding on either side
    const AccT *table;         // (p, ts)
    const int64_t *phase_nlo;  // (p)
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
template <typename AccT, int R, int BB>
__global__ void __launch_bounds__(kBlock) tile_kernel(const TileArgs<AccT> a) {
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
    const int64_t nlo_i0 = a.phase_nlo[i];
    const int64_t nlo0 = a.q * f0 + nlo_i0;
    int d[R];
    const AccT *wp[R];  // wp[j][u] is the weight of output j at input nlo0 + u
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
            d[j] = df * q + static_cast<int>(a.phase_nlo[i] - nlo_i0);
            wp[j] = a.table + static_cast<int64_t>(i) * a.ts + a.pad - d[j];
        }
    }
    const int dlast = d[R - 1];
    const int taps = static_cast<int>(a.taps);
    // The u range, with inputs in [0, T), that any output of the tile reads.
    const int64_t ulo64 = max(int64_t{0}, -nlo0);
    const int64_t uhi64 = min(static_cast<int64_t>(dlast) + taps, a.T - nlo0);
    AccT acc[A];
#pragma unroll
    for (int e = 0; e < A; ++e) {
        acc[e] = static_cast<AccT>(0);
    }
    if (ulo64 < uhi64) {
        const int ulo = static_cast<int>(ulo64), uhi = static_cast<int>(uhi64);
        const AccT *xp[BB];
#pragma unroll
        for (int bb = 0; bb < BB; ++bb) {
            xp[bb] = a.x + min(b0 + bb, a.B - 1) * a.T + nlo0;
        }
// Unrolled four times, so that the loads of the next iterations are issued
        // while the multiply-adds of this one run (measured: 1 to 3% in float).
#pragma unroll 4
        for (int u = ulo + lg; u < uhi; u += G) {
            AccT xv[BB];
#pragma unroll
            for (int bb = 0; bb < BB; ++bb) {
                xv[bb] = xp[bb][u];
            }
#pragma unroll
            for (int j = 0; j < R; ++j) {
                const AccT w = wp[j][u];
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
            a.out[b * a.out_len + m] = acc[e];
        }
    }
}

// The shape of a tile for B rows: BB rows (pow2_rows), and the R outputs that
// make kTileAcc accumulators, at most kMaxTileOutputs.
constexpr int tile_outputs(int bb) {
    return std::min(kTileAcc / bb, kMaxTileOutputs);
}

template <typename AccT, int BB>
void launch_tile(const TileArgs<AccT> &a, cudaStream_t stream) {
    constexpr int R = tile_outputs(BB);
    constexpr int tiles_per_block = kWarpsPerBlock * (kWarp / kGroupLanes);
    launch_chunked(a, int64_t{tiles_per_block} * R, BB,
                   [&](const TileArgs<AccT> &chunk, dim3 grid) {
                       tile_kernel<AccT, R, BB><<<grid, kBlock, 0, stream>>>(chunk);
                       C10_CUDA_KERNEL_LAUNCH_CHECK();
                   });
}

template <typename AccT>
void launch_tile_rows(const TileArgs<AccT> &a, int BB, cudaStream_t stream) {
    switch (BB) {
        case 8: launch_tile<AccT, 8>(a, stream); break;
        case 4: launch_tile<AccT, 4>(a, stream); break;
        case 2: launch_tile<AccT, 2>(a, stream); break;
        default: launch_tile<AccT, 1>(a, stream); break;
    }
}

// ---------------------------------------------------------------------------
// General path.

template <typename AccT>
struct GeneralArgs {
    const AccT *x;       // (B, T), contiguous, B of this launch
    AccT *out;           // (B, out_len)
    int64_t B, T, out_len;
    double S, c;
    int64_t p, q, taps;
    AccT inv_n;
    Coeffs<AccT> coef;   // scale * Kaiser series coefficients
    const AccT *sk;      // sin / cos (pi c k), (taps)
    const AccT *ck;
};

// G lanes per output (blockIdx.y selects the BB rows), 32 / G outputs per warp.
template <typename AccT, int BB, int G>
__global__ void __launch_bounds__(kBlock)
    general_kernel(const __grid_constant__ GeneralArgs<AccT> a) {
    constexpr int OPW = kWarp / G;  // outputs per warp
    const int lane = threadIdx.x & (kWarp - 1);
    const int lg = lane & (G - 1);
    const int64_t m0 =
        (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp) * OPW;
    if (m0 >= a.out_len) {
        return;  // whole warp
    }
    // The outputs of the warp past out_len still take part in the reduction.
    const int64_t m = m0 + lane / G;
    const bool active = m < a.out_len;
    const int64_t b0 = static_cast<int64_t>(blockIdx.y) * BB;
    const int64_t b_last = a.B - 1;

    const auto ph = philtorch::output_phase(min(m, a.out_len - 1), a.p, a.q, a.S);
    const int64_t nlo = ph.nlo;
    const int64_t kmin = max(int64_t{0}, -nlo);
    const int64_t kmax = min(a.taps, a.T - nlo);

    AccT acc[BB];
#pragma unroll
    for (int bb = 0; bb < BB; ++bb) {
        acc[bb] = static_cast<AccT>(0);
    }

    if (kmin < kmax) {
        const AccT *xp[BB];
#pragma unroll
        for (int bb = 0; bb < BB; ++bb) {
            xp[bb] = a.x + min(b0 + bb, b_last) * a.T + nlo;
        }

        AccT s0, c0;
        philtorch::sincos_pi_prod(a.c, ph.dc, s0, c0);
        const auto grid = philtorch::make_tap_grid<AccT>(a.c, ph.dc);

        for (int64_t k = kmin + lg; k < kmax; k += G) {
            const AccT z = grid.z(static_cast<int32_t>(k));
            const AccT s = s0 * a.ck[k] - c0 * a.sk[k];
            const AccT w = philtorch::kaiser_sinc<AccT>(z, s, a.inv_n, a.coef.v,
                                                        a.coef.terms);
#pragma unroll
            for (int bb = 0; bb < BB; ++bb) {
                acc[bb] = fma(w, xp[bb][k], acc[bb]);
            }
        }
    }

#pragma unroll
    for (int off = G / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int bb = 0; bb < BB; ++bb) {
            acc[bb] += WARP_SHFL_DOWN(acc[bb], off, G);
        }
    }
    if (lg == 0 && active) {
#pragma unroll
        for (int bb = 0; bb < BB; ++bb) {
            if (b0 + bb < a.B) {
                a.out[(b0 + bb) * a.out_len + m] = acc[bb];
            }
        }
    }
}

template <typename AccT, int BB, int G>
void launch_general(const GeneralArgs<AccT> &a, cudaStream_t stream) {
    constexpr int outputs_per_block = kWarpsPerBlock * (kWarp / G);
    launch_chunked(a, outputs_per_block, BB,
                   [&](const GeneralArgs<AccT> &chunk, dim3 grid) {
                       general_kernel<AccT, BB, G><<<grid, kBlock, 0, stream>>>(chunk);
                       C10_CUDA_KERNEL_LAUNCH_CHECK();
                   });
}

template <typename AccT, int BB>
void launch_general_lanes(const GeneralArgs<AccT> &a, cudaStream_t stream) {
    if (a.taps > kNarrowLanes) {
        launch_general<AccT, BB, kGroupLanes>(a, stream);
    } else {
        launch_general<AccT, BB, kNarrowLanes>(a, stream);
    }
}

template <typename AccT>
void launch_general_rows(const GeneralArgs<AccT> &a, int BB, cudaStream_t stream) {
    switch (BB) {
        case 16: launch_general_lanes<AccT, 16>(a, stream); break;
        case 8: launch_general_lanes<AccT, 8>(a, stream); break;
        case 4: launch_general_lanes<AccT, 4>(a, stream); break;
        case 2: launch_general_lanes<AccT, 2>(a, stream); break;
        default: launch_general_lanes<AccT, 1>(a, stream); break;
    }
}

// ---------------------------------------------------------------------------

template <typename AccT>
void resample_impl(const at::Tensor &x, at::Tensor &out, double beta,
                   double scale, const ResamplePlan &plan) {
    const auto opts = x.options();
    const auto dopts = opts.dtype(at::kDouble);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t B = x.size(0), T = x.size(1), out_len = out.size(1);

    // Kept alive until the launches are enqueued; the caching allocator
    // orders their reuse on the stream.
    at::Tensor tables, phases, phase_nlo, table;

    if (plan.table) {
        // The table path evaluates in double, whatever AccT is.
        tables = at::empty({2, plan.taps}, dopts);  // sin / cos (pi c k)
        phases = at::empty({3, plan.p}, dopts);     // s0, c0, dc
        phase_nlo = at::empty({plan.p}, opts.dtype(at::kLong));
        double *sk = tables.mutable_data_ptr<double>(), *ck = sk + plan.taps;
        double *s0 = phases.mutable_data_ptr<double>();
        const PhaseOut ph{plan.p, plan.q, plan.S, phase_nlo.mutable_data_ptr<int64_t>(),
                          s0, s0 + plan.p, s0 + 2 * plan.p};
        const int64_t prep_n = std::max(plan.p, plan.taps);
        prep_kernel<double><<<at::ceil_div<int64_t>(prep_n, kBlock), kBlock, 0, stream>>>(
            sk, ck, plan.taps, plan.c, ph);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // The tiles' batch rows BB and outputs R, and the zero padding pad of a
        // table row on either side: the first inputs of two outputs of a tile
        // are at most the exact bound ceil((R - 1) q / p) apart, and pad is one
        // more, a margin against a rounding in output_phase.  The plan's bounds
        // keep the int arithmetic of tile_kernel (offsets within a tile, p, q)
        // from overflowing, whatever T is: a table plan has p <= kMaxTable /
        // taps, and q <= p taps / 2, since S = N q / (rolloff p) >= q / p when
        // p < q and q <= p otherwise; so p, q <= kMaxTable, pad <= R taps / 2 + 2,
        // and the padded table has at most kMaxTable + 2 (R q + 2 p) elements.
        const int BB = pow2_rows(B, kMaxTileRows);
        const int R = tile_outputs(BB);
        const int64_t pad = at::ceil_div<int64_t>((R - 1) * plan.q, plan.p) + 1;
        TORCH_INTERNAL_ASSERT(plan.p <= philtorch::kMaxTable &&
                              plan.q <= philtorch::kMaxTable);
        const int64_t ts = plan.taps + 2 * pad;
        const int64_t n = plan.p * ts;
        table = at::empty({n}, opts);
        table_kernel<AccT><<<at::ceil_div<int64_t>(n, kBlock), kBlock, 0, stream>>>(
            table.mutable_data_ptr<AccT>(), sk, ck, ph, plan.taps, pad, plan.c,
            plan.inv_n, kaiser_args<double>(beta, scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        TileArgs<AccT> a;
        a.x = x.const_data_ptr<AccT>();
        a.out = out.mutable_data_ptr<AccT>();
        a.B = B;
        a.T = T;
        a.out_len = out_len;
        a.p = plan.p;
        a.q = plan.q;
        a.taps = plan.taps;
        a.ts = ts;
        a.pad = pad;
        a.table = table.const_data_ptr<AccT>();
        a.phase_nlo = phase_nlo.const_data_ptr<int64_t>();
        launch_tile_rows<AccT>(a, BB, stream);
    } else {
        GeneralArgs<AccT> a;
        a.x = x.const_data_ptr<AccT>();
        a.out = out.mutable_data_ptr<AccT>();
        a.B = B;
        a.T = T;
        a.out_len = out_len;
        a.S = plan.S;
        a.c = plan.c;
        a.p = plan.p;
        a.q = plan.q;
        a.taps = plan.taps;
        a.inv_n = static_cast<AccT>(plan.inv_n);
        a.coef = kaiser_args<AccT>(beta, scale);
        tables = at::empty({2, plan.taps}, opts);
        AccT *sk = tables.mutable_data_ptr<AccT>();
        a.sk = sk;
        a.ck = sk + plan.taps;
        prep_kernel<AccT><<<at::ceil_div<int64_t>(plan.taps, kBlock), kBlock, 0, stream>>>(
            sk, sk + plan.taps, plan.taps, plan.c, PhaseOut{});
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        launch_general_rows<AccT>(a, pow2_rows(B, kMaxGeneralRows), stream);
    }
}

}  // namespace

at::Tensor sinc_resample_cuda_impl(const at::Tensor &x, int64_t orig_freq,
                                   int64_t new_freq, double scale,
                                   double rolloff, double beta,
                                   int64_t num_zeros, int64_t out_len) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    return philtorch::resample_entry(
        x, orig_freq, new_freq, scale, rolloff, beta, num_zeros, out_len,
        [&](auto type, const at::Tensor &xa, at::Tensor &out,
            const ResamplePlan &plan) {
            resample_impl<typename decltype(type)::type>(xa, out, beta, scale,
                                                         plan);
        });
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m) {
    m.impl("resample", &sinc_resample_cuda_impl);
}
