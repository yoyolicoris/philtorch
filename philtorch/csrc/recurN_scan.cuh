#pragma once

// The recurrence h[t] = A[t] h[t - 1] + x[t] of M states, for any M up to 128
// (64 for complex), as one single-pass scan with decoupled look-back: the CUDA
// kernel of the ops recurN (A varies with t) and lti_recurN (it does not).
//
// A grid block claims the next chunk of a batch item from an atomic ticket, so
// every chunk it waits on belongs to a block that is already running. Chunk 0
// runs its steps from the initial state and publishes its end state. Every
// other chunk
//
//   1. composes its map h_end = P h_start + o, o being the chunk's steps from
//      a zero start, and, but for the last chunk, publishes it (flag 1);
//   2. looks back over the flags to the latest predecessor with its end state
//      (flag 2), and applies the maps in between to get its own start;
//   3. publishes its end state (flag 2), but for the last chunk;
//   4. carries its start through its steps, writing the output.
//
// Time-invariant chunks share P = A^chunk, which a kernel computes by repeated
// squaring before the scan, so a chunk's map costs O(M^2) per step. A
// time-varying chunk composes its own P, as the columns Q[j] = P e_j, each
// step turning Q[j] into A[n] Q[j] (O(M^3) per step); the columns live in a
// slab in shared memory, in passes of PR columns where the slab would not fit.
//
// Layout. BM, the state size padded to a power of 2 (at least 2), fixes it.
// KS lanes share a row of a matrix, each holding CS = BM / KS consecutive
// entries of the row in registers; a matrix-vector product is, per lane, CS
// multiply-adds against the matching slice of the vector, then a butterfly
// over the row's KS lanes that leaves the sum in all of them. The slice comes
// by shuffles from the lanes that hold the entries, when the BM KS lanes of a
// matrix fit a warp, and otherwise through a shared array (padded, so that the
// slices fall into different banks) and one barrier per product. Several
// groups of lanes fill a warp when BM KS < 32. A block holds units of the
// lanes of one matrix; in a time-varying chunk's composition the units work on
// disjoint columns of P, and one unit runs everything else. Loads of entries
// past M are predicated, never a branch around a shuffle, and A is read
// unpadded. A time-varying scan reads A and x ahead through a ring in
// registers; a time-invariant one keeps A (or P) in registers and reads x one
// step ahead.
//
// A thread's steps compute in Acc: float for Half and BFloat16, the type
// itself otherwise; int64 wraps around, as integer arithmetic does, and a
// complex number is a pair of reals, four real products per product. A map,
// offset or state is published with stores, a fence, a barrier, and one
// release store of the flag by one thread; a look-back is one acquire load by
// one thread, a barrier, then the reads.

#include "stable_support.h"

#include <cstdint>
#include <type_traits>

namespace philtorch::recur_n
{
    constexpr unsigned FULL = 0xffffffffu;
    // The largest padded state size, a complex state counting twice.
    constexpr int MAX_STATES = 128;

    // A complex number in registers and shared memory: torch's c10::complex
    // has default member initializers, so a static __shared__ array of it is an
    // error. Same layout.
    template <class R>
    struct alignas(2 * sizeof(R)) cx
    {
        R re, im;
    };

    template <class A>
    constexpr bool is_cx = false;
    template <class R>
    constexpr bool is_cx<cx<R>> = true;

    template <class T>
    struct device_type
    {
        using type = T;
    };
    template <class R>
    struct device_type<c10::complex<R>>
    {
        using type = cx<R>;
    };
    // The type a scalar type of the op accumulates in: philtorch::opmath_type,
    // as cx for complex.
    template <class S>
    using acc_t = typename device_type<philtorch::opmath_type<S>>::type;

    // ---- arithmetic in Acc: float, double, int64_t, cx<float>, cx<double>

    template <class T>
    constexpr bool is_c10_complex = false;
    template <class R>
    constexpr bool is_c10_complex<c10::complex<R>> = true;

    // A stored value in the accumulation type, and back.
    template <class T>
    __device__ __forceinline__ auto to_acc(T v)
    {
        if constexpr (is_c10_complex<T>)
            return cx<typename T::value_type>{v.real(), v.imag()};
        else
            return static_cast<philtorch::opmath_type<T>>(v);
    }

    template <class S>
    __device__ __forceinline__ void store(S *p, acc_t<S> v)
    {
        if constexpr (is_c10_complex<S>)
            *p = S(v.re, v.im);
        else
            *p = static_cast<S>(v);
    }

    template <class A>
    __device__ __forceinline__ A num(int v)
    {
        if constexpr (is_cx<A>)
        {
            using R = decltype(A::re);
            return {R(v), R(0)};
        }
        else
            return A(v);
    }

    template <class A>
    __device__ __forceinline__ A plus(A a, A b)
    {
        if constexpr (is_cx<A>)
            return {plus(a.re, b.re), plus(a.im, b.im)};
        else if constexpr (std::is_integral_v<A>)
            return static_cast<A>(static_cast<unsigned long long>(a) + static_cast<unsigned long long>(b));
        else
            return a + b;
    }

    // c + a b
    template <class A>
    __device__ __forceinline__ A mad(A a, A b, A c)
    {
        if constexpr (is_cx<A>)
        {
            auto re = mad(a.re, b.re, c.re);
            auto im = mad(a.re, b.im, c.im);
            re = mad(-a.im, b.im, re);
            im = mad(a.im, b.re, im);
            return {re, im};
        }
        else if constexpr (std::is_integral_v<A>)
            return static_cast<A>(static_cast<unsigned long long>(a) * static_cast<unsigned long long>(b) +
                                  static_cast<unsigned long long>(c));
        else if constexpr (std::is_same_v<A, float>)
            return fmaf(a, b, c);
        else
            return ::fma(a, b, c);
    }

    template <class A>
    __device__ __forceinline__ A shfl(A v, int lane)
    {
        if constexpr (is_cx<A>)
            return {shfl(v.re, lane), shfl(v.im, lane)};
        else
            return __shfl_sync(FULL, v, lane);
    }

    template <class A>
    __device__ __forceinline__ A shfl_xor(A v, int mask)
    {
        if constexpr (is_cx<A>)
            return {shfl_xor(v.re, mask), shfl_xor(v.im, mask)};
        else
            return __shfl_xor_sync(FULL, v, mask);
    }

    // Loads and stores through L2, for what other blocks publish.
    template <class A>
    __device__ __forceinline__ A ldcg(const A *p)
    {
        if constexpr (is_cx<A>)
        {
            using V = std::conditional_t<sizeof(A) == 8, float2, double2>;
            const V v = __ldcg(reinterpret_cast<const V *>(p));
            return {v.x, v.y};
        }
        else
            return __ldcg(p);
    }

    template <class A>
    __device__ __forceinline__ void stcg(A *p, A v)
    {
        if constexpr (is_cx<A>)
        {
            using V = std::conditional_t<sizeof(A) == 8, float2, double2>;
            __stcg(reinterpret_cast<V *>(p), V{v.re, v.im});
        }
        else
            __stcg(p, v);
    }

    __device__ __forceinline__ int ld_acquire(const int *p)
    {
        int v;
        asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
        return v;
    }

    __device__ __forceinline__ void st_release(int *p, int v)
    {
        asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
    }

    // V elements of T at p, aligned for V * sizeof(T) bytes: one wide load.
    template <int V, class T>
    __device__ __forceinline__ void load_vec(const T *p, T (&v)[V])
    {
        struct alignas(V * sizeof(T)) Vec
        {
            T v[V];
        };
        const Vec u = *reinterpret_cast<const Vec *>(p);
#pragma unroll
        for (int q = 0; q < V; ++q)
            v[q] = u.v[q];
    }

    // The largest power of 2 up to cap that divides n.
    __host__ __device__ constexpr int vec_width(int n, int cap)
    {
        int v = cap;
        while (n % v != 0)
            v /= 2;
        return v;
    }

    // The most columns of a slab of BM, at row_bytes per column, up to 40 KB.
    __host__ __device__ constexpr int pass_columns(int BM, int row_bytes)
    {
        int pr = BM;
        while (pr * row_bytes > 40960)
            pr /= 2;
        return pr;
    }

    // ---- the layout, a function of the padded state size BM and the type

    template <class A, int BM, bool VARYING>
    struct Cfg
    {
        static constexpr int KS = BM < 32 ? (BM < 32 / BM ? BM : 32 / BM) : 2; // lanes per row
        static constexpr int CS = BM / KS;                                      // entries per lane
        static constexpr int LANES = BM * KS;                                   // lanes of a matrix
        static constexpr int NGL = LANES < 32 ? 32 / LANES : 1;                 // groups of lanes (a matrix each) per warp
        static constexpr int GW = LANES >= 32 ? LANES / 32 : 1;                 // warps per matrix
        // A time-varying block of BM >= 16 has at least 4 warps, whose units
        // compose different columns of P.
        static constexpr int MIN_NT = VARYING && BM >= 16 ? 128 : 32;
        static constexpr int NT = LANES > MIN_NT ? LANES : MIN_NT;
        static constexpr int NGW = NT / (32 * GW); // units
        static constexpr int NG = NGL * NGW;       // groups of lanes (a matrix each) per block
        static constexpr bool SHFL = GW == 1;      // the vector's slices by shuffles
        static constexpr int VS = vec_width(CS, 16 / (int)sizeof(A)); // entries per vector in shared memory
        static constexpr int PAD = VS > 1 && VS == 16 / (int)sizeof(A) ? VS : 0;
        static constexpr int SR = BM + PAD * KS; // padded length of a vector
        // Composition: PR columns of P per pass, the most whose slab (two
        // buffers) fits 40 KB of shared memory.
        static constexpr int PR = pass_columns(BM, 2 * SR * (int)sizeof(A));
        static constexpr int NPASS = BM / PR;
        static constexpr int RB_MAX = 8 / KS > 1 ? 8 / KS : 1;
        static constexpr int RB_LIM = BM / (NG * KS) > 1 ? BM / (NG * KS) : 1;
        static constexpr int RB = RB_MAX < RB_LIM ? RB_MAX : RB_LIM; // column groups per iteration
        static constexpr int CPI = NG * KS * RB;                     // columns per iteration
        static constexpr int NQ = (PR + CPI - 1) / CPI;
        static constexpr int SLAB_ROWS = NQ * CPI; // rows read, those past PR junk
        // Steps of A and x read ahead, plus one: 4 while the ring of A takes
        // at most 64 registers, else 2. Measured on an RTX 5060 Ti.
        static constexpr int RING = 4 * CS * (int)sizeof(A) <= 256 ? 4 : 2;
    };

    // Barrier of the threads of unit u: the GW warps that hold a matrix. The
    // units of a block work on disjoint columns and never wait for each other.
    template <int GW, int NGW>
    __device__ __forceinline__ void usync(int u)
    {
        if constexpr (GW == 1)
            __syncwarp();
        else if constexpr (NGW == 1)
            __syncthreads();
        else
        {
            static_assert(NGW == 2, "several warps per matrix and units: BM = 32, time-varying");
            if (u == 0)
                asm volatile("bar.sync 1, %0;" ::"n"(GW * 32) : "memory");
            else
                asm volatile("bar.sync 2, %0;" ::"n"(GW * 32) : "memory");
        }
    }

    // v as an O: converted to the accumulation type, or unchanged where O is T.
    template <class O, class T>
    __device__ __forceinline__ O convert(T v)
    {
        if constexpr (std::is_same_v<O, T>)
            return v;
        else
            return to_acc(v);
    }

    // This thread's entries of a row of a matrix in memory, as p points at the
    // first: the first nvalid of the CS are real, the rest zeros, as O (the
    // type T, or the accumulation type). vec: the entries at p are aligned for
    // vector loads, and nvalid is a multiple of their width.
    template <class O, int CS, class T>
    __device__ __forceinline__ void load_slice(O (&a)[CS], const T *p, int nvalid, bool vec)
    {
        constexpr int V = vec_width(CS, 16 / (int)sizeof(T));
        if constexpr (V > 1)
        {
            if (vec)
            {
#pragma unroll
                for (int i = 0; i < CS; i += V)
                {
                    T v[V] = {};
                    if (i < nvalid)
                        load_vec<V>(p + i, v);
#pragma unroll
                    for (int q = 0; q < V; ++q)
                        a[i + q] = convert<O>(v[q]);
                }
                return;
            }
        }
#pragma unroll
        for (int i = 0; i < CS; ++i)
        {
            T v = T();
            if (i < nvalid)
                v = p[i];
            a[i] = convert<O>(v);
        }
    }

    // A thread's view of the layout, and the products that use it.
    template <class A, int BM, bool VARYING>
    struct Blk
    {
        using Cf = Cfg<A, BM, VARYING>;
        static constexpr int KS = Cf::KS, CS = Cf::CS, VS = Cf::VS, PAD = Cf::PAD;
        static constexpr int NACC = CS >= 4 ? 4 : CS; // independent chains of a dot

        int u, gl, g, row, k;
        bool valid;  // row < M
        bool lead;   // k == 0 of the first group of a unit's warp
        bool writer; // stores the row: the lead of unit 0
        int nvalid;  // real entries of this thread's slice of its row
        int par;
        A (*s_vec)[Cf::SR]; // [2][SR] the vector's entries, several warps

        __device__ void init(int M, A (*sv)[Cf::SR])
        {
            const int w = threadIdx.x >> 5;
            u = Cf::NGW == 1 ? 0 : w / Cf::GW;
            const int lu = (w % Cf::GW) * 32 + (threadIdx.x & 31); // lane in the unit
            gl = lu / Cf::LANES;
            const int r = lu % Cf::LANES;
            g = u * Cf::NGL + gl;
            row = r / KS;
            k = r % KS;
            valid = row < M;
            lead = k == 0 && gl == 0;
            writer = valid && lead && u == 0;
            const int nv = M - k * CS;
            nvalid = !valid ? 0 : nv < 0 ? 0 : nv < CS ? nv : CS;
            par = 0;
            s_vec = sv;
        }

        static __device__ __forceinline__ int sidx(int c) { return c + (c / CS) * PAD; }

        // This thread's slice of the vector whose entry `row` the KS lanes of
        // the row hold in y: by shuffles, or from buf, which the callers have
        // filled and made visible.
        __device__ __forceinline__ void slice(A y, const A *buf, A (&v)[CS]) const
        {
            if constexpr (Cf::SHFL)
            {
                __syncwarp(); // convergence for ptxas: the shuffles then compile without a slow path
                const int base = gl * Cf::LANES + k * CS * KS;
#pragma unroll
                for (int i = 0; i < CS; ++i)
                    v[i] = shfl(y, base + i * KS);
            }
            else
            {
                const A *src = buf + k * (CS + PAD);
#pragma unroll
                for (int i = 0; i < CS; i += VS)
                {
                    A r[VS];
                    load_vec<VS>(src + i, r);
#pragma unroll
                    for (int q = 0; q < VS; ++q)
                        v[i + q] = r[q];
                }
            }
        }

        // Writes the row's entry into buf, for slice (the caller then barriers).
        __device__ __forceinline__ void put(A y, A *buf) const
        {
            if (lead)
                buf[sidx(row)] = y;
        }

        __device__ __forceinline__ void usync() const { recur_n::usync<Cf::GW, Cf::NGW>(u); }

        // The sum of a[i] v[i] over the slice, then over the KS lanes: the
        // row's value in all of them.
        __device__ __forceinline__ A dot(const A (&a)[CS], const A (&v)[CS]) const
        {
            A acc[NACC];
#pragma unroll
            for (int q = 0; q < NACC; ++q)
                acc[q] = num<A>(0);
#pragma unroll
            for (int i = 0; i < CS; ++i)
                acc[i % NACC] = mad(a[i], v[i], acc[i % NACC]);
            A s = acc[0];
#pragma unroll
            for (int q = 1; q < NACC; ++q)
                s = plus(s, acc[q]);
#pragma unroll
            for (int off = 1; off < KS; off <<= 1)
                s = plus(s, shfl_xor(s, off));
            return s;
        }

        // The matrix a times the vector y of the rows' entries, zero in the
        // padding rows.
        __device__ __forceinline__ A matvec(const A (&a)[CS], A y)
        {
            A v[CS];
            if constexpr (Cf::SHFL)
                slice(y, nullptr, v);
            else
            {
                A *b = s_vec[par];
                put(y, b);
                usync();
                slice(y, b, v);
                par ^= 1;
            }
            const A s = dot(a, v);
            return valid ? s : num<A>(0);
        }
    };

    // A (Ba, M, M) or (Ba, T, M, M) as the op takes it, strides in elements: A
    // + b stride_ab is batch item b's, + n M M for step n if VARYING. P (Ba,
    // M, M) is A^chunk in Acc, and stride_pb its batch stride.
    // x, out (B, T, M); zi (B, M); offsets, states (B C, M) and maps (B C, M, M)
    // in Acc; flags B C + 1 ints, zeroed, the last the ticket counter. aligned:
    // A's address is a multiple of 16 bytes.
    template <class S, int BM, bool VARYING>
    __global__ void __launch_bounds__(Cfg<acc_t<S>, BM, VARYING>::NT)
        scan_kernel(const S *__restrict__ A, const S *__restrict__ x, const S *__restrict__ zi,
                    const acc_t<S> *__restrict__ P, S *__restrict__ out, acc_t<S> *offsets,
                    acc_t<S> *maps, acc_t<S> *states, int *flags, int T, int C, int chunk,
                    int64_t stride_ab, int64_t stride_pb, int M, bool aligned)
    {
        using Acc = acc_t<S>;
        using Cf = Cfg<Acc, BM, VARYING>;
        constexpr int NT = Cf::NT, NG = Cf::NG, KS = Cf::KS, CS = Cf::CS, PR = Cf::PR,
                      NPASS = Cf::NPASS, RB = Cf::RB, CPI = Cf::CPI, SR = Cf::SR, PAD = Cf::PAD,
                      VS = Cf::VS, RING = Cf::RING;
        __shared__ int s_int;
        __shared__ __align__(16) Acc s_slab[VARYING ? 2 : 1][VARYING ? Cf::SLAB_ROWS * SR : 1];
        __shared__ __align__(16) Acc s_vec[Cf::SHFL ? 1 : 2][SR];
        __shared__ __align__(16) Acc s_ov[2][VARYING ? SR : 1]; // the offset's vector, as s_vec, during a composition
        static_assert(sizeof(s_slab) + sizeof(s_vec) + sizeof(s_ov) <= 48 * 1024 - 64, "shared memory");

        Blk<Acc, BM, VARYING> blk;
        blk.init(M, s_vec);
        const int tid = threadIdx.x;
        const Acc zero = num<Acc>(0);

        if (tid == 0)
            s_int = atomicAdd(flags + static_cast<int64_t>(gridDim.x), 1);
        __syncthreads();
        const int index = s_int;
        __syncthreads();
        const int b = index / C;
        const int c = index - b * C;
        const int first = c * chunk;
        const int rest = T - first;
        const int steps = rest < chunk ? rest : chunk;
        const int64_t xb = static_cast<int64_t>(b) * T * M;
        const int MM = M * M;
        const bool publish = c < C - 1; // the last chunk's map and end state are read by no one
        // Thread's slice of its row of A[n]: arow + n MM; its entry of x[n]: xrow + n M.
        const S *arow = A + b * stride_ab + static_cast<int64_t>(blk.row) * M + blk.k * CS;
        const S *xrow = x + xb + blk.row;
        constexpr int VG = vec_width(CS, 16 / (int)sizeof(S));
        const bool vec = aligned && M % VG == 0;
        const bool vec_p = M % vec_width(CS, 16 / (int)sizeof(Acc)) == 0; // P, from the scratch: aligned

        Acc y = blk.valid ? to_acc(zi[static_cast<int64_t>(b) * M + blk.row]) : zero;
        Acc o = zero;
        Acc ma[CS]; // time-invariant: the matrix in registers, A or P

        // Publishes this chunk's vector (its offset: flag 1, or its end state: flag 2):
        // stores, a fence, a barrier, then the flag. Offsets are published by every
        // unit's threads, so the barrier is the block's; end states by unit 0's.
        auto announce = [&](Acc *dst, Acc v, int flag)
        {
            if (blk.writer)
                stcg(dst + static_cast<int64_t>(index) * M + blk.row, v);
            __threadfence();
            if (flag == 1)
                __syncthreads();
            else
                blk.usync();
            if (tid == 0)
                st_release(flags + index, flag);
        };

        // body(a, xv) for the chunk's steps in turn, a and xv read RING - 1
        // steps ahead (the last step's again past the end), through pointers
        // that advance by a step, so that no step computes an address.
        auto run = [&](auto &&body)
        {
            const S *xp = xrow + static_cast<int64_t>(first) * M;
            auto load_x = [&]() // as stored: see load_slice
            { return blk.valid ? *xp : S(); };
            if constexpr (!VARYING)
            {
                // Only x is read ahead, so no ring. The loop unrolls 4 times (2
                // above CS = 32, for code size): up to 40% faster than rolled.
                S xv = load_x();
#pragma unroll(CS <= 32 ? 4 : 2)
                for (int s = 0; s < steps; ++s)
                {
                    if (s + 1 < steps)
                        xp += M;
                    const S xn = load_x();
                    body(ma, to_acc(xv));
                    xv = xn;
                }
                return;
            }
            const S *ap = arow + static_cast<int64_t>(first) * MM;
            // Up to BM = 16, the ring holds A as stored and converts a step as
            // it uses it: converted at once, a 16-bit load stalls the thread
            // for its result, and the read-ahead is lost. Above, a step has
            // work enough to hide the load, and converting at once is faster.
            using R = std::conditional_t<(BM <= 16), S, Acc>;
            R ra[RING][CS], rx[RING];
            auto load = [&](int d, int n)
            {
                load_slice(ra[d], ap, blk.nvalid, vec);
                rx[d] = convert<R>(load_x());
                if (n + 1 < steps)
                {
                    ap += MM;
                    xp += M;
                }
            };
#pragma unroll
            for (int d = 0; d < RING - 1; ++d)
                load(d, d);
#pragma unroll 1
            for (int s = 0; s < steps; s += RING)
            {
#pragma unroll
                for (int d = 0; d < RING; ++d)
                {
                    if (s + d < steps)
                    {
                        load((d + RING - 1) % RING, s + d + RING - 1);
                        Acc a[CS];
#pragma unroll
                        for (int i = 0; i < CS; ++i)
                            a[i] = to_acc(ra[d][i]);
                        body(a, to_acc(rx[d]));
                    }
                }
            }
        };

        if constexpr (!VARYING)
            load_slice(ma, arow, blk.nvalid, vec);

        if (c > 0)
        {
            // 1. The chunk's map: o from a zero start, and P.
            // The last chunk composes its map too, unread: skipping it, its
            // look-back spins on predecessors still composing, up to 28% slower.
            if constexpr (VARYING)
            {
                // P as its columns Q[j] (Q[j] = P e_j, j < M), in passes of PR.
                for (int p = 0; p < NPASS; ++p)
                {
                    const int rp0 = p * PR;
                    const int Rp = (M - rp0) < PR ? (M - rp0) : PR;
                    if (Rp <= 0)
                        break;
                    const int nq = (Rp + CPI - 1) / CPI;
                    for (int e = tid; e < PR * BM; e += NT)
                    {
                        const int cc = e % BM;
                        s_slab[0][(e / BM) * SR + Blk<Acc, BM, VARYING>::sidx(cc)] =
                            num<Acc>(rp0 + e / BM == cc ? 1 : 0);
                    }
                    if (p == 0)
                        for (int e = tid; e < SR; e += NT)
                            s_ov[0][e] = zero;
                    __syncthreads();
                    int cur = 0;
                    run([&](const Acc (&a)[CS], Acc xv)
                        {
                            const Acc *src = s_slab[cur] + blk.k * (CS + PAD);
                            Acc *dst = s_slab[cur ^ 1];
#pragma unroll 1
                            for (int q = 0; q < nq; ++q)
                            {
                                const int j0 = (q * NG + blk.g) * (RB * KS); // first column of this group
                                Acc pa[RB][KS] = {};
#pragma unroll
                                for (int r = 0; r < RB; ++r)
#pragma unroll
                                    for (int kk = 0; kk < KS; ++kk)
                                    {
                                        const Acc *sp = src + (j0 + r * KS + kk) * SR;
#pragma unroll
                                        for (int i = 0; i < CS; i += VS)
                                        {
                                            Acc v[VS];
                                            load_vec<VS>(sp + i, v);
#pragma unroll
                                            for (int qq = 0; qq < VS; ++qq)
                                                pa[r][kk] = mad(a[i + qq], v[qq], pa[r][kk]);
                                        }
                                    }
                                // Reduce-scatter over the KS lanes: lane k keeps column j0 + r KS + k.
#pragma unroll
                                for (int r = 0; r < RB; ++r)
                                {
#pragma unroll
                                    for (int h = KS / 2; h >= 1; h >>= 1)
                                    {
                                        const bool hi = (blk.k & h) != 0;
#pragma unroll
                                        for (int qq = 0; qq < h; ++qq)
                                        {
                                            const Acc lo = pa[r][qq], up = pa[r][qq + h];
                                            pa[r][qq] = plus(hi ? up : lo, shfl_xor(hi ? lo : up, h));
                                        }
                                    }
                                    const int jc = j0 + r * KS + blk.k;
                                    if (jc < Rp)
                                        dst[jc * SR + Blk<Acc, BM, VARYING>::sidx(blk.row)] = blk.valid ? pa[r][0] : zero;
                                }
                            }
                            if (p == 0 && blk.u == 0)
                            {
                                Acc ov[CS];
                                blk.slice(o, s_ov[cur], ov);
                                // Every lane takes part in dot's shuffles; then zero in the
                                // padding rows, as matvec's: 0 inf there would be NaN.
                                const Acc d = plus(blk.dot(a, ov), xv);
                                o = blk.valid ? d : zero;
                                if constexpr (!Cf::SHFL)
                                    blk.put(o, s_ov[cur ^ 1]);
                            }
                            blk.usync();
                            cur ^= 1; });
                    __syncthreads(); // the columns of all units
                    if (publish)
                    {
                        const Acc *fin = s_slab[cur];
                        for (int e = tid; e < Rp * M; e += NT)
                        {
                            const int jl = e / M, tt = e - jl * M;
                            stcg(maps + static_cast<int64_t>(index) * MM + static_cast<int64_t>(rp0 + jl) * M + tt,
                                 fin[jl * SR + Blk<Acc, BM, VARYING>::sidx(tt)]);
                        }
                    }
                    __syncthreads();
                }
            }
            else
            {
                run([&](const Acc (&a)[CS], Acc xv)
                    { o = plus(blk.matvec(a, o), xv); });
            }
            if (publish)
                announce(offsets, o, 1);
            if constexpr (!VARYING)
                load_slice(ma, P + b * stride_pb + static_cast<int64_t>(blk.row) * M + blk.k * CS, blk.nvalid, vec_p);
            // 2. Back to the latest predecessor with its end state.
            if (tid == 0)
            {
                int j = c - 1;
                int f = ld_acquire(flags + static_cast<int64_t>(b) * C + j);
                while (f != 2)
                {
                    if (f == 1)
                        --j;
                    f = ld_acquire(flags + static_cast<int64_t>(b) * C + j);
                }
                s_int = j;
            }
            __syncthreads();
            const int j = s_int;
            if (blk.u == 0)
            {
                // A chunk's map: the matrix whose entry (row, j) is maps[j][row].
                auto apply = [&](int i, Acc v)
                {
                    if constexpr (VARYING)
                    {
                        const Acc *map = maps + static_cast<int64_t>(i) * MM;
                        Acc a[CS];
#pragma unroll
                        for (int q = 0; q < CS; ++q)
                        {
                            a[q] = q < blk.nvalid ? ldcg(map + static_cast<int64_t>(blk.k * CS + q) * M + blk.row) : zero;
                        }
                        return blk.matvec(a, v);
                    }
                    else
                        return blk.matvec(ma, v);
                };
                y = blk.valid ? ldcg(states + (static_cast<int64_t>(b) * C + j) * M + blk.row) : zero;
#pragma unroll 1
                for (int i = j + 1; i < c; ++i)
                {
                    const Acc a = blk.valid ? ldcg(offsets + (static_cast<int64_t>(b) * C + i) * M + blk.row) : zero;
                    y = plus(apply(b * C + i, y), a);
                }
                if (publish)
                {
                    // 3. The chunk's end state.
                    announce(states, plus(apply(index, y), o), 2);
                }
            }
            if constexpr (!VARYING)
                load_slice(ma, arow, blk.nvalid, vec);
        }

        // 4. The chunk's steps, by unit 0 (the other units are done).
        if (blk.u == 0)
        {
            S *op = out + xb + blk.row + static_cast<int64_t>(first) * M;
            run([&](const Acc (&a)[CS], Acc xv)
                {
                    y = plus(blk.matvec(a, y), xv);
                    if (blk.writer)
                        store(op, y);
                    op += M; });
            if (C > 1 && c == 0)
            {
                // Chunk 0 starts from zi: its steps give its end state, no map needed.
                announce(states, y, 2);
            }
        }
    }

    // ---- the chunk power: A^(2^squarings) of each matrix, in Acc, one tiled
    // matrix-product kernel for every dtype and size.
    //
    // A block computes one BT x BT tile of one matrix's product (grid: tiles x
    // matrices), BT = min(BM, 16), a thread one entry: it stages the BT rows of
    // the left factor and the BT columns of the right one (full K strips) in
    // shared memory. The K loop runs to M: the padding from M on is zero.
    // Entries from M on are written as zeros, never computed, so that a NaN or
    // inf in A cannot reach them.
    // The host decides how many squarings a launch does: see chunk_power.

    template <int BM>
    struct PowerCfg
    {
        static constexpr int BT = BM < 16 ? BM : 16; // tile edge
        static constexpr int TILES = BM / BT;        // tiles per matrix edge
        static constexpr int NT = BT * BT;
    };

    // a_in (M x M each, in S), when not null, is the matrix to square first;
    // else src0 (M x M each, in Acc) is. p and tmp: the buffers the squarings
    // land in, the last in p (tmp is unused when nsq == 1). nsq > 1 needs one
    // tile per matrix. The grid is (matrices, tiles).
    template <class S, int BM>
    __global__ void __launch_bounds__(PowerCfg<BM>::NT)
        power_kernel(const S *a_in, const acc_t<S> *src0, acc_t<S> *p, acc_t<S> *tmp, int M, int nsq)
    {
        using A = acc_t<S>;
        constexpr int BT = PowerCfg<BM>::BT, TILES = PowerCfg<BM>::TILES, NT = PowerCfg<BM>::NT;
        static_assert(2 * BT * BM * sizeof(A) <= 48 * 1024, "K strips exceed static shared memory");
        __shared__ A As[BT * BM];
        __shared__ A Bs[BM * BT];
        const int64_t off = static_cast<int64_t>(blockIdx.x) * M * M;
        const int row0 = (blockIdx.y / TILES) * BT, col0 = (blockIdx.y % TILES) * BT;
        const int tx = threadIdx.x % BT, ty = threadIdx.x / BT;
        const A *src = src0 == nullptr ? nullptr : src0 + off;
        // The tile of a matrix of M past its size has nothing to compute.
        if (row0 >= M || col0 >= M)
            return;
        for (int q = 0; q < nsq; ++q)
        {
            A *dst = (((nsq - 1 - q) & 1) == 0 ? p : tmp) + off;
            const bool input = q == 0 && a_in != nullptr;
            // Entry (i, j), zero from M on.
            auto fetch = [&](int i, int j) -> A
            {
                if (i >= M || j >= M)
                    return num<A>(0);
                const int64_t at = static_cast<int64_t>(i) * M + j;
                return input ? to_acc(a_in[off + at]) : src[at];
            };
            __syncthreads(); // the strips are free again; the last squaring's writes are in
            for (int e = threadIdx.x; e < BT * BM; e += NT)
                As[e] = fetch(row0 + e / BM, e % BM);
            for (int e = threadIdx.x; e < BM * BT; e += NT)
                Bs[e] = fetch(e / BT, col0 + e % BT);
            __syncthreads();
            A acc = num<A>(0);
#pragma unroll 8
            for (int k = 0; k < M; ++k)
                acc = mad(As[ty * BM + k], Bs[k * BT + tx], acc);
            const int i = row0 + ty, j = col0 + tx;
            if (i < M && j < M)
                dst[static_cast<int64_t>(i) * M + j] = acc;
            src = dst;
        }
    }
} // namespace philtorch::recur_n
