#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ceil_div.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "sinc_resample.h"

// CPU kernel of philtorch::resample (see sinc_resample.h for the definition, the
// time grid and the plan).
//
// Both paths first build the weights of one output, then reuse them for every
// batch row.  The weights of output m, at the taps n = nlo + k, k < taps, are
// scale * g(c * (dc - k)) with (nlo, dc) = output_phase(m).  The kernels get
// sin(pi z_k) without calling sin per tap, from the angle addition
// z_k = z_0 - c k with sin/cos(pi c k) tabulated once per call.
//
// With the plan's table, the weights of output p f + i depend on the phase i
// alone, so the kernel builds a table of them, p rows of taps weights, in
// double, rounded once to the compute type.  Otherwise every output builds its
// own weights, in the compute type.

// Run-time SIMD dispatch.  The extension is built for the baseline ISA (SSE2 on
// x86-64), so the `#pragma omp simd` loops would run at 4 floats.  The functions
// holding them, the workers below, are multiversioned instead: the compiler
// emits a clone per ISA (the code of the always-inlined header maths and of
// WeightGen::fill is compiled into each) and the dynamic loader picks the best
// one for the running CPU, once, through an ifunc resolver.  Only the workers
// are cloned, so everything they call that must be vectorised is inlined into
// them (C10_ALWAYS_INLINE), and they are called by chunk, from the lambdas of
// at::parallel_for, which are compiled for the baseline.  The clones are
// x86-64-v3 (AVX2, FMA, BMI, ...) and x86-64-v4 (AVX-512 F, BW, CD, DQ, VL); a
// CPU with neither runs the "default" clone.  The `arch=` names of CPU models
// such as "haswell" would not do: the resolver then tests for that exact model.
// Only GCC 12 or later on glibc/x86-64 is multiversioned (v3 and v4 names, and
// ifunc); everywhere else the macro is empty and the workers are plain
// functions.  Define PHILTORCH_NO_CLONES to switch it off.
#if defined(__x86_64__) && defined(__linux__) && defined(__GLIBC__) && \
    defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12 &&      \
    !defined(PHILTORCH_NO_CLONES)
#define PHILTORCH_CLONES                                       \
    __attribute__((target_clones("default", "arch=x86-64-v3", \
                                 "arch=x86-64-v4")))
#endif
#ifndef PHILTORCH_CLONES
#define PHILTORCH_CLONES
#endif

namespace {

using philtorch::ResamplePlan;

template <typename AccT>
struct WeightGen {
    // Taps are evaluated in blocks of kTapBlock, so that the Horner loop of the
    // window runs over a block as one vector operation per coefficient: 16 was
    // the best (or tied) of 4, 8, 16, 32 in float and double.
    static constexpr int64_t kTapBlock = 16;

    double c;
    AccT inv_n;
    std::vector<AccT> coef;    // scale * Kaiser series coefficients
    std::vector<AccT> sk, ck;  // sin / cos (pi c k), k < taps + kTapBlock

    WeightGen(const ResamplePlan &plan, std::vector<AccT> coeffs)
        : c(plan.c),
          inv_n(static_cast<AccT>(plan.inv_n)),
          coef(std::move(coeffs)),
          sk(plan.taps + kTapBlock),
          ck(plan.taps + kTapBlock) {
        for (int64_t k = 0; k < plan.taps; ++k) {
            philtorch::sincos_pi_prod(c, static_cast<double>(k), sk[k], ck[k]);
        }
    }

    // Multiply-adds that one weight costs, roughly: a Horner step per term and
    // about as many for the rest.
    int64_t weight_cost() const { return 2 * static_cast<int64_t>(coef.size()); }

    // w[k] = scale * g(c * (dc - k)) for k in [k0, k1).
    C10_ALWAYS_INLINE void fill(AccT *w, int64_t k0, int64_t k1, double dc) const {
        AccT s0, c0;
        philtorch::sincos_pi_prod(c, dc, s0, c0);
        const auto grid = philtorch::make_tap_grid<AccT>(c, dc);
        const AccT *skp = sk.data();
        const AccT *ckp = ck.data();
        const AccT in = inv_n;
        const AccT *cf = coef.data();
        const int terms = static_cast<int>(coef.size());
        for (int64_t kb = k0; kb < k1; kb += kTapBlock) {
            AccT z[kTapBlock], v[kTapBlock], vc[kTapBlock], acc[kTapBlock],
                wb[kTapBlock];
#pragma omp simd
            for (int64_t j = 0; j < kTapBlock; ++j) {
                z[j] = grid.z(static_cast<int32_t>(kb + j));
                v[j] = philtorch::detail::window_v<AccT>(z[j], in);
                vc[j] = std::max(static_cast<AccT>(0), v[j]);
                acc[j] = cf[terms - 1];
            }
            for (int t = terms - 2; t >= 0; --t) {
                const AccT at = cf[t];
#pragma omp simd
                for (int64_t j = 0; j < kTapBlock; ++j) {
                    acc[j] = acc[j] * vc[j] + at;
                }
            }
#pragma omp simd
            for (int64_t j = 0; j < kTapBlock; ++j) {
                const AccT s = s0 * ckp[kb + j] - c0 * skp[kb + j];
                wb[j] = philtorch::detail::sinc_gated<AccT>(z[j], s, v[j], acc[j]);
            }
            const int64_t n = std::min<int64_t>(kTapBlock, k1 - kb);
            for (int64_t j = 0; j < n; ++j) {
                w[kb + j] = wb[j];
            }
        }
    }
};

// The dot product runs in vector lanes: four vectors of 256 bits worth of
// elements at a time (four independent accumulator registers on AVX2, so that
// the adds of one lane do not wait for the previous iteration's), then one such
// vector at a time, then the few elements left.  The lanes are summed in a tree
// at the end.
template <typename AccT>
C10_ALWAYS_INLINE AccT dot_body(const AccT *w, const AccT *x, int64_t n) {
    constexpr int64_t kVecBytes = 32;  // one AVX2 register
    constexpr int64_t kVecsInFlight = 4;
    constexpr int64_t kVec = kVecBytes / sizeof(AccT);
    constexpr int64_t kWide = kVecsInFlight * kVec;
    AccT acc[kWide];
    for (int64_t j = 0; j < kWide; ++j) {
        acc[j] = static_cast<AccT>(0);
    }
    int64_t i = 0;
    for (; i + kWide <= n; i += kWide) {
#pragma omp simd
        for (int64_t j = 0; j < kWide; ++j) {
            acc[j] += w[i + j] * x[i + j];
        }
    }
    for (; i + kVec <= n; i += kVec) {
#pragma omp simd
        for (int64_t j = 0; j < kVec; ++j) {
            acc[j] += w[i + j] * x[i + j];
        }
    }
    AccT tail = static_cast<AccT>(0);
    for (; i < n; ++i) {
        tail += w[i] * x[i];
    }
    for (int64_t h = kWide / 2; h >= 1; h /= 2) {
#pragma omp simd
        for (int64_t j = 0; j < h; ++j) {
            acc[j] += acc[j + h];
        }
    }
    return acc[0] + tail;
}

// Cloned functions of their own, not inlined into the workers: inlined, the
// reduction loop compiles to scalar code.
PHILTORCH_CLONES C10_NOINLINE float dot(const float *w, const float *x,
                                        int64_t n) {
    return dot_body<float>(w, x, n);
}
PHILTORCH_CLONES C10_NOINLINE double dot(const double *w, const double *x,
                                         int64_t n) {
    return dot_body<double>(w, x, n);
}

// Weights of the phases [i0, i1) into the table, and their first input offsets.
template <typename AccT>
C10_ALWAYS_INLINE void table_rows_body(const WeightGen<double> &gen,
                                      const ResamplePlan &plan, AccT *table,
                                      int64_t *phase_nlo, double *row,
                                      int64_t i0, int64_t i1) {
    const int64_t taps = plan.taps;
    for (int64_t i = i0; i < i1; ++i) {
        const auto ph = philtorch::output_phase(i, plan.p, plan.q, plan.S);
        phase_nlo[i] = ph.nlo;
        gen.fill(row, 0, taps, ph.dc);
        AccT *t = table + i * taps;
        for (int64_t k = 0; k < taps; ++k) {
            t[k] = static_cast<AccT>(row[k]);
        }
    }
}

// The outputs of the frames [begin, end) of the rows of a (B, F) grid.
template <typename AccT>
C10_ALWAYS_INLINE void table_apply_body(const AccT *x, AccT *out, int64_t T,
                                       int64_t out_len, int64_t F,
                                       const ResamplePlan &plan,
                                       const AccT *table,
                                       const int64_t *phase_nlo, int64_t begin,
                                       int64_t end) {
    const int64_t p = plan.p, q = plan.q, taps = plan.taps;
    for (int64_t idx = begin; idx < end; ++idx) {
        const int64_t b = idx / F, f = idx % F;
        const AccT *xb = x + b * T;
        AccT *ob = out + b * out_len + p * f;
        const int64_t n_i = std::min<int64_t>(p, out_len - p * f);
        for (int64_t i = 0; i < n_i; ++i) {
            const int64_t base = q * f + phase_nlo[i];
            const int64_t kmin = std::max<int64_t>(0, -base);
            const int64_t kmax = std::min<int64_t>(taps, T - base);
            ob[i] = kmin < kmax ? dot(table + i * taps + kmin, xb + base + kmin,
                                      kmax - kmin)
                                : static_cast<AccT>(0);
        }
    }
}

// The outputs [m0, m1) of every row, each with its own weights.
template <typename AccT>
C10_ALWAYS_INLINE void general_body(const AccT *x, AccT *out, int64_t B,
                                   int64_t T, int64_t out_len,
                                   const ResamplePlan &plan,
                                   const WeightGen<AccT> &gen, AccT *w,
                                   int64_t m0, int64_t m1) {
    const int64_t taps = plan.taps;
    for (int64_t m = m0; m < m1; ++m) {
        const auto ph = philtorch::output_phase(m, plan.p, plan.q, plan.S);
        const int64_t kmin = std::max<int64_t>(0, -ph.nlo);
        const int64_t kmax = std::min<int64_t>(taps, T - ph.nlo);
        if (kmin >= kmax) {
            for (int64_t b = 0; b < B; ++b) {
                out[b * out_len + m] = static_cast<AccT>(0);
            }
            continue;
        }
        gen.fill(w, kmin, kmax, ph.dc);
        for (int64_t b = 0; b < B; ++b) {
            out[b * out_len + m] =
                dot(w + kmin, x + b * T + ph.nlo + kmin, kmax - kmin);
        }
    }
}

// The workers: one non-template function per compute type, cloned per ISA.
#define PHILTORCH_WORKERS(ACC)                                                 \
    PHILTORCH_CLONES void table_rows(                                          \
        const WeightGen<double> &gen, const ResamplePlan &plan, ACC *table,    \
        int64_t *phase_nlo, double *row, int64_t i0, int64_t i1) {             \
        table_rows_body<ACC>(gen, plan, table, phase_nlo, row, i0, i1);        \
    }                                                                          \
    PHILTORCH_CLONES void table_apply(                                         \
        const ACC *x, ACC *out, int64_t T, int64_t out_len, int64_t F,         \
        const ResamplePlan &plan, const ACC *table, const int64_t *phase_nlo,  \
        int64_t begin, int64_t end) {                                          \
        table_apply_body<ACC>(x, out, T, out_len, F, plan, table, phase_nlo,   \
                              begin, end);                                     \
    }                                                                          \
    PHILTORCH_CLONES void general_rows(                                        \
        const ACC *x, ACC *out, int64_t B, int64_t T, int64_t out_len,         \
        const ResamplePlan &plan, const WeightGen<ACC> &gen, ACC *w,           \
        int64_t m0, int64_t m1) {                                              \
        general_body<ACC>(x, out, B, T, out_len, plan, gen, w, m0, m1);        \
    }
PHILTORCH_WORKERS(float)
PHILTORCH_WORKERS(double)
#undef PHILTORCH_WORKERS

// The chunk size of at::parallel_for for items of work_a * work_b multiply-adds
// (a division chain: the product could overflow), about at::internal::GRAIN_SIZE
// of them per chunk, so that a chunk outweighs the cost of dispatching it.
int64_t grain(int64_t work_a, int64_t work_b) {
    return std::max<int64_t>(1, at::internal::GRAIN_SIZE / work_a / work_b);
}

template <typename AccT>
void resample_table(const AccT *x, AccT *out, int64_t B, int64_t T,
                    int64_t out_len, const ResamplePlan &plan,
                    const WeightGen<double> &gen) {
    const int64_t p = plan.p, taps = plan.taps;
    // The weights are evaluated in double and rounded to AccT once.
    std::vector<AccT> table(p * taps);
    std::vector<int64_t> phase_nlo(p);
    at::parallel_for(0, p, grain(taps, gen.weight_cost()),
                     [&](int64_t i0, int64_t i1) {
                         std::vector<double> row(taps);
                         table_rows(gen, plan, table.data(), phase_nlo.data(),
                                    row.data(), i0, i1);
                     });

    const int64_t F = at::ceil_div(out_len, p);
    at::parallel_for(0, B * F, grain(p, taps), [&](int64_t begin, int64_t end) {
        table_apply(x, out, T, out_len, F, plan, table.data(),
                    phase_nlo.data(), begin, end);
    });
}

template <typename AccT>
void resample_general(const AccT *x, AccT *out, int64_t B, int64_t T,
                      int64_t out_len, const ResamplePlan &plan,
                      const WeightGen<AccT> &gen) {
    const int64_t taps = plan.taps;
    at::parallel_for(0, out_len, grain(taps, gen.weight_cost() + B),
                     [&](int64_t m0, int64_t m1) {
                         std::vector<AccT> w(taps);
                         general_rows(x, out, B, T, out_len, plan, gen, w.data(),
                                      m0, m1);
                     });
}

template <typename AccT>
void resample_impl(const AccT *x, AccT *out, int64_t B, int64_t T,
                   int64_t out_len, double beta, double scale,
                   const ResamplePlan &plan) {
    if (plan.table) {
        const WeightGen<double> gen(
            plan, philtorch::make_kaiser_coeffs<double>(beta, scale));
        resample_table<AccT>(x, out, B, T, out_len, plan, gen);
    } else {
        const WeightGen<AccT> gen(
            plan, philtorch::make_kaiser_coeffs<AccT>(beta, scale));
        resample_general<AccT>(x, out, B, T, out_len, plan, gen);
    }
}

}  // namespace

at::Tensor sinc_resample_cpu_impl(const at::Tensor &x, int64_t orig_freq,
                                  int64_t new_freq, double scale,
                                  double rolloff, double beta,
                                  int64_t num_zeros, int64_t out_len) {
    return philtorch::resample_entry(
        x, orig_freq, new_freq, scale, rolloff, beta, num_zeros, out_len,
        [&](auto type, const at::Tensor &xa, at::Tensor &out,
            const ResamplePlan &plan) {
            using AccT = typename decltype(type)::type;
            resample_impl<AccT>(xa.const_data_ptr<AccT>(),
                                out.mutable_data_ptr<AccT>(), xa.size(0),
                                xa.size(1), out.size(1), beta, scale, plan);
        });
}

TORCH_LIBRARY_FRAGMENT(philtorch, m) {
    m.def(
        "philtorch::resample(Tensor x, int orig_freq, int new_freq, float scale, "
        "float rolloff, float beta, int num_zeros, SymInt out_len) -> Tensor",
        {at::Tag::pt2_compliant_tag});
}

TORCH_LIBRARY_IMPL(philtorch, CPU, m) {
    m.impl("resample", &sinc_resample_cpu_impl);
}
