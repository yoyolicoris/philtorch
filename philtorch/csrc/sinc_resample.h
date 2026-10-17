#pragma once

// Shared code of the native Kaiser-windowed sinc resampler
// (philtorch::resample): the math, the time grid and the plan of a call, and
// the argument checks and set-up that the CPU kernel (host_sinc_resample.cpp)
// and the CUDA kernel (sinc_resample.cu) have in common.
//
// The op computes, for integer rates orig_freq and new_freq reduced by their
// gcd to q and p (r = p / q), with inputs outside [0, T) taken as zero,
//
//   out[b, m] = scale * sum_n x[b, n] * g(c * (m q / p - n)),
//                                                    m = 0 .. out_len - 1,
//
// where c = p < q ? rolloff * p / q : rolloff, N = num_zeros, and g is the
// Kaiser-windowed sinc
//
//   g(z) = sinc(z) K(z),   sinc(z) = sin(pi z) / (pi z),
//   K(z) = I0(beta sqrt(1 - (z / N)^2)) / I0(beta)  for |z| < N,  else 0.
//
// g(z) = 0 for |z| >= N.  scale is folded into the weights; the op has no
// other prefactor (the caller multiplies by c if it wants the unit-gain
// filter).
//
// The window is evaluated from the power series of I0 in v = 1 - (z / N)^2,
//
//   K(z) = sum_{k < terms} a_k v^k,   a_k = (beta^2 / 4)^k / ((k!)^2 I0(beta)),
//
// by Horner's rule: only multiplies and adds, all terms positive, v in [0, 1].
// Every a_k is at most 1, so nothing overflows for any beta.  The coefficients
// are made once per call, on the host, in double, with scale folded in
// (make_kaiser_coeffs), and handed to the kernels by value.
//
// The caller of kaiser_sinc supplies sin(pi z), so that a kernel can get it
// from sin/cos(pi z0) and a table of sin/cos(pi c k) by angle addition,
// z_k = z0 - c k, instead of calling sin once per tap.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <numeric>
#include <type_traits>
#include <vector>

namespace philtorch {

// The most series terms the window may take, which limits beta: with the
// truncation rule of kaiser_term_count the limit is a beta of about 346 in
// double, about 398 in float.  The check of the op is the double one.
constexpr int kMaxKaiserTerms = 256;
// The product p q of the reduced rates, and the largest input position of an
// output, stay below this, so that the integer arithmetic of the time grid
// (i q, q f) cannot overflow int64.
constexpr int64_t kMaxRates = int64_t(1) << 62;
// The most elements of the table of the table path (p * taps): 8 MB in double,
// which stays in the last-level cache of a typical CPU.
constexpr int64_t kMaxTable = int64_t(1) << 20;
// The most taps of one output.  Tap indices and offsets are 32-bit integers in
// the vector loops of the CPU and in the CUDA kernels; this bound leaves room
// for the padding they add.
constexpr int64_t kMaxTaps = int64_t(1) << 30;
// |z| up to which sinc(z) is evaluated from its Taylor series.  Above it
// sin(pi z) / (pi z) has an absolute error of at most 1 / (pi kSincTaylorRange)
// = 1.3 times that of sin(pi z), at best an ulp of 1; below it that quotient
// would amplify the error without bound.
constexpr double kSincTaylorRange = 0.25;

namespace detail {

// c ? a : b.  On the host the select is made on the bits: with trapping math
// on, which is the default, GCC keeps a floating-point ternary whose arms are
// computed values as a branch, and no loop over it vectorises.
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T select_fp(bool c, T a, T b) {
#if defined(__CUDA_ARCH__)
    return c ? a : b;
#else
    if constexpr (std::is_same_v<T, float>) {
        uint32_t ua, ub;
        std::memcpy(&ua, &a, sizeof(ua));
        std::memcpy(&ub, &b, sizeof(ub));
        const uint32_t mk = uint32_t(0) - static_cast<uint32_t>(c);
        ub = (ua & mk) | (ub & ~mk);
        std::memcpy(&b, &ub, sizeof(b));
    } else {
        uint64_t ua, ub;
        std::memcpy(&ua, &a, sizeof(ua));
        std::memcpy(&ub, &b, sizeof(ub));
        const uint64_t mk = uint64_t(0) - static_cast<uint64_t>(c);
        ub = (ua & mk) | (ub & ~mk);
        std::memcpy(&b, &ub, sizeof(b));
    }
    return b;
#endif
}

// The number of terms N of the series sinc(z) = sum_n (-q)^n / (2n+1)!,
// q = (pi z)^2, that brings the first omitted one, at |z| = kSincTaylorRange,
// below eps / 4: 4 in float, 8 in double.
template <typename T>
constexpr int sinc_taylor_terms() {
    const double pz = std::numbers::pi * kSincTaylorRange;
    const double q = pz * pz;
    const double tol = 0.25 * static_cast<double>(std::numeric_limits<T>::epsilon());
    double term = 1.0;
    for (int n = 0;; ++n) {
        term *= q / (static_cast<double>(2 * n + 2) * static_cast<double>(2 * n + 3));
        if (term <= tol) {
            return n;
        }
    }
}

// The same as a constant, for the device code (a constexpr function of the host
// would draw a warning there).
template <typename T>
inline constexpr int kSincTaylorTerms = sinc_taylor_terms<T>();

// sinc(z) for |z| <= kSincTaylorRange, by the series in the nested form
// 1 - q/(2*3) (1 - q/(4*5) (1 - ...)).
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T sinc_taylor(T z) {
    const T pz = std::numbers::pi_v<T> * z;
    const T q = pz * pz;
    constexpr int terms = kSincTaylorTerms<T>;
    T s = static_cast<T>(1);
    for (int n = terms; n >= 1; --n) {
        const T den = static_cast<T>(2 * n) * static_cast<T>(2 * n + 1);
        s = static_cast<T>(1) - q / den * s;
    }
    return s;
}

// v = 1 - (z / N)^2, as (1 - u)(1 + u) with u = |z| / N, so that it is
// accurate near the edge.  Not clipped: the window is 0 where v <= 0.
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T window_v(T z, T inv_n) {
    const T u = std::abs(z) * inv_n;
    return (static_cast<T>(1) - u) * (static_cast<T>(1) + u);
}

// sinc(z) * window, 0 outside the window (v <= 0).  Every operand of a select
// is computed before it, and the quotient is never by 0: with trapping math
// on, GCC keeps arithmetic inside the arms of a select conditional, and then
// no loop over this vectorises.
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T sinc_gated(T z, T sin_pz, T v, T window) {
    const bool near0 = std::abs(z) < static_cast<T>(kSincTaylorRange);
    const T pz = std::numbers::pi_v<T> * z;
    const T quot = sin_pz / select_fp<T>(near0, static_cast<T>(1), pz);
    const T taylor = sinc_taylor<T>(z);
    const T sinc = select_fp<T>(near0, taylor, quot);
    return select_fp<T>(v > static_cast<T>(0), sinc * window, static_cast<T>(0));
}

// sin and cos of pi (hi + lo) in double, where hi + lo is a double double with
// |lo| << ulp(hi).  The reduction of hi modulo 2 is exact.
C10_HOST_DEVICE C10_ALWAYS_INLINE void sincos_pi_double(double hi, double lo, double &s,
                                            double &c) {
    const double r = hi - 2.0 * std::rint(hi * 0.5);
#if defined(__CUDA_ARCH__)
    sincospi(r + lo, &s, &c);
#else
    const double a = std::numbers::pi * (r + lo);
    s = std::sin(a);
    c = std::cos(a);
#endif
}

// sin and cos of pi (hi + lo) as T.  The CUDA file specialises it for float
// with sincospif, which double sincos is too slow to replace on most GPUs.
template <typename T>
struct SincosPi {
    C10_HOST_DEVICE C10_ALWAYS_INLINE static void apply(double hi, double lo, T &s, T &c) {
        double sd, cd;
        sincos_pi_double(hi, lo, sd, cd);
        s = static_cast<T>(sd);
        c = static_cast<T>(cd);
    }
};

}  // namespace detail

// sin and cos of pi c x as T, for double c and x: the product is taken as a
// double double, so that its rounding does not enter (|c x| is up to
// num_zeros).
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void sincos_pi_prod(double c, double x, T &s, T &co) {
    const double hi = c * x;
    const double lo = std::fma(c, x, -hi);
    detail::SincosPi<T>::apply(hi, lo, s, co);
}

// [host] The number of series terms of the window for beta in T: the terms
// k < K of t_k = (beta^2 / 4)^k / (k!)^2, t_k = t_{k-1} (beta^2 / 4) / k^2,
// that are kept, which end once past the peak (k^2 > beta^2 / 4, from which the
// ratio of consecutive terms is below 1 and decreasing) with the whole omitted
// tail below eps / 4 of the sum of the kept terms.  The next term t_K is the
// largest of the tail, and the ratios after it are at most q / (K+1)^2, so the
// tail is at most t_K / (1 - q / (K+1)^2).  Each a_k <= 1, so the truncation
// costs less than eps / 4 in K(z), whose maximum is 1.  0 when that takes more
// than kMaxKaiserTerms terms.
template <typename T>
inline int kaiser_term_count(double beta) {
    const double q = 0.25 * beta * beta;
    const double tol = 0.25 * static_cast<double>(std::numeric_limits<T>::epsilon());
    double t = 1.0, sum = 1.0;
    for (int k = 1;; ++k) {
        const double kd = static_cast<double>(k);
        const double next = t * q / (kd * kd);
        if (kd * kd > q &&
            next <= tol * sum * (1.0 - q / ((kd + 1.0) * (kd + 1.0)))) {
            return k;
        }
        if (k >= kMaxKaiserTerms) {
            return 0;
        }
        t = next;
        sum += next;
    }
}

// [host] The window's series coefficients, scale * a_k as T for k < terms,
// terms from kaiser_term_count.  I0(beta) is the sum of the t_k kept; the
// recurrence is evaluated twice, so that no array of the t_k is needed.
template <typename T>
inline std::vector<T> make_kaiser_coeffs(double beta, double scale) {
    const int terms = kaiser_term_count<T>(beta);
    const double q = 0.25 * beta * beta;
    double t = 1.0, sum = 1.0;
    for (int k = 1; k < terms; ++k) {
        const double kd = static_cast<double>(k);
        t = t * q / (kd * kd);
        sum += t;
    }
    std::vector<T> coef(terms);
    t = 1.0;
    coef[0] = static_cast<T>(scale * (t / sum));
    for (int k = 1; k < terms; ++k) {
        const double kd = static_cast<double>(k);
        t = t * q / (kd * kd);
        coef[k] = static_cast<T>(scale * (t / sum));
    }
    return coef;
}

// scale * g(z) for 1 / N = inv_n and the series coefficients coef[0 .. terms),
// by Horner's rule.  Needs sin_pz = sin(pi z) to within about an ulp of 1 in
// absolute value; it is not used near z = 0.
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T kaiser_sinc(T z, T sin_pz, T inv_n, const T *coef,
                                    int terms) {
    const T v = detail::window_v<T>(z, inv_n);
    const T vc = std::max(static_cast<T>(0), v);
    T k = coef[terms - 1];
    for (int j = terms - 2; j >= 0; --j) {
        k = k * vc + coef[j];
    }
    return detail::sinc_gated<T>(z, sin_pz, v, k);
}

// The time grid of a call.  The rates reduced by their gcd give r = p / q (p
// the new rate, q the original), and output m = p f + i reads the inputs
// q f + i q / p + (-S, S), with c = p < q ? rolloff * p / q : rolloff and
// S = N / c.  Everything about the grid is exact integer arithmetic except
// the fraction rem / p in [0, 1):
//
//   i q = w p + rem,   center = q f + w + rem / p,
//   nlo = q f + w + ceil(rem / p - S)       the first input of output m,
//   dc  = rem / p - ceil(rem / p - S)       center - nlo, in (S - 1, S],
//
// so that the weight of tap k, the input nlo + k, is scale * g(c (dc - k)).
// Requires p q <= kMaxRates and q (m / p + 1) <= kMaxRates (checked by
// resample_entry), so that i q and q f do not overflow.
struct OutputPhase {
    int64_t nlo;  // the first input of output m
    double dc;    // center - nlo
};

C10_HOST_DEVICE C10_ALWAYS_INLINE OutputPhase output_phase(int64_t m, int64_t p, int64_t q,
                                               double S) {
    const int64_t f = m / p;
    const int64_t i = m - f * p;
    const int64_t iq = i * q;
    const int64_t w = iq / p;
    const double frac = static_cast<double>(iq - w * p) / static_cast<double>(p);
    const double jl = std::ceil(frac - S);
    OutputPhase ph;
    ph.nlo = q * f + w + static_cast<int64_t>(jl);
    ph.dc = frac - jl;
    return ph;
}

// z_k = c (dc - k) of the taps of an output, as cc * ((dci - k) + frac) with
// dc = dci + frac, dci an integer: for small |z| in T this is accurate, where
// dc - k would be the difference of two rounded numbers.  Taps and dci are
// below kMaxTaps, so 32-bit integers (which convert to T in a vector).
template <typename T>
struct TapGrid {
    T cc;
    int32_t dci;
    T frac;

    C10_HOST_DEVICE C10_ALWAYS_INLINE T z(int32_t k) const {
        return cc * (static_cast<T>(dci - k) + frac);
    }
};

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE TapGrid<T> make_tap_grid(double c, double dc) {
    const double fl = std::floor(dc);
    return {static_cast<T>(c), static_cast<int32_t>(fl), static_cast<T>(dc - fl)};
}

// How a resample call is run, by the CPU and the CUDA kernel alike.  The reads
// of an output are the inputs nlo + k, k < taps.  With table true (p <= out_len
// and p * taps <= kMaxTable) the weights of output p f + i depend only on the
// phase i, so the kernel builds them once, as p rows of taps weights; the first
// input of row i is output_phase(i).nlo, and that of output p f + i is
// q f + output_phase(i).nlo.
struct ResamplePlan {
    bool table;
    int64_t p, q;
    double c, S, inv_n;  // inv_n = 1 / num_zeros
    int64_t taps;
};

// p = new_freq / gcd, q = orig_freq / gcd.
inline ResamplePlan make_resample_plan(int64_t p, int64_t q, double rolloff,
                                       int64_t num_zeros, int64_t out_len) {
    ResamplePlan plan;
    plan.p = p;
    plan.q = q;
    plan.c = p < q ? rolloff * static_cast<double>(p) / static_cast<double>(q)
                   : rolloff;
    plan.S = static_cast<double>(num_zeros) / plan.c;
    plan.inv_n = 1.0 / static_cast<double>(num_zeros);
    // The window is open, |dc - k| < S: its taps are the k in (dc - S, dc + S),
    // at most floor(2 S) + 1 of them since dc is in (S - 1, S].  One more is a
    // guard against a rounding of ceil(rem / p - S) at the edge; the weight of
    // a tap outside the window is 0, so a spare tap does no harm.
    const double taps = std::floor(2.0 * plan.S) + 2.0;
    plan.taps = taps < static_cast<double>(std::numeric_limits<int64_t>::max())
                    ? static_cast<int64_t>(taps)
                    : std::numeric_limits<int64_t>::max();
    plan.table = p <= out_len && p <= kMaxTable / plan.taps;
    return plan;
}

// The op of both back ends, apart from the kernels: checks the arguments
// (ValueError), handles the empty cases, makes the plan, converts x to the
// compute type (double for double, float for float, half and bfloat16) and
// calls run(std::type_identity<T>{}, x, out, plan) with the contiguous x of
// that type (B, T) and an out (B, out_len) of it to fill; returns out in the
// dtype of x.
template <typename Run>
at::Tensor resample_entry(const at::Tensor &x, int64_t orig_freq,
                          int64_t new_freq, double scale, double rolloff,
                          double beta, int64_t num_zeros, int64_t out_len,
                          Run &&run) {
    TORCH_CHECK_VALUE(x.dim() == 2, "x must be 2D (B, T), got ", x.dim(), "D");
    TORCH_CHECK_VALUE(x.is_floating_point(),
                      "x must be a floating-point tensor, got ", x.scalar_type());
    TORCH_CHECK_VALUE(orig_freq >= 1,
                      "orig_freq must be a positive integer, got ", orig_freq);
    TORCH_CHECK_VALUE(new_freq >= 1,
                      "new_freq must be a positive integer, got ", new_freq);
    TORCH_CHECK_VALUE(std::isfinite(scale), "scale must be finite, got ", scale);
    TORCH_CHECK_VALUE(rolloff > 0.0 && rolloff <= 1.0,
                      "rolloff must be in (0, 1], got ", rolloff);
    TORCH_CHECK_VALUE(!(beta < 0.0), "beta must be non-negative, got ", beta);
    TORCH_CHECK_VALUE(std::isfinite(beta), "beta must be finite, got ", beta);
    TORCH_CHECK_VALUE(
        kaiser_term_count<double>(beta) > 0, "beta = ", beta,
        " is too large: the Kaiser window needs more than ", kMaxKaiserTerms,
        " series terms");
    TORCH_CHECK_VALUE(num_zeros >= 1,
                      "num_zeros must be a positive integer, got ", num_zeros);
    TORCH_CHECK_VALUE(out_len >= 0,
                      "out_len must be a non-negative integer, got ", out_len);

    const int64_t g = std::gcd(orig_freq, new_freq);
    const int64_t p = new_freq / g, q = orig_freq / g;
    constexpr int kRatesBits = std::bit_width(static_cast<uint64_t>(kMaxRates)) - 1;
    TORCH_CHECK_VALUE(p <= kMaxRates / q, "orig_freq ", orig_freq,
                      " and new_freq ", new_freq,
                      " are too large: the product of the reduced rates exceeds 2^",
                      kRatesBits);
    // The input positions q f of the outputs, f < ceil(out_len / p).
    TORCH_CHECK_VALUE(at::ceil_div(out_len, p) <= kMaxRates / q, "out_len ",
                      out_len, " is too large for orig_freq ", orig_freq,
                      " and new_freq ", new_freq,
                      ": the input positions would exceed 2^", kRatesBits);
    const ResamplePlan plan =
        make_resample_plan(p, q, rolloff, num_zeros, out_len);
    TORCH_CHECK_VALUE(plan.taps <= kMaxTaps, "the filter needs ", plan.taps,
                      " taps for these rates, rolloff and num_zeros, more than ",
                      kMaxTaps);

    const int64_t B = x.size(0), T = x.size(1);
    if (B == 0 || out_len == 0) {
        return at::empty({B, out_len}, x.options());
    }
    if (T == 0) {
        return at::zeros({B, out_len}, x.options());
    }
    const auto acc_type =
        x.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
    const at::Tensor x_acc = x.to(acc_type).contiguous();
    at::Tensor out = at::empty({B, out_len}, x_acc.options());
    AT_DISPATCH_FLOATING_TYPES(acc_type, "philtorch::resample", [&] {
        run(std::type_identity<scalar_t>{}, x_acc, out, plan);
    });
    return out.to(x.scalar_type());
}

}  // namespace philtorch
