#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

// The recurrence h[t] = A[t] h[t - 1] + x[t] on CPU. A sequence is split over
// time into chunks, when there are fewer sequences than threads: each chunk's
// transition matrix and local end state give its true incoming state, and the
// chunks then run in parallel again from it.
constexpr int64_t min_chunk = 1024;

template <typename scalar_t, typename acc_t>
inline acc_t host_dot(const scalar_t *left, const acc_t *right, int size)
{
    constexpr int minimum_simd_size = 8;
    acc_t sum = 0;
    if constexpr (std::is_same_v<acc_t, float> ||
                  std::is_same_v<acc_t, double>)
    {
        if (size >= minimum_simd_size)
        {
#pragma omp simd reduction(+ : sum)
            for (int index = 0; index < size; ++index)
                sum += left[index] * right[index];
            return sum;
        }
    }
    for (int index = 0; index < size; ++index)
        sum += left[index] * right[index];
    return sum;
}

// Whether every entry is below the smallest normal number (zero for integers).
// For a constant A, the remaining factors of A^L are powers of a tiny matrix,
// and the rest of the product would go on in denormals, which are slow, and
// contribute below rounding.
template <typename acc_t>
bool tiny(const acc_t *v, int64_t n)
{
    for (int64_t i = 0; i < n; ++i)
    {
        if constexpr (std::is_integral_v<acc_t>)
        {
            if (v[i] != 0)
                return false;
        }
        else
        {
            using real = typename c10::scalar_value_type<acc_t>::type;
            if (!(std::abs(v[i]) < std::numeric_limits<real>::min()))
                return false;
        }
    }
    return true;
}

// Whether every entry is finite (integers wrap, and always are).
template <typename acc_t>
bool finite(const acc_t *v, int64_t n)
{
    if constexpr (!std::is_integral_v<acc_t>)
        for (int64_t i = 0; i < n; ++i)
            if (!std::isfinite(std::abs(v[i])))
                return false;
    return true;
}

// A_b points at the matrix of step 1, A_b + (t - 1) * stride_t at step t's.
// Steps s to e - 1 of a sequence from the state h, which is updated: H_b[t]
// holds x[t] and gets h[t] if write. M is an int, which makes the loop faster.
template <bool write, typename scalar_t, typename acc_t>
void run_steps(const scalar_t *A_b, int64_t stride_t, scalar_t *H_b,
               int64_t s, int64_t e, int M, acc_t *h, acc_t *tmp)
{
    if (M == 1)
    {
        acc_t v = h[0];
        for (int64_t t = s; t < e; ++t)
        {
            v = static_cast<acc_t>(H_b[t]) +
                static_cast<acc_t>(A_b[(t - 1) * stride_t]) * v;
            if constexpr (write)
                H_b[t] = v;
        }
        h[0] = v;
        return;
    }
    acc_t *prev = h, *curr = tmp;
    for (int64_t t = s; t < e; ++t)
    {
        const scalar_t *A_t = A_b + (t - 1) * stride_t;
        scalar_t *H_t = H_b + t * M;
        for (int i = 0; i < M; ++i)
        {
            curr[i] = static_cast<acc_t>(H_t[i]) + host_dot(A_t + i * M, prev, M);
            if constexpr (write)
                H_t[i] = curr[i];
        }
        std::swap(prev, curr);
    }
    if (prev != h)
        std::copy(prev, prev + M, h);
}

// out = X Y for row-major M x M matrices.
template <typename x_t, typename acc_t>
void matmul(const x_t *X, const acc_t *Y, acc_t *out, int M)
{
    std::fill(out, out + M * M, acc_t(0));
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < M; ++k)
            for (int j = 0; j < M; ++j)
                out[i * M + j] += X[i * M + k] * Y[k * M + j];
}

// Phi = A[e - 1] ... A[s], row-major. A constant A (stride_t == 0) gives
// A^(e - s), by binary exponentiation.
template <typename scalar_t, typename acc_t>
void transition(const scalar_t *A_b, int64_t stride_t, int64_t s, int64_t e,
                int M, acc_t *Phi)
{
    std::vector<acc_t> R(M * M, acc_t(0)), W(M * M);
    for (int i = 0; i < M; ++i)
        R[i * M + i] = 1;
    if (stride_t == 0)
    {
        std::vector<acc_t> power(A_b, A_b + M * M);
        for (int64_t L = e - s; L; L >>= 1)
        {
            if (L & 1)
            {
                matmul(R.data(), power.data(), W.data(), M);
                std::swap(R, W);
            }
            if (L > 1)
            {
                matmul(power.data(), power.data(), W.data(), M);
                std::swap(power, W);
                if (tiny(power.data(), M * M)) // the remaining factor, a power
                {                              // of this one, is tiny
                    std::fill(R.begin(), R.end(), acc_t(0));
                    break;
                }
            }
        }
    }
    else
    {
        for (int64_t t = s; t < e; ++t)
        {
            matmul(A_b + (t - 1) * stride_t, R.data(), W.data(), M);
            std::swap(R, W);
            // A zero product stays zero; a tiny one may grow again.
            if (std::all_of(R.begin(), R.end(), [](acc_t v) { return v == acc_t(0); }))
                break;
        }
    }
    std::copy(R.begin(), R.end(), Phi);
}

// A[b, t] is the M x M row-major matrix at A + b * stride_b + (t - 1) * stride_t
// for t = 1 to T - 1. H is [B][T][M], in place: H[b][0] is the initial state,
// H[b][t] holds x[t] on entry and h[t] on exit.
template <typename scalar_t>
void chunked_recurrence(const scalar_t *A, int64_t stride_b, int64_t stride_t,
                        scalar_t *H, int64_t B, int64_t T, int M)
{
    using acc_t = at::opmath_type<scalar_t>;
    const int64_t steps = T - 1;
    const int64_t P = at::get_num_threads();
    int64_t C = B > 0 ? std::min(P / B, steps / min_chunk) : 1;
    // A constant A costs 2 passes over the steps, a varying one M + 2.
    if (C <= (stride_t == 0 ? 2 : M + 2))
        C = 1;

    // Chunk c is steps [start(c), start(c + 1)).
    auto start = [&](int64_t c)
    { return 1 + c * steps / C; };
    // Phi[b][c - 1] is chunk c's, for c >= 1.
    std::vector<acc_t> Phi(B * (C - 1) * M * M), hin(B * C * M);
    std::vector<char> unsplit(B, 0);

    // Local end states, and Phi; chunk 0 is the true recurrence, and writes.
    at::parallel_for(0, B * C, 1, [&](int64_t begin, int64_t end)
                     {
        std::vector<acc_t> h(M), tmp(M);
        for (int64_t k = begin; k < end; ++k)
        {
            int64_t b = k / C, c = k % C;
            const scalar_t *A_b = A + b * stride_b;
            scalar_t *H_b = H + b * T * M;
            if (c == 0)
            {
                std::copy(H_b, H_b + M, h.begin());
                run_steps<true>(A_b, stride_t, H_b, start(0), start(1), M,
                                h.data(), tmp.data());
            }
            else
            {
                std::fill(h.begin(), h.end(), acc_t(0));
                run_steps<false>(A_b, stride_t, H_b, start(c), start(c + 1), M,
                                 h.data(), tmp.data());
                transition(A_b, stride_t, start(c), start(c + 1), M,
                           Phi.data() + (b * (C - 1) + c - 1) * M * M);
            }
            std::copy(h.begin(), h.end(), hin.begin() + k * M);
        } });
    if (C == 1)
        return;

    // The true end states over the chunks. hin[b][c] holds the local end state
    // on entry and the incoming state on exit. A sequence with a non-finite
    // Phi, whose product with a zero would be NaN, is left unsplit.
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end)
                     {
        std::vector<acc_t> prev(M), next(M);
        for (int64_t b = begin; b < end; ++b)
        {
            acc_t *hin_b = hin.data() + b * C * M;
            std::copy(hin_b, hin_b + M, prev.begin());
            for (int64_t c = 1; c < C; ++c)
            {
                acc_t *local = hin_b + c * M;
                const acc_t *Phi_c = Phi.data() + (b * (C - 1) + c - 1) * M * M;
                std::copy(local, local + M, next.begin());
                std::copy(prev.begin(), prev.end(), local);
                if (!finite(Phi_c, M * M))
                {
                    unsplit[b] = 1;
                    break;
                }
                for (int i = 0; i < M; ++i)
                    next[i] += host_dot(Phi_c + i * M, prev.data(), M);
                std::swap(prev, next);
            }
        } });

    // The chunks c >= 1 again, from their incoming states, writing; an unsplit
    // sequence from chunk 1 on, whose hin[b][1] is chunk 0's end state.
    at::parallel_for(0, B * (C - 1), 1, [&](int64_t begin, int64_t end)
                     {
        std::vector<acc_t> h(M), tmp(M);
        for (int64_t k = begin; k < end; ++k)
        {
            int64_t b = k / (C - 1), c = k % (C - 1) + 1;
            if (unsplit[b] && c > 1)
                continue;
            std::copy_n(hin.begin() + (b * C + c) * M, M, h.begin());
            run_steps<true>(A + b * stride_b, stride_t, H + b * T * M,
                            start(c), unsplit[b] ? T : start(c + 1), M,
                            h.data(), tmp.data());
        } });
}

// x is [B][T][M], or [B][T] with M = 1.
at::Tensor recur_cpu_launch(const at::Tensor &A, const at::Tensor &zi,
                            const at::Tensor &x, int64_t M, int64_t stride_b,
                            int64_t stride_t)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    auto A_contiguous = A.contiguous();
    auto out = at::cat({zi.unsqueeze(1), x}, 1).contiguous();
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
        at::kLong, at::kHalf, at::kBFloat16, x.scalar_type(),
        "chunked_recurrence", [&]
        { chunked_recurrence<scalar_t>(
              A_contiguous.const_data_ptr<scalar_t>(), stride_b, stride_t,
              out.mutable_data_ptr<scalar_t>(), out.size(0), out.size(1),
              static_cast<int>(M)); });
    return out.slice(1, 1, out.size(1))
        .contiguous(); // Remove the initial state from the output
}

at::Tensor lti_recur_cpu_impl(const at::Tensor &a,
                              const at::Tensor &zi, const at::Tensor &x)
{
    TORCH_CHECK(a.dim() <= 1, "A must be a vector or a scalar");
    TORCH_CHECK(zi.dim() == 1, "zi must be a vector");
    TORCH_CHECK(x.size(1) > 0, "x must contain at least one time step");
    bool per_item = a.dim() == 1 && a.numel() == x.size(0);
    TORCH_CHECK(per_item || a.numel() == 1, "A must be a scalar or (B,)");
    return recur_cpu_launch(a, zi, x, 1, per_item ? 1 : 0, 0);
}

at::Tensor lti_mat_recur_N_order_cpu_impl(const at::Tensor &A,
                                          const at::Tensor &zi,
                                          const at::Tensor &x)
{
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be a 2D or 3D tensor");
    TORCH_CHECK(x.size(2) == A.size(-1),
                "Last dimension of x must match last dimension of A");
    TORCH_CHECK(A.size(-1) == A.size(-2),
                "Last two dimensions of A must be equal");
    TORCH_CHECK(A.dim() == 2 || A.size(0) == x.size(0),
                "A must have the batch size of x");
    auto order = A.size(-1);
    return recur_cpu_launch(A, zi, x, order, A.dim() == 3 ? order * order : 0, 0);
}

at::Tensor mat_recur_N_order_cpu_impl(const at::Tensor &A, const at::Tensor &zi,
                                      const at::Tensor &x)
{
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    TORCH_CHECK(x.size(2) == A.size(-1),
                "Last dimension of x must match last dimension of A");
    TORCH_CHECK(A.size(-1) == A.size(-2),
                "Last two dimensions of A must be equal");
    TORCH_CHECK(A.size(-3) == x.size(1),
                "A must have one matrix per time step of x");
    TORCH_CHECK(A.dim() == 3 || A.size(0) == x.size(0),
                "A must have the batch size of x");
    auto order = A.size(-1);
    return recur_cpu_launch(A, zi, x, order,
                            A.dim() == 4 ? x.size(1) * order * order : 0,
                            order * order);
}

TORCH_LIBRARY_IMPL(philtorch, CPU, m)
{
    m.impl("lti_recur", &lti_recur_cpu_impl);
    m.impl("lti_recurN", &lti_mat_recur_N_order_cpu_impl);
    m.impl("lti_recur2", &lti_mat_recur_N_order_cpu_impl);
    m.impl("recurN", &mat_recur_N_order_cpu_impl);
    m.impl("recur2", &mat_recur_N_order_cpu_impl);
}
