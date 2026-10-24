#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <torch/script.h>
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

// Whether every entry is below the smallest normal number (zero for integers):
// products of contractions go on with denormals, which are slow, and no more.
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

// A_b points at the matrix of step 1, A_b + (t - 1) * stride_t at step t's.
// Steps s to e - 1 of a sequence from the state h, which is updated: H_b[t]
// holds x[t] and gets h[t] if write.
template <bool write, typename scalar_t, typename acc_t>
void run_steps(const scalar_t *A_b, int64_t stride_t, scalar_t *H_b,
               int64_t s, int64_t e, int N, acc_t *h, acc_t *tmp)
{
    if (N == 1)
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
        scalar_t *H_t = H_b + t * N;
        for (int i = 0; i < N; ++i)
        {
            curr[i] = static_cast<acc_t>(H_t[i]) + host_dot(A_t + i * N, prev, N);
            if constexpr (write)
                H_t[i] = curr[i];
        }
        std::swap(prev, curr);
    }
    if (prev != h)
        std::copy(prev, prev + N, h);
}

// out = X Y for row-major N x N matrices.
template <typename acc_t>
void matmul(const acc_t *X, const acc_t *Y, acc_t *out, int64_t N)
{
    std::fill(out, out + N * N, acc_t(0));
    for (int64_t i = 0; i < N; ++i)
        for (int64_t k = 0; k < N; ++k)
            for (int64_t j = 0; j < N; ++j)
                out[i * N + j] += X[i * N + k] * Y[k * N + j];
}

// Phi = A[e - 1] ... A[s], stored by columns: PhiT[j * N + i] = Phi[i][j]. A
// constant A (stride_t == 0) gives A^(e - s), by binary exponentiation.
template <typename scalar_t, typename acc_t>
void transition(const scalar_t *A_b, int64_t stride_t, int64_t s, int64_t e,
                int64_t N, acc_t *PhiT, acc_t *tmp)
{
    if (stride_t == 0)
    {
        std::vector<acc_t> R(N * N, acc_t(0)), M(A_b, A_b + N * N), W(N * N);
        for (int64_t i = 0; i < N; ++i)
            R[i * N + i] = 1;
        for (int64_t L = e - s; L; L >>= 1)
        {
            if (L & 1)
            {
                matmul(R.data(), M.data(), W.data(), N);
                std::swap(R, W);
            }
            if (L > 1)
            {
                matmul(M.data(), M.data(), W.data(), N);
                std::swap(M, W);
                if (tiny(M.data(), N * N)) // the rest of the product is tiny
                {
                    std::fill(R.begin(), R.end(), acc_t(0));
                    break;
                }
            }
        }
        for (int64_t i = 0; i < N; ++i)
            for (int64_t j = 0; j < N; ++j)
                PhiT[j * N + i] = R[i * N + j];
        return;
    }
    if (N == 1)
    {
        acc_t p = 1;
        for (int64_t t = s; t < e; ++t)
        {
            p = static_cast<acc_t>(A_b[(t - 1) * stride_t]) * p;
            if (tiny(&p, 1))
            {
                p = 0;
                break;
            }
        }
        PhiT[0] = p;
        return;
    }
    for (int64_t i = 0; i < N; ++i)
        for (int64_t j = 0; j < N; ++j)
            PhiT[j * N + i] = i == j ? 1 : 0;
    for (int64_t t = s; t < e; ++t)
    {
        const scalar_t *A_t = A_b + (t - 1) * stride_t;
        for (int64_t j = 0; j < N; ++j)
        {
            acc_t *v = PhiT + j * N;
            for (int64_t i = 0; i < N; ++i)
                tmp[i] = host_dot(A_t + i * N, v, N);
            for (int64_t i = 0; i < N; ++i)
                v[i] = tmp[i];
        }
        if (tiny(PhiT, N * N))
        {
            std::fill(PhiT, PhiT + N * N, acc_t(0));
            break;
        }
    }
}

// A[b, t] is the N x N row-major matrix at A + b * stride_b + (t - 1) * stride_t
// for t = 1 to T - 1. H is [B][T][N], in place: H[b][0] is the initial state,
// H[b][t] holds x[t] on entry and h[t] on exit.
template <typename scalar_t>
void chunked_recurrence(const scalar_t *A, int64_t stride_b, int64_t stride_t,
                        scalar_t *H, int64_t B, int64_t T, int64_t N)
{
    using acc_t = at::opmath_type<scalar_t>;
    const int64_t steps = T - 1;
    const int64_t P = at::get_num_threads();
    int64_t C = 1;
    // A varying in time costs N^3 a step for Phi, against N^2 for a step.
    if (B < P && (stride_t == 0 || P / B > N + 2))
        C = std::min((P + B - 1) / B,
                     std::max<int64_t>(1, steps / min_chunk));

    if (C == 1)
    {
        at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end)
                         {
            std::vector<acc_t> h(N), tmp(N);
            for (int64_t b = begin; b < end; ++b)
            {
                scalar_t *H_b = H + b * T * N;
                std::copy(H_b, H_b + N, h.begin());
                run_steps<true>(A + b * stride_b, stride_t, H_b, 1, T, N,
                                h.data(), tmp.data());
            } });
        return;
    }

    // Chunk c is steps [start(c), start(c + 1)).
    auto start = [&](int64_t c)
    { return 1 + c * steps / C; };
    std::vector<acc_t> Phi(B * C * N * N), hin(B * C * N);

    // Local end states, and Phi; chunk 0 is the true recurrence, and writes.
    at::parallel_for(0, B * C, 1, [&](int64_t begin, int64_t end)
                     {
        std::vector<acc_t> h(N), tmp(N);
        for (int64_t k = begin; k < end; ++k)
        {
            int64_t b = k / C, c = k % C;
            const scalar_t *A_b = A + b * stride_b;
            scalar_t *H_b = H + b * T * N;
            if (c == 0)
            {
                std::copy(H_b, H_b + N, h.begin());
                run_steps<true>(A_b, stride_t, H_b, start(0), start(1), N,
                                h.data(), tmp.data());
            }
            else
            {
                std::fill(h.begin(), h.end(), acc_t(0));
                run_steps<false>(A_b, stride_t, H_b, start(c), start(c + 1), N,
                                 h.data(), tmp.data());
                transition(A_b, stride_t, start(c), start(c + 1), N,
                           Phi.data() + k * N * N, tmp.data());
            }
            std::copy(h.begin(), h.end(), hin.begin() + k * N);
        } });

    // The true end states over the chunks. hin[b][c] holds the local end state
    // on entry and the incoming state on exit.
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end)
                     {
        std::vector<acc_t> prev(N), next(N);
        for (int64_t b = begin; b < end; ++b)
        {
            acc_t *hin_b = hin.data() + b * C * N;
            std::copy(hin_b, hin_b + N, prev.begin());
            for (int64_t c = 1; c < C; ++c)
            {
                acc_t *local = hin_b + c * N;
                const acc_t *PhiT = Phi.data() + (b * C + c) * N * N;
                for (int64_t i = 0; i < N; ++i)
                {
                    next[i] = local[i];
                    for (int64_t j = 0; j < N; ++j)
                        next[i] += PhiT[j * N + i] * prev[j];
                }
                std::copy(prev.begin(), prev.end(), local);
                std::swap(prev, next);
            }
        } });

    // The chunks c >= 1 again, from their incoming states, writing.
    at::parallel_for(0, B * (C - 1), 1, [&](int64_t begin, int64_t end)
                     {
        std::vector<acc_t> h(N), tmp(N);
        for (int64_t k = begin; k < end; ++k)
        {
            int64_t b = k / (C - 1), c = k % (C - 1) + 1;
            std::copy_n(hin.begin() + (b * C + c) * N, N, h.begin());
            run_steps<true>(A + b * stride_b, stride_t, H + b * T * N,
                            start(c), start(c + 1), N, h.data(), tmp.data());
        } });
}

// x is [B][T][N], or [B][T] with N = 1.
at::Tensor recur_cpu_launch(const at::Tensor &A, const at::Tensor &zi,
                            const at::Tensor &x, int64_t N, int64_t stride_b,
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
              N); });
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
