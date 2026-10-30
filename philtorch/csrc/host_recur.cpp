#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "host_dot.h"

// The recurrence h[t] = A[t] h[t - 1] + x[t] on CPU. A sequence is split over
// time into chunks, when there are fewer sequences than threads: each chunk's
// transition matrix and local end state give its true incoming state, and the
// chunks then run in parallel again from it. Each has at least min_chunk steps.
constexpr int64_t min_chunk = 1024;

// A zero product stays zero (a tiny one may grow again), so the rest of it
// can be skipped.
template <typename acc_t>
bool is_zero(const std::vector<acc_t> &v)
{
    return std::all_of(v.begin(), v.end(),
                       [](acc_t x)
                       { return x == acc_t(0); });
}

// A_b + t * stride_t is the matrix of step t. Steps s to e - 1 of a sequence
// from the state h, which is updated: X_b[t] is x[t], and Y_b[t] gets h[t] if
// write. M is an int, which makes the loop faster.
template <bool write, typename scalar_t, typename acc_t>
void run_steps(const scalar_t *A_b, int64_t stride_t, const scalar_t *X_b,
               scalar_t *Y_b, int64_t s, int64_t e, int M, acc_t *h,
               acc_t *tmp)
{
    if (M == 1)
    {
        acc_t v = h[0];
        for (int64_t t = s; t < e; ++t)
        {
            v = static_cast<acc_t>(X_b[t]) +
                static_cast<acc_t>(A_b[t * stride_t]) * v;
            if constexpr (write)
                Y_b[t] = v;
        }
        h[0] = v;
        return;
    }
    acc_t *prev = h, *curr = tmp;
    for (int64_t t = s; t < e; ++t)
    {
        const scalar_t *A_t = A_b + t * stride_t;
        const scalar_t *X_t = X_b + t * M;
        for (int i = 0; i < M; ++i)
        {
            curr[i] = static_cast<acc_t>(X_t[i]) + host_dot(A_t + i * M, prev, M);
            if constexpr (write)
                Y_b[t * M + i] = curr[i];
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
                if (is_zero(power)) // the remaining factor is a power of it
                {
                    std::fill(R.begin(), R.end(), acc_t(0));
                    break;
                }
            }
        }
    }
    else if (M == 1) // a scalar product, which matmul and is_zero would slow
    {
        acc_t p = 1;
        for (int64_t t = s; t < e && p != acc_t(0); ++t)
            p *= static_cast<acc_t>(A_b[t * stride_t]);
        R[0] = p;
    }
    else
    {
        for (int64_t t = s; t < e; ++t)
        {
            matmul(A_b + t * stride_t, R.data(), W.data(), M);
            std::swap(R, W);
            if (is_zero(R))
                break;
        }
    }
    std::copy(R.begin(), R.end(), Phi);
}

// A[b, t] is the M x M row-major matrix at A + b * stride_b + t * stride_t.
// X and Y are [B][T][M], Z is [B][M]: Y[b][t] = A[b, t] Y[b][t - 1] + X[b][t],
// from Y[b][-1] = Z[b].
template <typename scalar_t>
void chunked_recurrence(const scalar_t *A, int64_t stride_b, int64_t stride_t,
                        const scalar_t *X, const scalar_t *Z, scalar_t *Y,
                        int64_t B, int64_t T, int M)
{
    using acc_t = at::opmath_type<scalar_t>;
    const int64_t P = at::get_num_threads();
    int64_t C = B > 0 ? std::min(P / B, T / min_chunk) : 1;
    // A constant A costs 2 passes over the steps, a varying one M + 2.
    if (C <= (stride_t == 0 ? 2 : M + 2))
        C = 1;

    // Chunk c is steps [start(c), start(c + 1)).
    auto start = [&](int64_t c)
    { return c * T / C; };
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
            const scalar_t *X_b = X + b * T * M;
            scalar_t *Y_b = Y + b * T * M;
            if (c == 0)
            {
                std::copy(Z + b * M, Z + (b + 1) * M, h.begin());
                run_steps<true>(A_b, stride_t, X_b, Y_b, start(0), start(1), M,
                                h.data(), tmp.data());
            }
            else
            {
                std::fill(h.begin(), h.end(), acc_t(0));
                run_steps<false>(A_b, stride_t, X_b, Y_b, start(c),
                                 start(c + 1), M, h.data(), tmp.data());
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
                if (!std::all_of(Phi_c, Phi_c + M * M, [](acc_t v)
                                 { return std::isfinite(std::abs(v)); }))
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
            const acc_t *hin_c = hin.data() + (b * C + c) * M;
            std::copy(hin_c, hin_c + M, h.begin());
            run_steps<true>(A + b * stride_b, stride_t, X + b * T * M,
                            Y + b * T * M, start(c),
                            unsplit[b] ? T : start(c + 1), M, h.data(),
                            tmp.data());
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
    auto x_contiguous = x.contiguous();
    auto zi_contiguous = zi.contiguous();
    auto out = at::empty_like(x_contiguous);
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
        at::kLong, at::kHalf, at::kBFloat16, x.scalar_type(),
        "chunked_recurrence", [&]
        { chunked_recurrence<scalar_t>(
              A_contiguous.const_data_ptr<scalar_t>(), stride_b, stride_t,
              x_contiguous.const_data_ptr<scalar_t>(),
              zi_contiguous.const_data_ptr<scalar_t>(),
              out.mutable_data_ptr<scalar_t>(), x.size(0), x.size(1),
              static_cast<int>(M)); });
    return out;
}

at::Tensor lti_recur_cpu(const at::Tensor &a, const at::Tensor &zi,
                         const at::Tensor &x)
{
    TORCH_CHECK(a.dim() <= 1, "A must be a vector or a scalar");
    TORCH_CHECK(x.dim() == 2 && zi.dim() == 1 && zi.size(0) == x.size(0),
                "x must be (B, T) and zi (B,)");
    TORCH_CHECK(x.size(1) > 0, "x must contain at least one time step");
    bool per_item = a.dim() == 1 && a.numel() == x.size(0);
    TORCH_CHECK(per_item || a.numel() == 1, "A must be a scalar or (B,)");
    return recur_cpu_launch(a, zi, x, 1, per_item ? 1 : 0, 0);
}

at::Tensor lti_recurN_cpu(const at::Tensor &A, const at::Tensor &zi,
                          const at::Tensor &x)
{
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be a 2D or 3D tensor");
    TORCH_CHECK(x.dim() == 3 && zi.dim() == 2 && zi.size(0) == x.size(0) &&
                    zi.size(1) == A.size(-1),
                "x must be (B, T, M) and zi (B, M)");
    TORCH_CHECK(x.size(2) == A.size(-1),
                "Last dimension of x must match last dimension of A");
    TORCH_CHECK(A.size(-1) == A.size(-2),
                "Last two dimensions of A must be equal");
    TORCH_CHECK(A.dim() == 2 || A.size(0) == x.size(0),
                "A must have the batch size of x");
    auto order = A.size(-1);
    return recur_cpu_launch(A, zi, x, order, A.dim() == 3 ? order * order : 0, 0);
}

at::Tensor recurN_cpu(const at::Tensor &A, const at::Tensor &zi,
                      const at::Tensor &x)
{
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    TORCH_CHECK(x.dim() == 3 && zi.dim() == 2 && zi.size(0) == x.size(0) &&
                    zi.size(1) == A.size(-1),
                "x must be (B, T, M) and zi (B, M)");
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

// h[t] = decay[t] h[t - 1] + impulse[t]: the scalar recurrence with a
// (B, T) A, one coefficient per step.
at::Tensor scan_cpu(const at::Tensor &impulse, const at::Tensor &decay,
                    const at::Tensor &init)
{
    TORCH_CHECK(impulse.dim() == 2 && decay.sizes() == impulse.sizes() &&
                    init.dim() == 1 && init.size(0) == impulse.size(0),
                "impulse and decay must be (B, T) and init (B,)");
    // recur_cpu_launch also takes integers and half precision.
    TORCH_CHECK(impulse.scalar_type() == at::kFloat ||
                    impulse.scalar_type() == at::kDouble ||
                    impulse.scalar_type() == at::kComplexFloat ||
                    impulse.scalar_type() == at::kComplexDouble,
                "Input must be float32, float64, complex64 or complex128");
    return recur_cpu_launch(decay, init, impulse, 1, impulse.size(1), 1);
}

TORCH_LIBRARY_IMPL(philtorch, CPU, m)
{
    m.impl("scan", &scan_cpu);
    m.impl("lti_recur", &lti_recur_cpu);
    m.impl("lti_recurN", &lti_recurN_cpu);
    m.impl("lti_recur2", &lti_recurN_cpu);
    m.impl("lti_recur3", &lti_recurN_cpu);
    m.impl("recurN", &recurN_cpu);
    m.impl("recur2", &recurN_cpu);
    m.impl("recur3", &recurN_cpu);
}
