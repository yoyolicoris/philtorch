#include "stable_support.h"

#include <algorithm>
#include <type_traits>
#include <vector>

#include "host_dot.h"
#include "recur_checks.h"

using torch::stable::Tensor;

// y[b][t] = x[b][t] - sum_i a[b][t][i] y[b][t - 1 - i], from y[b][-1 - k] =
// zi[b][k]. x and y are [B][T], a is [B][T][M], zi is [B][M]. The sequences
// run in parallel, each one serially. Where the accumulation type is wider than
// scalar_t (float16, bfloat16, complex32), a sequence's history is carried in
// it, not read back from the rounded output, as the recurrence kernels carry
// their state.
template <typename scalar_t>
void allpole_kernel(const scalar_t *x, const scalar_t *a, const scalar_t *zi,
                    scalar_t *y, int64_t B, int64_t T, int64_t M)
{
    using acc_t = philtorch::opmath_type<scalar_t>;
    torch::stable::parallel_for(0, B, 1, [&](int64_t begin, int64_t end)
                                {
        // h[0..M) is y[-M..-1], so that h[M - 1 - k] = zi[k], and h[M + s] =
        // y[s]: step t reads h[M + t - 1 - i] with no special case at the start.
        std::vector<acc_t> h;
        if constexpr (!std::is_same_v<acc_t, scalar_t>)
            h.resize(M + T);
        for (int64_t b = begin; b < end; ++b)
        {
            const scalar_t *x_b = x + b * T;
            const scalar_t *a_b = a + b * T * M;
            const scalar_t *zi_b = zi + b * M;
            scalar_t *y_b = y + b * T;
            if constexpr (std::is_same_v<acc_t, scalar_t>)
            {
                // The first M steps reach back into zi.
                const int64_t head = std::min(T, M);
                for (int64_t t = 0; t < head; ++t)
                {
                    const scalar_t *a_t = a_b + t * M;
                    y_b[t] = static_cast<acc_t>(x_b[t]) -
                             host_dot<-1>(a_t, y_b + t - 1, static_cast<int>(t)) -
                             host_dot(a_t + t, zi_b, static_cast<int>(M - t));
                }
                for (int64_t t = head; t < T; ++t)
                {
                    const scalar_t *a_t = a_b + t * M;
                    y_b[t] = static_cast<acc_t>(x_b[t]) -
                             host_dot<-1>(a_t, y_b + t - 1, static_cast<int>(M));
                }
            }
            else
            {
                for (int64_t k = 0; k < M; ++k)
                    h[M - 1 - k] = static_cast<acc_t>(zi_b[k]);
                for (int64_t t = 0; t < T; ++t)
                {
                    const acc_t y_t =
                        static_cast<acc_t>(x_b[t]) -
                        host_dot<-1>(a_b + t * M, h.data() + M + t - 1,
                                     static_cast<int>(M));
                    h[M + t] = y_t;
                    y_b[t] = y_t;
                }
            }
        } });
}

Tensor allpole_cpu(const Tensor &x, const Tensor &a, const Tensor &zi)
{
    check_lpc_inputs(x, a, zi);
    auto x_contiguous = torch::stable::contiguous(x);
    auto a_contiguous = torch::stable::contiguous(a);
    auto zi_contiguous = torch::stable::contiguous(zi);
    auto out = torch::stable::empty_like(x_contiguous);
    using S = torch::headeronly::ScalarType;
    THO_DISPATCH_V2(
        x.scalar_type(), "allpole_cpu",
        AT_WRAP([&]
                { allpole_kernel<scalar_t>(
                      x_contiguous.const_data_ptr<scalar_t>(),
                      a_contiguous.const_data_ptr<scalar_t>(),
                      zi_contiguous.const_data_ptr<scalar_t>(),
                      out.mutable_data_ptr<scalar_t>(), x.size(0), x.size(1),
                      a.size(2)); }),
        AT_FLOATING_TYPES, AT_COMPLEX_TYPES, S::Half, S::BFloat16,
        S::ComplexHalf);
    return out;
}

// The CPU scan is in host_recur.cpp; the schemas of scan and lpc in ops.cpp.
STABLE_TORCH_LIBRARY_IMPL(philtorch, CPU, m)
{
    m.impl("lpc", TORCH_BOX(&allpole_cpu));
}
