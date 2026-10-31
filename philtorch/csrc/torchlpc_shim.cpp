#include "stable_support.h"

#include <algorithm>

#include "host_dot.h"

using torch::stable::Tensor;

// y[b][t] = x[b][t] - sum_i a[b][t][i] y[b][t - 1 - i], from y[b][-1 - k] =
// zi[b][k]. x and y are [B][T], a is [B][T][M], zi is [B][M]. The sequences
// run in parallel, each one serially.
template <typename scalar_t>
void allpole_kernel(const scalar_t *x, const scalar_t *a, const scalar_t *zi,
                    scalar_t *y, int64_t B, int64_t T, int64_t M)
{
    using acc_t = philtorch::opmath_type<scalar_t>;
    torch::stable::parallel_for(0, B, 1, [&](int64_t begin, int64_t end)
                                {
        for (int64_t b = begin; b < end; ++b)
        {
            const scalar_t *x_b = x + b * T;
            const scalar_t *a_b = a + b * T * M;
            const scalar_t *zi_b = zi + b * M;
            scalar_t *y_b = y + b * T;
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
        } });
}

Tensor allpole_cpu(const Tensor &x, const Tensor &a, const Tensor &zi)
{
    STD_TORCH_CHECK(philtorch::is_floating(x.scalar_type()) || philtorch::is_complex(x.scalar_type()), "Input must be floating point or complex");
    STD_TORCH_CHECK(a.scalar_type() == x.scalar_type(), "Coefficients must have the same scalar type as input");
    STD_TORCH_CHECK(zi.scalar_type() == x.scalar_type(), "Initial conditions must have the same scalar type as input");
    STD_TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    STD_TORCH_CHECK(zi.dim() == 2, "Initial conditions must be 2D");
    STD_TORCH_CHECK(x.size(0) == zi.size(0), "Batch size of input and initial conditions must match");
    STD_TORCH_CHECK(a.dim() == 3, "a must be 3-dimensional");
    STD_TORCH_CHECK(a.size(0) == x.size(0) && a.size(1) == x.size(1), "Batch and time dimensions of a and x must match");
    STD_TORCH_CHECK(zi.size(1) == a.size(2), "Initial conditions must have one state per coefficient of a");
    auto x_contiguous = torch::stable::contiguous(x);
    auto a_contiguous = torch::stable::contiguous(a);
    auto zi_contiguous = torch::stable::contiguous(zi);
    auto out = torch::stable::empty_like(x_contiguous);
    THO_DISPATCH_V2(
        x.scalar_type(), "allpole_cpu",
        AT_WRAP([&]
                { allpole_kernel<scalar_t>(
                      x_contiguous.const_data_ptr<scalar_t>(),
                      a_contiguous.const_data_ptr<scalar_t>(),
                      zi_contiguous.const_data_ptr<scalar_t>(),
                      out.mutable_data_ptr<scalar_t>(), x.size(0), x.size(1),
                      a.size(2)); }),
        AT_FLOATING_TYPES, AT_COMPLEX_TYPES);
    return out;
}

// The CPU scan is in host_recur.cpp.
STABLE_TORCH_LIBRARY_FRAGMENT(philtorch, m)
{
    m.def("scan(Tensor a, Tensor b, Tensor c) -> Tensor");
    m.def("lpc(Tensor a, Tensor b, Tensor c) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(philtorch, CPU, m)
{
    m.impl("lpc", TORCH_BOX(&allpole_cpu));
}
