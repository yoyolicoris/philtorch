#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/util/complex.h>
#include <torch/library.h>

#include <algorithm>

#include "host_dot.h"

// y[b][t] = x[b][t] - sum_i a[b][t][i] y[b][t - 1 - i], from y[b][-1 - k] =
// zi[b][k]. x and y are [B][T], a is [B][T][M], zi is [B][M]. The sequences
// run in parallel, each one serially.
template <typename scalar_t>
void allpole_kernel(const scalar_t *x, const scalar_t *a, const scalar_t *zi,
                    scalar_t *y, int64_t B, int64_t T, int64_t M)
{
    using acc_t = at::opmath_type<scalar_t>;
    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end)
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

at::Tensor allpole_cpu(const at::Tensor &x, const at::Tensor &a, const at::Tensor &zi)
{
    TORCH_CHECK(x.is_floating_point() || x.is_complex(), "Input must be floating point or complex");
    TORCH_CHECK(a.scalar_type() == x.scalar_type(), "Coefficients must have the same scalar type as input");
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(), "Initial conditions must have the same scalar type as input");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(zi.dim() == 2, "Initial conditions must be 2D");
    TORCH_CHECK(x.size(0) == zi.size(0), "Batch size of input and initial conditions must match");
    TORCH_CHECK(a.dim() == 3, "a must be 3-dimensional");
    TORCH_CHECK(a.size(0) == x.size(0) && a.size(1) == x.size(1), "Batch and time dimensions of a and x must match");
    TORCH_CHECK(zi.size(1) == a.size(2), "Initial conditions must have one state per coefficient of a");
    auto x_contiguous = x.contiguous();
    auto a_contiguous = a.contiguous();
    auto zi_contiguous = zi.contiguous();
    auto out = at::empty_like(x_contiguous);
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        x.scalar_type(), "allpole_cpu", [&]
        { allpole_kernel<scalar_t>(
              x_contiguous.const_data_ptr<scalar_t>(),
              a_contiguous.const_data_ptr<scalar_t>(),
              zi_contiguous.const_data_ptr<scalar_t>(),
              out.mutable_data_ptr<scalar_t>(), x.size(0), x.size(1),
              a.size(2)); });
    return out;
}

// The CPU scan is in host_recur.cpp.
TORCH_LIBRARY_FRAGMENT(philtorch, m)
{
    m.def("scan(Tensor a, Tensor b, Tensor c) -> Tensor");
    m.def("lpc(Tensor a, Tensor b, Tensor c) -> Tensor");
}

TORCH_LIBRARY_IMPL(philtorch, CPU, m)
{
    m.impl("lpc", &allpole_cpu);
}
