#include <torch/script.h>
#include <torch/torch.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/complex.h>
#include <algorithm>
#include <utility>
#include <vector>

// MSVC does not support OpenMP user-defined reductions.
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp declare reduction(+ : c10::complex<float> : omp_out += omp_in) initializer(omp_priv = 0)
#pragma omp declare reduction(+ : c10::complex<double> : omp_out += omp_in) initializer(omp_priv = 0)
#pragma omp declare reduction(+ : std::complex<float> : omp_out += omp_in) initializer(omp_priv = 0)
#pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in) initializer(omp_priv = 0)
#endif

template <typename scalar_t>
void scan_cpu_vendored(const at::Tensor &input, const at::Tensor &weights,
                       const at::Tensor &initials, const at::Tensor &output)
{
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(initials.dim() == 1, "Initials must be 1D");
    TORCH_CHECK(weights.sizes() == input.sizes(),
                "Weights must have the same size as input");
    TORCH_CHECK(output.sizes() == input.sizes(),
                "Output must have the same size as input");
    TORCH_CHECK(initials.size(0) == input.size(0),
                "The first dimension of initials must be the same as the first dimension of input");
    TORCH_INTERNAL_ASSERT(input.device().is_cpu(), "Input must be on CPU");
    TORCH_INTERNAL_ASSERT(initials.device().is_cpu(), "Initials must be on CPU");
    TORCH_INTERNAL_ASSERT(weights.device().is_cpu(), "Weights must be on CPU");
    TORCH_INTERNAL_ASSERT(output.device().is_cpu(), "Output must be on CPU");
    TORCH_INTERNAL_ASSERT(output.is_contiguous(), "Output must be contiguous");

    const auto n_batch = input.size(0);
    const auto T = input.size(1);
    auto input_contiguous = input.contiguous();
    auto initials_contiguous = initials.contiguous();
    auto weights_contiguous = weights.contiguous();
    const scalar_t *input_ptr = input_contiguous.const_data_ptr<scalar_t>();
    const scalar_t *initials_ptr = initials_contiguous.const_data_ptr<scalar_t>();
    const scalar_t *weights_ptr = weights_contiguous.const_data_ptr<scalar_t>();
    scalar_t *output_ptr = output.mutable_data_ptr<scalar_t>();

#pragma omp parallel for
    for (int64_t b = 0; b < n_batch; ++b)
    {
        scalar_t h = initials_ptr[b];
        const scalar_t *w_row = weights_ptr + b * T;
        const scalar_t *x_row = input_ptr + b * T;
        scalar_t *y_row = output_ptr + b * T;
        for (int64_t t = 0; t < T; ++t)
            y_row[t] = h = h * w_row[t] + x_row[t];
    }
}

template <typename scalar_t>
void allpole_cpu_core(const torch::Tensor &a, const torch::Tensor &padded_out)
{
    TORCH_CHECK(a.dim() == 3, "a must be 3-dimensional");
    TORCH_CHECK(padded_out.dim() == 2, "out must be 2-dimensional");
    TORCH_CHECK(padded_out.size(0) == a.size(0), "Batch size of out and x must match");
    TORCH_CHECK(padded_out.size(1) == (a.size(1) + a.size(2)), "Time dimension of out must match x and a");
    TORCH_INTERNAL_ASSERT(a.device().is_cpu(), "a must be on CPU");
    TORCH_INTERNAL_ASSERT(padded_out.device().is_cpu(), "Output must be on CPU");
    TORCH_INTERNAL_ASSERT(padded_out.is_contiguous(), "Output must be contiguous");

    const auto B = a.size(0);
    const auto T = a.size(1);
    const auto order = a.size(2);
    // Negate coefficients once so the inner recurrence is a pure running sum,
    // which is compatible with `+` reductions and lets the compiler vectorize.
    auto neg_a = (-a).contiguous();
    const scalar_t *a_ptr = neg_a.const_data_ptr<scalar_t>();
    scalar_t *out_ptr = padded_out.mutable_data_ptr<scalar_t>();

#pragma omp parallel for
    for (int64_t b = 0; b < B; ++b)
    {
        scalar_t *out_b = out_ptr + b * (T + order) + order;
        const scalar_t *a_b = a_ptr + b * T * order;
        for (int64_t t = 0; t < T; ++t)
        {
            const scalar_t *a_bt = a_b + t * order;
            scalar_t y = 0;
#pragma omp simd reduction(+ : y)
            for (int64_t i = 0; i < order; ++i)
                y += a_bt[i] * out_b[t - i - 1];
            out_b[t] = out_b[t] + y;
        }
    }
}

at::Tensor scan_cpu_wrapper_vendored(const at::Tensor &input, const at::Tensor &weights,
                                     const at::Tensor &initials)
{
    TORCH_CHECK(input.is_floating_point() || input.is_complex(), "Input must be floating point or complex");
    TORCH_CHECK(initials.scalar_type() == input.scalar_type(), "Initials must have the same scalar type as input");
    TORCH_CHECK(weights.scalar_type() == input.scalar_type(), "Weights must have the same scalar type as input");
    auto output = at::empty_like(input);
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        input.scalar_type(), "scan_cpu",
        [&]
        { scan_cpu_vendored<scalar_t>(input, weights, initials, output); });
    return output;
}

at::Tensor allpole_cpu(const at::Tensor &x, const at::Tensor &a, const at::Tensor &zi)
{
    TORCH_CHECK(x.is_floating_point() || x.is_complex(), "Input must be floating point or complex");
    TORCH_CHECK(a.scalar_type() == x.scalar_type(), "Coefficients must have the same scalar type as input");
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(), "Initial conditions must have the same scalar type as input");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(zi.dim() == 2, "Initial conditions must be 2D");
    TORCH_CHECK(x.size(0) == zi.size(0), "Batch size of input and initial conditions must match");
    auto out = at::cat({zi.flip(1), x}, 1).contiguous();
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        x.scalar_type(), "allpole_cpu", [&]
        { allpole_cpu_core<scalar_t>(a, out); });
    return out.slice(1, zi.size(1), out.size(1)).contiguous();
}

TORCH_LIBRARY(torchlpc, m)
{
    m.def("scan(Tensor a, Tensor b, Tensor c) -> Tensor");
    m.def("lpc(Tensor a, Tensor b, Tensor c) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchlpc, CPU, m)
{
    m.impl("scan", &scan_cpu_wrapper_vendored);
    m.impl("lpc", &allpole_cpu);
}
