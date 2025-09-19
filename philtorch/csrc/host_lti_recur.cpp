#include <torch/script.h>
#include <torch/torch.h>

template <typename scalar_t>
void host_lti_batch_linear_recurrence(int B, int T,
                                      const scalar_t *a,
                                      scalar_t *out)
{
// Process each batch in parallel
#pragma omp parallel for
    for (int b = 0; b < B; ++b)
    {
        const scalar_t a_b = a[b];
        scalar_t *out_b = out + b * T;

        // t loop
        for (int t = 1; t < T; ++t)
        {
            out_b[t] += a_b * out_b[t - 1];
        }
    }
}

template <typename scalar_t>
void host_lti_shared_linear_recurrence(int B, int T,
                                       const scalar_t a,
                                       scalar_t *out)
{
// Process each batch in parallel
#pragma omp parallel for
    for (int b = 0; b < B; ++b)
    {
        scalar_t *out_b = out + b * T;
        // t loop
        for (int t = 1; t < T; ++t)
        {
            out_b[t] += a * out_b[t - 1];
        }
    }
}

at::Tensor lti_recur_cpu_impl(const at::Tensor &a,
                              const at::Tensor &zi, const at::Tensor &x)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(a.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(a.dim() <= 1, "A must be a vector or a scalar");
    TORCH_CHECK(zi.dim() == 1, "zi must be a vector");

    auto n_steps = x.size(1) + 1; // +1 for the initial state
    auto n_batches = x.size(0);
    auto output =
        at::cat({zi.unsqueeze(1), x}, 1).contiguous();
    auto a_contiguous = a.contiguous();

    if (a.dim() == 1 && a.numel() == n_batches)
    {

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            x.scalar_type(), "host_lti_batch_linear_recurrence", [&]
            { host_lti_batch_linear_recurrence<scalar_t>(
                  n_batches, n_steps,
                  a_contiguous.const_data_ptr<scalar_t>(),
                  output.mutable_data_ptr<scalar_t>()); });
    }
    else
    {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            x.scalar_type(), "host_lti_shared_linear_recurrence", [&]
            { host_lti_shared_linear_recurrence<scalar_t>(
                  n_batches, n_steps,
                  a_contiguous.item<scalar_t>(),
                  output.mutable_data_ptr<scalar_t>()); });
    }

    return output.slice(1, 1, output.size(1))
        .contiguous(); // Remove the initial state from the output
}

TORCH_LIBRARY_IMPL(philtorch, CPU, m)
{
    m.impl("lti_recur", &lti_recur_cpu_impl);
}