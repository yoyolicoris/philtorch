#include <torch/script.h>
#include <torch/torch.h>

template <typename scalar_t>
void host_lti_batch_mat_recur_N_order(int B, int T, int N,
                                      const scalar_t *A, // [B][N][N]
                                      scalar_t *H_out    // [B][T][N]
)
{
// Process each batch in parallel
#pragma omp parallel for
    for (int b = 0; b < B; ++b)
    {
        const scalar_t *A_b = A + b * N * N;
        scalar_t *H_b = H_out + b * T * N;

        // t loop
        for (int t = 1; t < T; ++t)
        {
            const scalar_t *H_prev = H_b + (t - 1) * N;
            scalar_t *H_curr = H_b + t * N;

            for (int i = 0; i < N; ++i)
            {
                scalar_t sum = 0;
                const scalar_t *Arow = A_b + i * N;
                for (int j = 0; j < N; ++j)
                    sum += Arow[j] * H_prev[j];
                H_curr[i] += sum;
            }
        }
    }
}

template <typename scalar_t>
void host_lti_share_mat_recur_N_order(int B, int T, int N,
                                      const scalar_t *A, // N][N]
                                      scalar_t *H_out    // [B][T][N]
)
{
// Process each batch in parallel
#pragma omp parallel for
    for (int b = 0; b < B; ++b)
    {
        scalar_t *H_b = H_out + b * T * N;
        // t loop
        for (int t = 1; t < T; ++t)
        {
            const scalar_t *H_prev = H_b + (t - 1) * N;
            scalar_t *H_curr = H_b + t * N;
            for (int i = 0; i < N; ++i)
            {
                scalar_t sum = 0;
                const scalar_t *Arow = A + i * N;
                for (int j = 0; j < N; ++j)
                    sum += Arow[j] * H_prev[j];
                H_curr[i] += sum;
            }
        }
    }
}

at::Tensor lti_mat_recur_N_order_cpu_impl(const at::Tensor &A,
                                          const at::Tensor &zi,
                                          const at::Tensor &x)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be a 2D or 3D tensor");
    TORCH_CHECK(x.size(2) == A.size(-1),
                "Last dimension of x must match last dimension of A");
    TORCH_CHECK(A.size(-1) == A.size(-2),
                "Last two dimensions of A must be equal");

    auto n_steps = x.size(1) + 1; // +1 for the initial state
    auto n_batches = x.size(0);
    auto order = A.size(-1);

    auto A_contiguous = A.contiguous();
    auto out = at::cat({zi.unsqueeze(1), x}, 1).contiguous();

    if (A.dim() == 3)
    {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            at::kLong, x.scalar_type(), "host_lti_batch_mat_recur_N_order",
            [&]
            {
                host_lti_batch_mat_recur_N_order<scalar_t>(
                    n_batches, n_steps, order,
                    A_contiguous.const_data_ptr<scalar_t>(),
                    out.mutable_data_ptr<scalar_t>());
            });
    }
    else
    {
        // Shared
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            at::kLong, x.scalar_type(), "host_lti_share_mat_recur_N_order", [&]
            { host_lti_share_mat_recur_N_order<scalar_t>(
                  n_batches, n_steps, order,
                  A_contiguous.const_data_ptr<scalar_t>(),
                  out.mutable_data_ptr<scalar_t>()); });
    }
    return out.slice(1, 1, n_steps)
        .contiguous(); // Remove the initial state from the output
}

TORCH_LIBRARY_IMPL(philtorch, CPU, m)
{
    m.impl("lti_recurN", &lti_mat_recur_N_order_cpu_impl);
    m.impl("lti_recur2", &lti_mat_recur_N_order_cpu_impl);
}