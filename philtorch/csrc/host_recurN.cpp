#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <utility>
#include <valarray>
#include <vector>

template <typename T>
using host_sqmN_pair = std::valarray<T>;

template <typename T>
host_sqmN_pair<T> recurN_binary_op(const int &n,
                                   const std::valarray<int> &indexes,
                                   const host_sqmN_pair<T> &a,
                                   const host_sqmN_pair<T> &b)
{
    int size = a.size();
    host_sqmN_pair<T> result(size);
    std::transform(
        std::begin(indexes), std::end(indexes), std::begin(result),
        [&](const auto &i)
        {
            auto row = i / n;
            auto col = i % n;
            T tmp;
            if (row == n)
            {
                tmp = std::transform_reduce(
                    std::begin(b) + col * n, std::begin(b) + (col + 1) * n,
                    std::begin(a) + n * n, b[i], std::plus<T>(),
                    std::multiplies<T>());
            }
            else
            {
                tmp = (a[std::slice(col, n, n)] * b[std::slice(row * n, n, 1)])
                          .sum();
            }
            return tmp;
        });
    return result;
}

template <typename scalar_t>
void host_batch_mat_recur_N_order(const scalar_t *Ax, scalar_t *out,
                                  int n_steps, int order)
{
    std::vector<host_sqmN_pair<scalar_t>> buffer(n_steps);
    auto vec_size = order * order + order;

    at::parallel_for(0, n_steps, 1, [&](int64_t start, int64_t end)
                     {
        for (auto i = start; i < end; i++) {
            buffer[i].resize(vec_size);
            auto offset = i * vec_size;
            std::copy(Ax + offset, Ax + offset + vec_size,
                      std::begin(buffer[i]));
        } });

    auto indexes = std::valarray<int>(vec_size);
    at::parallel_for(0, vec_size, 1, [&](int64_t start, int64_t end)
                     {
        for (auto i = start; i < end; i++) {
            indexes[i] = i;
        } });
    std::inclusive_scan(
        buffer.begin(), buffer.end(), buffer.begin(),
        std::bind(recurN_binary_op<scalar_t>, order, indexes,
                  std::placeholders::_1, std::placeholders::_2));

    at::parallel_for(0, n_steps, 1, [&](int64_t start, int64_t end)
                     {
        for (auto i = start; i < end; i++) {
            auto &result = buffer[i];
            auto offset = i * order;
            std::copy(std::begin(result) + order * order,
                      std::begin(result) + vec_size, out + offset);
        } });
}

template <typename scalar_t>
void host_batch_mat_recur_N_order_omp(int B, int T, int N,
                                      const scalar_t *A, // [B][T][N][N]
                                      scalar_t *H_out    // [B][T][N]
)
{
// Process each batch in parallel
#pragma omp parallel for
    for (int b = 0; b < B; ++b)
    {
        const scalar_t *A_b = A + b * T * N * N;
        scalar_t *H_b = H_out + b * T * N;

        // t loop
        for (int t = 1; t < T; ++t)
        {
            const scalar_t *A_bt = A_b + t * N * N;
            const scalar_t *H_prev = H_b + (t - 1) * N;
            scalar_t *H_curr = H_b + t * N;

            for (int i = 0; i < N; ++i)
            {
                scalar_t sum = 0;
                const scalar_t *Arow = A_bt + i * N;
                for (int j = 0; j < N; ++j)
                    sum += Arow[j] * H_prev[j];
                H_curr[i] += sum;
            }
        }
    }
}

template <typename scalar_t>
void host_share_mat_recur_N_order_omp(int B, int T, int N,
                                      const scalar_t *A, // [T][N][N]
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
            const scalar_t *A_bt = A + t * N * N;
            const scalar_t *H_prev = H_b + (t - 1) * N;
            scalar_t *H_curr = H_b + t * N;
            for (int i = 0; i < N; ++i)
            {
                scalar_t sum = 0;
                const scalar_t *Arow = A_bt + i * N;
                for (int j = 0; j < N; ++j)
                    sum += Arow[j] * H_prev[j];
                H_curr[i] += sum;
            }
        }
    }
}

template <typename scalar_t>
void host_share_mat_recur_N_order(const scalar_t *A, const scalar_t *x,
                                  scalar_t *out, int n_steps, int order,
                                  int n_batches)
{
    auto total_steps = n_steps * n_batches;
    auto order_squared = order * order;
    std::vector<host_sqmN_pair<scalar_t>> buffer(total_steps);

    at::parallel_for(0, total_steps, 1, [&](int64_t start, int64_t end)
                     {
        for (auto i = start; i < end; i++) {
            auto offset = i % n_steps * order_squared;
            buffer[i].resize(order_squared + order);
            std::copy(A + offset, A + offset + order_squared,
                      std::begin(buffer[i]));
            offset = i * order;
            std::copy(x + offset, x + offset + order,
                      std::begin(buffer[i]) + order_squared);
        } });

    auto indexes = std::valarray<int>(order_squared + order);
    at::parallel_for(0, order_squared + order, 1,
                     [&](int64_t start, int64_t end)
                     {
                         for (auto i = start; i < end; i++)
                         {
                             indexes[i] = i;
                         }
                     });

    std::inclusive_scan(
        buffer.begin(), buffer.end(), buffer.begin(),
        std::bind(recurN_binary_op<scalar_t>, order, indexes,
                  std::placeholders::_1, std::placeholders::_2));

    at::parallel_for(0, total_steps, 1, [&](int64_t start, int64_t end)
                     {
        for (auto i = start; i < end; i++) {
            auto &result = buffer[i];
            auto offset = i * order;
            std::copy(std::begin(result) + order_squared,
                      std::begin(result) + order_squared + order, out + offset);
        } });
}

at::Tensor mat_recur_N_order_cpu_impl(const at::Tensor &A, const at::Tensor &zi,
                                      const at::Tensor &x)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    TORCH_CHECK(x.size(2) == A.size(-1),
                "Last dimension of x must match last dimension of A");
    TORCH_CHECK(A.size(-1) == A.size(-2),
                "Last two dimensions of A must be equal");

    auto n_steps = x.size(1) + 1; // +1 for the initial state
    auto n_batches = x.size(0);
    auto order = A.size(-1);

    auto A_contiguous = at::pad(A, {0, 0, 0, 0, 1, 0}).contiguous();
    auto x_contiguous = at::cat({zi.unsqueeze(1), x}, 1).contiguous();
    auto out = at::empty_like(x_contiguous);

    if (A.dim() == 4)
    {
        // Batch
        auto Ax = at::cat({A_contiguous, x_contiguous.unsqueeze(-2)}, -2)
                      .contiguous();
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            at::kLong, x.scalar_type(), "host_batch_mat_recur_N_order", [&]
            { host_batch_mat_recur_N_order<scalar_t>(
                  Ax.const_data_ptr<scalar_t>(),
                  out.mutable_data_ptr<scalar_t>(), n_steps * n_batches,
                  order); });
    }
    else
    {
        // Shared
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            at::kLong, x.scalar_type(), "host_share_mat_recur_N_order", [&]
            { host_share_mat_recur_N_order<scalar_t>(
                  A_contiguous.const_data_ptr<scalar_t>(),
                  x_contiguous.const_data_ptr<scalar_t>(),
                  out.mutable_data_ptr<scalar_t>(), n_steps, order,
                  n_batches); });
    }
    return out.slice(1, 1, n_steps)
        .contiguous(); // Remove the initial state from the output
}

at::Tensor mat_recur_N_order_cpu_omp_impl(const at::Tensor &A,
                                          const at::Tensor &zi,
                                          const at::Tensor &x)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    TORCH_CHECK(x.size(2) == A.size(-1),
                "Last dimension of x must match last dimension of A");
    TORCH_CHECK(A.size(-1) == A.size(-2),
                "Last two dimensions of A must be equal");

    auto n_steps = x.size(1) + 1; // +1 for the initial state
    auto n_batches = x.size(0);
    auto order = A.size(-1);

    auto A_contiguous = at::pad(A, {0, 0, 0, 0, 1, 0}).contiguous();
    auto out = at::cat({zi.unsqueeze(1), x}, 1).contiguous();

    if (A.dim() == 4)
    {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            at::kLong, x.scalar_type(), "host_batch_mat_recur_N_order_omp",
            [&]
            {
                host_batch_mat_recur_N_order_omp<scalar_t>(
                    n_batches, n_steps, order,
                    A_contiguous.const_data_ptr<scalar_t>(),
                    out.mutable_data_ptr<scalar_t>());
            });
    }
    else
    {
        // Shared
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            at::kLong, x.scalar_type(), "host_share_mat_recur_N_order", [&]
            { host_share_mat_recur_N_order_omp<scalar_t>(
                  n_batches, n_steps, order,
                  A_contiguous.const_data_ptr<scalar_t>(),
                  out.mutable_data_ptr<scalar_t>()); });
    }
    return out.slice(1, 1, n_steps)
        .contiguous(); // Remove the initial state from the output
}

TORCH_LIBRARY_IMPL(philtorch, CPU, m)
{
    m.impl("recurN", &mat_recur_N_order_cpu_omp_impl);
    m.impl("recur2", &mat_recur_N_order_cpu_omp_impl);
}