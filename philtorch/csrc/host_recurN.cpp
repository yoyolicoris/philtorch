#include <Python.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <execution>
#include <utility>
#include <valarray>
#include <vector>

template <typename T>
using host_sqmN_pair = std::valarray<T>;

template <typename T>
host_sqmN_pair<T> recurN_binary_op(const int &n,
                                   const std::valarray<int> &indexes,
                                   const host_sqmN_pair<T> &a,
                                   const host_sqmN_pair<T> &b) {
    int size = a.size();
    host_sqmN_pair<T> result(size);
    // auto indexes = std::make_index_sequence<size>{};
    std::for_each(std::execution::par, std::begin(indexes), std::end(indexes),
                  [&](const auto &i) {
                      auto row = i / n;
                      auto col = i % n;
                      if (row == n) {
                          result[i] = std::transform_reduce(
                              std::execution::par, std::begin(b) + col * n,
                              std::begin(b) + (col + 1) * n,
                              std::begin(a) + n * n, b[i], std::plus<T>(),
                              std::multiplies<T>());
                      } else {
                          //   auto a_col = a[std::slice(col, n, n)];
                          //   result[i] = std::transform_reduce(
                          //       std::execution::par, std::begin(b) + row * n,
                          //       std::begin(b) + (row + 1) * n,
                          //       std::begin(a_col), std::plus<T>(),
                          //       std::multiplies<T>());
                          T sum = 0;
                          for (int j = 0; j < n; ++j) {
                              sum += b[row * n + j] * a[j * n + col];
                          }
                          result[i] = sum;
                      }
                  });
    return result;
}

template <typename scalar_t>
void host_batch_mat_recur_N_order(const scalar_t *A, const scalar_t *x,
                                  scalar_t *out, int n_steps, int order) {
    std::vector<host_sqmN_pair<scalar_t>> buffer(n_steps);
    auto order_squared = order * order;

    at::parallel_for(0, n_steps, 1, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
            buffer[i].resize(order_squared + order);
            auto offset = i * order_squared;
            std::copy(std::execution::par, A + offset,
                      A + offset + order_squared, std::begin(buffer[i]));
            offset = i * order;
            std::copy(std::execution::par, x + offset, x + offset + order,
                      std::begin(buffer[i]) + order_squared);
        }
    });

    auto indexes = std::valarray<int>(order_squared + order);
    at::parallel_for(0, order_squared + order, 1,
                     [&](int64_t start, int64_t end) {
                         for (auto i = start; i < end; i++) {
                             indexes[i] = i;
                         }
                     });
    std::inclusive_scan(
        std::execution::par, buffer.begin(), buffer.end(), buffer.begin(),
        std::bind(recurN_binary_op<scalar_t>, order, indexes,
                  std::placeholders::_1, std::placeholders::_2));

    at::parallel_for(0, n_steps, 1, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
            auto &result = buffer[i];
            auto offset = i * order;
            std::copy(std::execution::par, std::begin(result) + order_squared,
                      std::begin(result) + order_squared + order, out + offset);
        }
    });
}

template <typename scalar_t>
void host_share_mat_recur_N_order(const scalar_t *A, const scalar_t *x,
                                  scalar_t *out, int n_steps, int order,
                                  int n_batches) {
    auto total_steps = n_steps * n_batches;
    auto order_squared = order * order;
    std::vector<host_sqmN_pair<scalar_t>> buffer(total_steps);

    at::parallel_for(0, total_steps, 1, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
            auto offset = i % n_steps * order_squared;
            buffer[i].resize(order_squared + order);
            std::copy(std::execution::par, A + offset,
                      A + offset + order_squared, std::begin(buffer[i]));
            offset = i * order;
            std::copy(std::execution::par, x + offset, x + offset + order,
                      std::begin(buffer[i]) + order_squared);
        }
    });

    auto indexes = std::valarray<int>(order_squared + order);
    at::parallel_for(0, order_squared + order, 1,
                     [&](int64_t start, int64_t end) {
                         for (auto i = start; i < end; i++) {
                             indexes[i] = i;
                         }
                     });

    std::inclusive_scan(
        std::execution::par, buffer.begin(), buffer.end(), buffer.begin(),
        std::bind(recurN_binary_op<scalar_t>, order, indexes,
                  std::placeholders::_1, std::placeholders::_2));

    at::parallel_for(0, total_steps, 1, [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; i++) {
            auto &result = buffer[i];
            auto offset = i * order;
            std::copy(std::execution::par, std::begin(result) + order_squared,
                      std::begin(result) + order_squared + order, out + offset);
        }
    });
}

at::Tensor mat_recur_N_order_cpu_impl(const at::Tensor &A, const at::Tensor &zi,
                                      const at::Tensor &x) {
    TORCH_CHECK(x.is_floating_point() || x.is_complex(),
                "Input must be floating point or complex");
    TORCH_CHECK(zi.scalar_type() == zi.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == A.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    TORCH_CHECK(x.size(2) == A.size(-1),
                "Last dimension of x must match last dimension of A");
    TORCH_CHECK(A.size(-1) == A.size(-2),
                "Last two dimensions of A must be equal");

    auto n_steps = x.size(1) + 1;  // +1 for the initial state
    auto n_batches = x.size(0);
    auto order = A.size(-1);

    auto A_contiguous = at::pad(A, {0, 0, 0, 0, 1, 0}).contiguous();
    auto x_contiguous = at::cat({zi.unsqueeze(1), x}, 1).contiguous();
    auto out = at::empty_like(x_contiguous);

    if (A.dim() == 4) {
        // Batch
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            x.scalar_type(), "host_batch_mat_recur_N_order", [&] {
                host_batch_mat_recur_N_order<scalar_t>(
                    A_contiguous.const_data_ptr<scalar_t>(),
                    x_contiguous.const_data_ptr<scalar_t>(),
                    out.mutable_data_ptr<scalar_t>(), n_steps * n_batches,
                    order);
            });
    } else {
        // Shared
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
            x.scalar_type(), "host_share_mat_recur_N_order", [&] {
                host_share_mat_recur_N_order<scalar_t>(
                    A_contiguous.const_data_ptr<scalar_t>(),
                    x_contiguous.const_data_ptr<scalar_t>(),
                    out.mutable_data_ptr<scalar_t>(), n_steps, order,
                    n_batches);
            });
    }
    return out.reshape({n_batches, n_steps, order})
        .slice(1, 1, n_steps)
        .contiguous();  // Remove the initial state from the output
}

TORCH_LIBRARY_IMPL(philtorch, CPU, m) {
    m.impl("recurN", &mat_recur_N_order_cpu_impl);
}