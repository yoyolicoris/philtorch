#include <Python.h>
#include <assert.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <torch/script.h>
#include <torch/torch.h>

template <typename T>
using sqm2_pair = cuda::std::tuple<T, T, T, T, T, T>;

template <typename T>
struct recur2_binary_op {
    __host__ __device__ sqm2_pair<T> operator()(const sqm2_pair<T> &a,
                                                const sqm2_pair<T> &b) const {
        auto [a_a, a_b, a_c, a_d, a_a_vec, a_b_vec] = a;
        auto [b_a, b_b, b_c, b_d, b_a_vec, b_b_vec] = b;

        return cuda::std::make_tuple(
            b_a * a_a + b_b * a_c, b_a * a_b + b_b * a_d, b_c * a_a + b_d * a_c,
            b_c * a_b + b_d * a_d, b_a * a_a_vec + b_b * a_b_vec + b_a_vec,
            b_c * a_a_vec + b_d * a_b_vec + b_b_vec);
    }
};

template <typename T>
struct output_unary_op {
    __host__ __device__ cuda::std::tuple<T, T> operator()(
        const sqm2_pair<T> &state) const {
        return thrust::make_tuple(thrust::get<4>(state), thrust::get<5>(state));
    }
};

template <typename T>
struct share_A_input_op {
    const T *A;
    int n_steps;
    __host__ __device__ sqm2_pair<T> operator()(int i, const T &x_0,
                                                const T &x_1) const {
        int idx = i % n_steps;
        return thrust::make_tuple(A[idx], A[idx + n_steps],
                                  A[idx + n_steps * 2], A[idx + n_steps * 3],
                                  x_0, x_1);
    }
};

template <typename scalar_t>
void batch_mat_recur_second_order(const scalar_t *A, const scalar_t *x,
                                  scalar_t *out, int n_steps) {
    thrust::inclusive_scan(
        thrust::device,
        thrust::make_zip_iterator(A, A + n_steps, A + n_steps * 2,
                                  A + n_steps * 3, x, x + n_steps),
        thrust::make_zip_iterator(A + n_steps, A + n_steps * 2, A + n_steps * 3,
                                  A + n_steps * 4, x + n_steps,
                                  x + n_steps * 2),
        thrust::make_transform_output_iterator(
            thrust::make_zip_iterator(out, out + n_steps),
            output_unary_op<scalar_t>()),
        recur2_binary_op<scalar_t>());
}

template <typename scalar_t>
void share_mat_recur_second_order(const scalar_t *A, const scalar_t *x,
                                  scalar_t *out, int n_steps, int n_batches) {
    auto total_steps = n_steps * n_batches;
    thrust::counting_iterator<int> iter(0);
    auto share_input_op =
        thrust::make_zip_function(share_A_input_op<scalar_t>{A, n_steps});
    thrust::inclusive_scan(
        thrust::device,
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(iter, x, x + total_steps),
            share_input_op),
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(iter + total_steps, x + total_steps,
                                      x + total_steps * 2),
            share_input_op),
        thrust::make_transform_output_iterator(
            thrust::make_zip_iterator(out, out + total_steps),
            output_unary_op<scalar_t>()),
        recur2_binary_op<scalar_t>());
}

at::Tensor mat_recur_second_order_cuda_impl(const at::Tensor &A,
                                            const at::Tensor &zi,
                                            const at::Tensor &x) {
    TORCH_CHECK(zi.scalar_type() == zi.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == A.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    TORCH_CHECK(x.size(2) == 2, "Input x must have a last dimension of size 2");
    TORCH_CHECK(A.size(-1) == 2 && A.size(-2) == 2,
                "Last two dimensions of A must be of size 2");

    auto n_steps = x.size(1) + 1;  // +1 for the initial state
    auto n_batches = x.size(0);

    auto A_contiguous =
        at::pad(A, {0, 0, 0, 0, 1, 0}).view({-1, 4}).t().contiguous();
    auto x_contiguous =
        at::cat({zi.unsqueeze(1), x}, 1).view({-1, 2}).t().contiguous();
    auto out = at::empty_like(x_contiguous);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    if (A.dim() == 4) {
        // Batch
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
            x.scalar_type(), "batch_mat_recur_second_order", [&] {
                batch_mat_recur_second_order<scalar_t>(
                    A_contiguous.const_data_ptr<scalar_t>(),
                    x_contiguous.const_data_ptr<scalar_t>(),
                    out.mutable_data_ptr<scalar_t>(), n_steps * n_batches);
            });
    } else {
        // Shared
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX(
            x.scalar_type(), "share_mat_recur_second_order", [&] {
                share_mat_recur_second_order<scalar_t>(
                    A_contiguous.const_data_ptr<scalar_t>(),
                    x_contiguous.const_data_ptr<scalar_t>(),
                    out.mutable_data_ptr<scalar_t>(), n_steps, n_batches);
            });
    }
    return out.t()
        .reshape({n_batches, n_steps, 2})
        .slice(1, 1, n_steps)
        .contiguous();  // Remove the initial state from the output
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m) {
    m.impl("recur2", &mat_recur_second_order_cuda_impl);
}