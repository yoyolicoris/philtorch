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

#include "recur2.cuh"

template <typename scalar_t>
void lti_batch_mat_recur_second_order(const scalar_t *A, const scalar_t *x,
                                      scalar_t *out, int n_steps, int n_batches)
{
    auto total_steps = n_steps * n_batches;
    thrust::counting_iterator<int> iter(0);
    auto batch_input_op =
        thrust::make_zip_function(lti_batch_A_input_op<scalar_t>{A, n_steps});
    thrust::inclusive_scan(
        thrust::device,
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(iter, x, x + total_steps),
            batch_input_op),
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(iter + total_steps, x + total_steps,
                                      x + total_steps * 2),
            batch_input_op),
        thrust::make_transform_output_iterator(
            thrust::make_zip_iterator(out, out + total_steps),
            output_unary_op<scalar_t>()),
        recur2_binary_op<scalar_t>());
}

template <typename scalar_t>
void lti_share_mat_recur_second_order(const scalar_t *A, const scalar_t *x,
                                      scalar_t *out, int n_steps, int n_batches)
{
    auto total_steps = n_steps * n_batches;
    thrust::counting_iterator<int> iter(0);
    auto share_input_op =
        thrust::make_zip_function(lti_share_A_input_op<scalar_t>{A, n_steps});
    thrust::inclusive_scan(
        thrust::device,
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(iter, x, x + total_steps),
            share_input_op),
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(iter + total_steps, x + total_steps, x + total_steps * 2),
            share_input_op),
        thrust::make_transform_output_iterator(
            thrust::make_zip_iterator(out, out + total_steps),
            output_unary_op<scalar_t>()),
        recur2_binary_op<scalar_t>());
}

at::Tensor lti_mat_recur_second_order_cuda_impl(const at::Tensor &A,
                                                const at::Tensor &zi,
                                                const at::Tensor &x)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be a 2D or 3D tensor");
    TORCH_CHECK(x.size(2) == 2, "Input x must have a last dimension of size 2");
    TORCH_CHECK(A.size(-1) == 2 && A.size(-2) == 2,
                "Last two dimensions of A must be of size 2");

    auto n_steps = x.size(1) + 1; // +1 for the initial state
    auto n_batches = x.size(0);

    auto A_contiguous = A.contiguous();
    auto x_contiguous =
        at::cat({zi.unsqueeze(1), x}, 1).view({-1, 2}).t().contiguous();
    auto out = at::empty_like(x_contiguous);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    if (A.dim() == 3)
    {
        // Batch
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            at::kLong, x.scalar_type(), "lti_batch_mat_recur_second_order", [&]
            { lti_batch_mat_recur_second_order<scalar_t>(
                  A_contiguous.const_data_ptr<scalar_t>(),
                  x_contiguous.const_data_ptr<scalar_t>(),
                  out.mutable_data_ptr<scalar_t>(), n_steps, n_batches); });
    }
    else
    {
        // Shared
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
            at::kLong, x.scalar_type(), "lti_share_mat_recur_second_order", [&]
            { lti_share_mat_recur_second_order<scalar_t>(
                  A_contiguous.const_data_ptr<scalar_t>(),
                  x_contiguous.const_data_ptr<scalar_t>(),
                  out.mutable_data_ptr<scalar_t>(), n_steps, n_batches); });
    }
    return out.t()
        .reshape({n_batches, n_steps, 2})
        .slice(1, 1, n_steps)
        .contiguous(); // Remove the initial state from the output
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("lti_recur2", &lti_mat_recur_second_order_cuda_impl);
}