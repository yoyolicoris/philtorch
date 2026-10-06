#include <assert.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "recur2.cuh"

template <typename T>
struct recur_binary_op
{
    __host__ __device__ cuda::std::tuple<T, T> operator()(
        const cuda::std::tuple<T, T> &a,
        const cuda::std::tuple<T, T> &b) const
    {
        auto [a_first, a_second] = a;
        auto [b_first, b_second] = b;
        return cuda::std::make_tuple(a_first * b_first,
                                     a_second * b_first + b_second);
    }
};

template <typename T>
struct take_second
{
    __host__ __device__ T operator()(const cuda::std::tuple<T, T> &state) const
    {
        return thrust::get<1>(state);
    }
};

template <typename T>
struct lti_batch_recur_input_op
{
    const T *decays;
    int n_steps;
    __host__ __device__ cuda::std::tuple<T, T> operator()(int i, const T &x) const
    {
        int idx = i / n_steps;
        int offset = i % n_steps;
        if (offset > 0)
            return thrust::make_tuple(decays[idx], x);
        return thrust::make_tuple(0, x);
    }
};

template <typename T>
struct lti_shared_recur_input_op
{
    // Read on the device, so the host doesn't wait for a to be computed.
    const T *decay;
    int n_steps;
    __host__ __device__ cuda::std::tuple<T, T> operator()(int i, const T &x) const
    {
        int offset = i % n_steps;
        if (offset > 0)
            return thrust::make_tuple(*decay, x);
        return thrust::make_tuple(0, x);
    }
};

template <typename scalar_t>
void lti_batch_linear_recurrence(const scalar_t *decays,
                                 const scalar_t *impulses,
                                 scalar_t *out, int n_steps, int64_t total_steps)
{
    thrust::counting_iterator<int> it(0);
    auto batch_input_op = thrust::make_zip_function(lti_batch_recur_input_op<scalar_t>{decays, n_steps});
    index2key key_op{n_steps};

    scan_by_key_on_current_stream(
        thrust::make_transform_iterator(it, key_op),
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(it, impulses), batch_input_op),
        thrust::make_transform_output_iterator(out, take_second<scalar_t>()),
        recur_binary_op<scalar_t>(), total_steps);
}

template <typename scalar_t>
void lti_shared_linear_recurrence(const scalar_t *decay,
                                  const scalar_t *impulses,
                                  scalar_t *out, int n_steps, int64_t total_steps)
{
    thrust::counting_iterator<int> it(0);
    auto shared_input_op = thrust::make_zip_function(lti_shared_recur_input_op<scalar_t>{decay, n_steps});
    index2key key_op{n_steps};

    scan_by_key_on_current_stream(
        thrust::make_transform_iterator(it, key_op),
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(it, impulses), shared_input_op),
        thrust::make_transform_output_iterator(out, take_second<scalar_t>()),
        recur_binary_op<scalar_t>(), total_steps);
}

at::Tensor lti_recur_cuda_impl(const at::Tensor &a,
                               const at::Tensor &zi, const at::Tensor &x)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(a.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(a.dim() <= 1, "A must be a vector or a scalar");
    TORCH_CHECK(zi.dim() == 1, "zi must be a vector");
    TORCH_CHECK(x.size(1) > 0, "x must contain at least one time step");

    auto n_steps = x.size(1) + 1; // +1 for the initial state
    auto n_batches = x.size(0);
    auto x_contiguous =
        at::cat({zi.unsqueeze(1), x}, 1).contiguous();
    auto a_contiguous = a.contiguous();
    auto output = at::empty_like(x_contiguous);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    if (a.dim() == 1 && a.numel() == n_batches)
    {

        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kHalf, at::kBFloat16,
            x.scalar_type(), "lti_batch_linear_recurrence", [&]
            { lti_batch_linear_recurrence<scalar_t>(
                  a_contiguous.const_data_ptr<scalar_t>(),
                  x_contiguous.const_data_ptr<scalar_t>(),
                  output.mutable_data_ptr<scalar_t>(),
                  n_steps, n_batches * n_steps); });
    }
    else
    {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
            at::kHalf, at::kBFloat16,
            x.scalar_type(), "lti_shared_linear_recurrence", [&]
            { lti_shared_linear_recurrence<scalar_t>(
                  a_contiguous.const_data_ptr<scalar_t>(),
                  x_contiguous.const_data_ptr<scalar_t>(),
                  output.mutable_data_ptr<scalar_t>(),
                  n_steps, n_batches * n_steps); });
    }

    return output.slice(1, 1, output.size(1))
        .contiguous(); // Remove the initial state from the output
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m) { m.impl("lti_recur", &lti_recur_cuda_impl); }