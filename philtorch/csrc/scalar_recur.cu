#include <assert.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>
#include <ATen/OpMathType.h>
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

template <typename T, typename acc_t>
struct take_second
{
    __host__ __device__ T operator()(const cuda::std::tuple<acc_t, acc_t> &state) const
    {
        return thrust::get<1>(state);
    }
};

// The decay of sequence b at step t is decays[b * stride_b + t * stride_t]:
// strides (0, 0) read one shared decay, (1, 0) one per sequence, (T, 1) one
// per step. The initial state zi[b] is folded into the first step, whose
// decay is then dropped, so the scan runs over the B x T inputs alone.
template <typename T, typename acc_t>
struct scalar_recur_input_op
{
    // Read on the device, so the host doesn't wait for the decays to be computed.
    const T *decays;
    const T *zi;
    int n_steps;
    int64_t stride_b;
    int64_t stride_t;
    __host__ __device__ cuda::std::tuple<acc_t, acc_t> operator()(int i, const T &x) const
    {
        int64_t idx = i / n_steps;
        int64_t offset = i % n_steps;
        acc_t a = static_cast<acc_t>(decays[idx * stride_b + offset * stride_t]);
        if (offset > 0)
            return thrust::make_tuple(a, static_cast<acc_t>(x));
        return thrust::make_tuple(
            acc_t(0), a * static_cast<acc_t>(zi[idx]) + static_cast<acc_t>(x));
    }
};

template <typename scalar_t>
void scalar_recurrence(const scalar_t *decays, int64_t stride_b,
                       int64_t stride_t, const scalar_t *zi,
                       const scalar_t *impulses, scalar_t *out,
                       int n_steps, int64_t total_steps)
{
    using acc_t = at::opmath_type<scalar_t>; // the state stays in the accumulation type
    thrust::counting_iterator<int> it(0);
    auto input_op = thrust::make_zip_function(
        scalar_recur_input_op<scalar_t, acc_t>{decays, zi, n_steps, stride_b, stride_t});
    index2key key_op{n_steps};

    scan_by_key_on_current_stream(
        thrust::make_transform_iterator(it, key_op),
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(it, impulses), input_op),
        thrust::make_transform_output_iterator(out, take_second<scalar_t, acc_t>()),
        recur_binary_op<acc_t>(), total_steps);
}

// h[t] = decays[b * stride_b + t * stride_t] h[t - 1] + x[b, t], h[-1] = zi[b]
at::Tensor scalar_recur_cuda_launch(const at::Tensor &decays, int64_t stride_b,
                                    int64_t stride_t, const at::Tensor &zi,
                                    const at::Tensor &x)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    auto decays_contiguous = decays.contiguous();
    auto zi_contiguous = zi.contiguous();
    auto x_contiguous = x.contiguous();
    auto output = at::empty_like(x_contiguous);
    auto n_steps = x.size(1);
    auto n_batches = x.size(0);

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
        at::kLong, at::kHalf, at::kBFloat16,
        x.scalar_type(), "scalar_recurrence", [&]
        { scalar_recurrence<scalar_t>(
              decays_contiguous.const_data_ptr<scalar_t>(), stride_b, stride_t,
              zi_contiguous.const_data_ptr<scalar_t>(),
              x_contiguous.const_data_ptr<scalar_t>(),
              output.mutable_data_ptr<scalar_t>(),
              static_cast<int>(n_steps), n_batches * n_steps); });

    return output;
}

at::Tensor lti_recur_cuda_impl(const at::Tensor &a,
                               const at::Tensor &zi, const at::Tensor &x)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(a.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(a.device() == x.device(),
                "A must be on the same device as input");
    TORCH_CHECK(a.dim() <= 1, "A must be a vector or a scalar");
    TORCH_CHECK(zi.dim() == 1, "zi must be a vector");
    TORCH_CHECK(x.dim() == 2 && zi.size(0) == x.size(0),
                "x must be (B, T) and zi (B,)");
    TORCH_CHECK(x.size(1) > 0, "x must contain at least one time step");

    // One decay per sequence, or one shared by all of them.
    int64_t stride_b = a.dim() == 1 && a.numel() == x.size(0) ? 1 : 0;
    return scalar_recur_cuda_launch(a, stride_b, 0, zi, x);
}

// h[t] = decay[t] h[t - 1] + impulse[t]: the scalar recurrence with a
// (B, T) A, one coefficient per step. Like scan_cpu in host_recur.cpp.
at::Tensor scan_cuda(const at::Tensor &impulse, const at::Tensor &decay,
                     const at::Tensor &init)
{
    TORCH_CHECK(impulse.dim() == 2 && decay.sizes() == impulse.sizes() &&
                    init.dim() == 1 && init.size(0) == impulse.size(0),
                "impulse and decay must be (B, T) and init (B,)");
    TORCH_CHECK(impulse.scalar_type() == at::kFloat ||
                    impulse.scalar_type() == at::kDouble ||
                    impulse.scalar_type() == at::kComplexFloat ||
                    impulse.scalar_type() == at::kComplexDouble,
                "Input must be float32, float64, complex64 or complex128");
    TORCH_CHECK(decay.scalar_type() == impulse.scalar_type(),
                "decay must have the same scalar type as impulse");
    TORCH_CHECK(init.scalar_type() == impulse.scalar_type(),
                "init must have the same scalar type as impulse");
    TORCH_CHECK(decay.device() == impulse.device(),
                "decay must be on the same device as impulse");
    TORCH_CHECK(init.device() == impulse.device(),
                "init must be on the same device as impulse");

    return scalar_recur_cuda_launch(decay, impulse.size(1), 1, init, impulse);
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("lti_recur", &lti_recur_cuda_impl);
    m.impl("scan", &scan_cuda);
}
