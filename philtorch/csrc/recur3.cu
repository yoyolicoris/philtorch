#include <c10/cuda/CUDAGuard.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <torch/torch.h>

#include "recur2.cuh"

// The affine map h -> A h + x of a step, or of a run of steps.
template <typename T>
struct affine3
{
    T a[9];
    T x[3];
};

template <typename T>
struct recur3_binary_op
{
    // q after p: A = Aq Ap, x = Aq xp + xq.
    __host__ __device__ affine3<T> operator()(const affine3<T> &p, const affine3<T> &q) const
    {
        affine3<T> r;
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
#pragma unroll
            for (int j = 0; j < 3; ++j)
                r.a[i * 3 + j] = q.a[i * 3] * p.a[j] + q.a[i * 3 + 1] * p.a[3 + j] +
                                 q.a[i * 3 + 2] * p.a[6 + j];
            r.x[i] = q.a[i * 3] * p.x[0] + q.a[i * 3 + 1] * p.x[1] + q.a[i * 3 + 2] * p.x[2] +
                     q.x[i];
        }
        return r;
    }
};

// Item i of the scan is step s = i % n_steps of signal b = i / n_steps, read
// in place: step 0 is the initial state, as x with A = 0, and step s > 0 is
// (A[s - 1], x[s - 1]), A's batch and time strides 0 where it is shared.
template <typename scalar_t, typename acc_t>
struct recur3_input_op
{
    const scalar_t *A, *zi, *x;
    int n_steps;
    int64_t stride_b, stride_t;
    __host__ __device__ affine3<acc_t> operator()(int i) const
    {
        int64_t b = i / n_steps;
        int s = i % n_steps;
        affine3<acc_t> r;
        if (s == 0)
        {
#pragma unroll
            for (int k = 0; k < 9; ++k)
                r.a[k] = acc_t(0);
#pragma unroll
            for (int k = 0; k < 3; ++k)
                r.x[k] = acc_t(zi[b * 3 + k]);
            return r;
        }
        const scalar_t *a = A + b * stride_b + (s - 1) * stride_t;
        const scalar_t *v = x + (b * (n_steps - 1) + s - 1) * 3;
#pragma unroll
        for (int k = 0; k < 9; ++k)
            r.a[k] = acc_t(a[k]);
#pragma unroll
        for (int k = 0; k < 3; ++k)
            r.x[k] = acc_t(v[k]);
        return r;
    }
};

// Writes the state after step s > 0 to out[b, s - 1]; step 0's is zi.
template <typename scalar_t, typename acc_t>
struct recur3_output_op
{
    scalar_t *out;
    int n_steps;
    __host__ __device__ void operator()(int i, const affine3<acc_t> &state) const
    {
        int64_t b = i / n_steps;
        int s = i % n_steps;
        if (s == 0)
            return;
        scalar_t *o = out + (b * (n_steps - 1) + s - 1) * 3;
#pragma unroll
        for (int k = 0; k < 3; ++k)
            o[k] = scalar_t(state.x[k]);
    }
};

// The recurrence of 3 states, A read with these strides; inputs checked.
at::Tensor recur3_launch(const at::Tensor &A, const at::Tensor &zi, const at::Tensor &x,
                         int64_t stride_b, int64_t stride_t)
{
    auto n_steps = x.size(1) + 1; // +1 for the initial state
    auto A_contiguous = A.contiguous();
    auto zi_contiguous = zi.contiguous();
    auto x_contiguous = x.contiguous();
    auto out = at::empty_like(x_contiguous);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    // float64 and int64 run faster in the Triton kernels on GPUs measured, and
    // complex128's 24 values overflow CUB's default shared memory.
    AT_DISPATCH_V2(
        x.scalar_type(), "recur3", AT_WRAP([&] {
            using acc_t = at::opmath_type<scalar_t>;
            thrust::counting_iterator<int> iter(0);
            index2key key_op{static_cast<int>(n_steps)};
            scan_by_key_on_current_stream(
                thrust::make_transform_iterator(iter, key_op),
                thrust::make_transform_iterator(
                    iter, recur3_input_op<scalar_t, acc_t>{
                              A_contiguous.const_data_ptr<scalar_t>(),
                              zi_contiguous.const_data_ptr<scalar_t>(),
                              x_contiguous.const_data_ptr<scalar_t>(),
                              static_cast<int>(n_steps), stride_b, stride_t}),
                thrust::make_tabulate_output_iterator(recur3_output_op<scalar_t, acc_t>{
                    out.mutable_data_ptr<scalar_t>(), static_cast<int>(n_steps)}),
                recur3_binary_op<acc_t>(), x.size(0) * n_steps);
        }),
        at::kFloat, at::kHalf, at::kBFloat16, at::kComplexFloat);
    return out;
}

void check_recur3_inputs(const at::Tensor &A, const at::Tensor &zi, const at::Tensor &x)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(A.device() == x.device() && zi.device() == x.device(),
                "A and zi must be on the same device as input");
    TORCH_CHECK(x.dim() == 3 && x.size(2) == 3, "Input x must have a last dimension of size 3");
    TORCH_CHECK(A.size(-1) == 3 && A.size(-2) == 3, "Last two dimensions of A must be of size 3");
    TORCH_CHECK(zi.dim() == 2 && zi.size(0) == x.size(0) && zi.size(1) == 3,
                "zi must be of shape (B, 3)");
}

at::Tensor mat_recur_third_order_cuda_impl(const at::Tensor &A, const at::Tensor &zi,
                                           const at::Tensor &x)
{
    check_recur3_inputs(A, zi, x);
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    TORCH_CHECK(A.size(-3) == x.size(1), "A must have one matrix per step of x");
    TORCH_CHECK(A.dim() == 3 || A.size(0) == x.size(0), "A must have x's batch size");
    return recur3_launch(A, zi, x, A.dim() == 4 ? x.size(1) * 9 : 0, 9);
}

at::Tensor lti_mat_recur_third_order_cuda_impl(const at::Tensor &A, const at::Tensor &zi,
                                               const at::Tensor &x)
{
    check_recur3_inputs(A, zi, x);
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be a 2D or 3D tensor");
    TORCH_CHECK(A.dim() == 2 || A.size(0) == x.size(0), "A must have x's batch size");
    return recur3_launch(A, zi, x, A.dim() == 3 ? 9 : 0, 0);
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("recur3", &mat_recur_third_order_cuda_impl);
    m.impl("lti_recur3", &lti_mat_recur_third_order_cuda_impl);
}
