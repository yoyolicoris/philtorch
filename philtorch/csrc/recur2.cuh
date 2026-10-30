#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/device/device_scan.cuh>
#include <cuda/std/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <ATen/OpMathType.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>

// The affine map h -> A h + x of a step, or of a run of steps, of M states.
template <int M, typename T>
struct affine
{
    T a[M * M];
    T x[M];
};

template <int M, typename T>
struct affine_binary_op
{
    // q after p: A = Aq Ap, x = Aq xp + xq.
    __host__ __device__ affine<M, T> operator()(const affine<M, T> &p,
                                                const affine<M, T> &q) const
    {
        affine<M, T> r;
#pragma unroll
        for (int i = 0; i < M; ++i)
        {
#pragma unroll
            for (int j = 0; j < M; ++j)
            {
                T s(0);
#pragma unroll
                for (int k = 0; k < M; ++k)
                    s += q.a[i * M + k] * p.a[k * M + j];
                r.a[i * M + j] = s;
            }
            T s = q.x[i];
#pragma unroll
            for (int k = 0; k < M; ++k)
                s += q.a[i * M + k] * p.x[k];
            r.x[i] = s;
        }
        return r;
    }
};

// Item i of the scan is step s = i % n_steps of signal b = i / n_steps, read
// in place: step 0 is the initial state, as x with A = 0, and step s > 0 is
// (A[s - 1], x[s - 1]), A's batch and time strides 0 where it is shared.
template <int M, typename scalar_t, typename acc_t>
struct recur_input_op
{
    const scalar_t *A, *zi, *x;
    int n_steps;
    int64_t stride_b, stride_t;
    __host__ __device__ affine<M, acc_t> operator()(int i) const
    {
        int64_t b = i / n_steps;
        int s = i % n_steps;
        affine<M, acc_t> r;
        if (s == 0)
        {
#pragma unroll
            for (int k = 0; k < M * M; ++k)
                r.a[k] = acc_t(0);
#pragma unroll
            for (int k = 0; k < M; ++k)
                r.x[k] = acc_t(zi[b * M + k]);
            return r;
        }
        const scalar_t *a = A + b * stride_b + (s - 1) * stride_t;
        const scalar_t *v = x + (b * (n_steps - 1) + s - 1) * M;
#pragma unroll
        for (int k = 0; k < M * M; ++k)
            r.a[k] = acc_t(a[k]);
#pragma unroll
        for (int k = 0; k < M; ++k)
            r.x[k] = acc_t(v[k]);
        return r;
    }
};

// Writes the state after step s > 0 to out[b, s - 1]; step 0's is zi.
template <int M, typename scalar_t, typename acc_t>
struct recur_output_op
{
    scalar_t *out;
    int n_steps;
    __host__ __device__ void operator()(int i, const affine<M, acc_t> &state) const
    {
        int64_t b = i / n_steps;
        int s = i % n_steps;
        if (s == 0)
            return;
        scalar_t *o = out + (b * (n_steps - 1) + s - 1) * M;
#pragma unroll
        for (int k = 0; k < M; ++k)
            o[k] = scalar_t(state.x[k]);
    }
};

struct index2key
{
    int n_steps;
    __host__ __device__ int operator()(int i) const { return i / n_steps; }
};

// Scan each run of equal keys with scan_op, like thrust::inclusive_scan_by_key,
// but asynchronously on PyTorch's current stream, with scratch memory from its
// caching allocator. thrust::device runs on the legacy default stream and
// allocates with cudaMalloc, which synchronized the whole device on every call.
template <typename KeysIt, typename InputIt, typename OutputIt, typename ScanOp>
void scan_by_key_on_current_stream(KeysIt keys, InputIt input, OutputIt output,
                                   ScanOp scan_op, int64_t num_items)
{
    TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
                "the recurrence has ", num_items,
                " steps over all batches, but at most INT_MAX are supported");
    if (num_items == 0)
        return;
    auto stream = c10::cuda::getCurrentCUDAStream();
    ::cuda::std::equal_to<int> equality_op;
    size_t temp_storage_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveScanByKey(
        nullptr, temp_storage_bytes, keys, input, output, scan_op,
        static_cast<int>(num_items), equality_op, stream));
    auto temp_storage =
        c10::cuda::CUDACachingAllocator::get()->allocate(temp_storage_bytes);
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveScanByKey(
        temp_storage.get(), temp_storage_bytes, keys, input, output, scan_op,
        static_cast<int>(num_items), equality_op, stream));
}

template <int M>
void check_recur_inputs(const at::Tensor &A, const at::Tensor &zi, const at::Tensor &x)
{
    TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                "zi must have the same scalar type as input");
    TORCH_CHECK(A.scalar_type() == x.scalar_type(),
                "A must have the same scalar type as input");
    TORCH_CHECK(A.device() == x.device() && zi.device() == x.device(),
                "A and zi must be on the same device as input");
    TORCH_CHECK(x.dim() == 3 && x.size(2) == M,
                "Input x must have a last dimension of size ", M);
    TORCH_CHECK(A.size(-1) == M && A.size(-2) == M,
                "Last two dimensions of A must be of size ", M);
    TORCH_CHECK(zi.dim() == 2 && zi.size(0) == x.size(0) && zi.size(1) == M,
                "zi must be of shape (B, ", M, ")");
}

// The recurrence of M states, A read with these strides; inputs checked.
// dispatch(type, f) calls f(scalar_t{}) for a scalar type of the dtypes the op
// supports: a macro needs its name as a literal, so the caller dispatches.
template <int M, typename Dispatch>
at::Tensor recur_small_launch(const at::Tensor &A, const at::Tensor &zi,
                              const at::Tensor &x, int64_t stride_b,
                              int64_t stride_t, Dispatch dispatch)
{
    auto n_steps = x.size(1) + 1; // +1 for the initial state
    auto A_contiguous = A.contiguous();
    auto zi_contiguous = zi.contiguous();
    auto x_contiguous = x.contiguous();
    auto out = at::empty_like(x_contiguous);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    dispatch(x.scalar_type(), [&](auto tag) {
        using scalar_t = decltype(tag);
        using acc_t = at::opmath_type<scalar_t>; // the state stays in the accumulation type
        thrust::counting_iterator<int> iter(0);
        index2key key_op{static_cast<int>(n_steps)};
        scan_by_key_on_current_stream(
            thrust::make_transform_iterator(iter, key_op),
            thrust::make_transform_iterator(
                iter, recur_input_op<M, scalar_t, acc_t>{
                          A_contiguous.const_data_ptr<scalar_t>(),
                          zi_contiguous.const_data_ptr<scalar_t>(),
                          x_contiguous.const_data_ptr<scalar_t>(),
                          static_cast<int>(n_steps), stride_b, stride_t}),
            thrust::make_tabulate_output_iterator(recur_output_op<M, scalar_t, acc_t>{
                out.mutable_data_ptr<scalar_t>(), static_cast<int>(n_steps)}),
            affine_binary_op<M, acc_t>(), x.size(0) * n_steps);
    });
    return out;
}
