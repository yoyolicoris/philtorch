#pragma once

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/device/device_scan.cuh>
#include <cuda/std/functional>
#include <thrust/tuple.h>

#include <cstdint>
#include <limits>

// Six-entry tuple used by second-order recurrence (A0,A1,A2,A3,x0,x1)
template <typename T>
using sqm2_pair = thrust::tuple<T, T, T, T, T, T>;

template <typename T>
struct recur2_binary_op
{
    __host__ __device__ sqm2_pair<T> operator()(const sqm2_pair<T> &a,
                                                const sqm2_pair<T> &b) const
    {
        auto [a_a, a_b, a_c, a_d, a_a_vec, a_b_vec] = a;
        auto [b_a, b_b, b_c, b_d, b_a_vec, b_b_vec] = b;

        return cuda::std::make_tuple(
            b_a * a_a + b_b * a_c, b_a * a_b + b_b * a_d, b_c * a_a + b_d * a_c,
            b_c * a_b + b_d * a_d, b_a * a_a_vec + b_b * a_b_vec + b_a_vec,
            b_c * a_a_vec + b_d * a_b_vec + b_b_vec);
    }
};

template <typename T, typename acc_t>
struct output_unary_op
{
    __host__ __device__ cuda::std::tuple<T, T> operator()(
        const sqm2_pair<acc_t> &state) const
    {
        return thrust::make_tuple(static_cast<T>(thrust::get<4>(state)),
                                  static_cast<T>(thrust::get<5>(state)));
    }
};

template <typename T, typename acc_t>
struct batch_A_input_op
{
    __host__ __device__ sqm2_pair<acc_t> operator()(
        const T &a, const T &b, const T &c, const T &d, const T &x_0,
        const T &x_1) const
    {
        return thrust::make_tuple(
            static_cast<acc_t>(a), static_cast<acc_t>(b),
            static_cast<acc_t>(c), static_cast<acc_t>(d),
            static_cast<acc_t>(x_0), static_cast<acc_t>(x_1));
    }
};

template <typename T, typename acc_t>
struct share_A_input_op
{
    const T *A;
    int n_steps;
    __host__ __device__ sqm2_pair<acc_t> operator()(int i, const T &x_0,
                                                    const T &x_1) const
    {
        // A holds four rows of n_steps entries; index in 64 bits, since
        // 4 * n_steps can exceed INT_MAX.
        int64_t idx = i % n_steps;
        int64_t stride = n_steps;
        return thrust::make_tuple(
            static_cast<acc_t>(A[idx]), static_cast<acc_t>(A[idx + stride]),
            static_cast<acc_t>(A[idx + stride * 2]),
            static_cast<acc_t>(A[idx + stride * 3]), static_cast<acc_t>(x_0),
            static_cast<acc_t>(x_1));
    }
};

template <typename T, typename acc_t>
struct lti_batch_A_input_op
{
    const T *A;
    int n_steps;
    __host__ __device__ sqm2_pair<acc_t> operator()(int i, const T &x_0,
                                                    const T &x_1) const
    {
        int64_t idx = static_cast<int64_t>(i / n_steps) * 4;
        int offset = i % n_steps;
        if (offset > 0)
            return thrust::make_tuple(
                static_cast<acc_t>(A[idx]), static_cast<acc_t>(A[idx + 1]),
                static_cast<acc_t>(A[idx + 2]), static_cast<acc_t>(A[idx + 3]),
                static_cast<acc_t>(x_0), static_cast<acc_t>(x_1));
        return thrust::make_tuple(acc_t(0), acc_t(0), acc_t(0), acc_t(0),
                                  static_cast<acc_t>(x_0),
                                  static_cast<acc_t>(x_1));
    }
};

template <typename T, typename acc_t>
struct lti_share_A_input_op
{
    const T *A;
    int n_steps;
    __host__ __device__ sqm2_pair<acc_t> operator()(int i, const T &x_0,
                                                    const T &x_1) const
    {
        int offset = i % n_steps;
        if (offset > 0)
            return thrust::make_tuple(
                static_cast<acc_t>(A[0]), static_cast<acc_t>(A[1]),
                static_cast<acc_t>(A[2]), static_cast<acc_t>(A[3]),
                static_cast<acc_t>(x_0), static_cast<acc_t>(x_1));
        return thrust::make_tuple(acc_t(0), acc_t(0), acc_t(0), acc_t(0),
                                  static_cast<acc_t>(x_0),
                                  static_cast<acc_t>(x_1));
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
