#include <assert.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <torch/script.h>
#include <torch/torch.h>

// template <typename T>
// struct sqm2_pair {
//     T mat_entries[4];
//     T vec_entries[2];
// };

template <typename T>
using sqm2_pair = cuda::std::tuple<T, T, T, T, T, T>;

template <typename T>
struct recur2_binary_op {
    __host__ __device__ sqm2_pair<T> operator()(const sqm2_pair<T> &a,
                                                const sqm2_pair<T> &b) const {
        auto a_a = thrust::get<0>(a);
        auto a_b = thrust::get<1>(a);
        auto a_c = thrust::get<2>(a);
        auto a_d = thrust::get<3>(a);
        auto b_a = thrust::get<0>(b);
        auto b_b = thrust::get<1>(b);
        auto b_c = thrust::get<2>(b);
        auto b_d = thrust::get<3>(b);

        auto b_a_vec = thrust::get<4>(b);
        auto b_b_vec = thrust::get<5>(b);
        auto a_a_vec = thrust::get<4>(a);
        auto a_b_vec = thrust::get<5>(a);

        return cuda::std::make_tuple(
            b_a * a_a + b_b * a_c, b_a * a_b + b_b * a_d, b_c * a_a + b_d * a_c,
            b_c * a_b + b_d * a_d, b_a * a_a_vec + b_b * a_b_vec + a_a_vec,
            b_c * a_a_vec + b_d * a_b_vec + b_b_vec);
    }
};

template <typename scalar_t>
void matrix_recurrence_second_order(const scalar_t *A, const scalar_t *x,
                                    scalar_t *out, int n_steps) {
    thrust::device_vector<sqm2_pair<scalar_t>> pairs(n_steps);

    thrust::counting_iterator<int> A_iter(0);
    // Initialize input_states and output_states
    thrust::copy(thrust::device,
                 thrust::make_zip_iterator(A, A + n_steps, A + n_steps * 2,
                                           A + n_steps * 3, x, x + n_steps),
                 thrust::make_zip_iterator(A + n_steps, A + n_steps * 2,
                                           A + n_steps * 3, A + n_steps * 4,
                                           x + n_steps, x + n_steps * 2),
                 pairs.begin());

    recur2_binary_op<scalar_t> binary_op;
    thrust::inclusive_scan(thrust::device, pairs.begin(), pairs.end(),
                           pairs.begin(), binary_op);

    auto take_last2 = [](const sqm2_pair<scalar_t> &p) {
        return thrust::make_tuple(thrust::get<4>(p), thrust::get<5>(p));
    };

    thrust::transform(thrust::device, pairs.begin(), pairs.end(),
                      thrust::make_zip_iterator(out, out + n_steps),
                      take_last2);
}
