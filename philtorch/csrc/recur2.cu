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

template <typename T>
struct sqm2_pair {
    T mat_entries[4];
    T vec_entries[2];
};

template <typename T>
struct recur2_binary_op {
    __host__ __device__ sqm2_pair<T> operator()(const sqm2_pair<T> &a,
                                                const sqm2_pair<T> &b) const {
        auto a_a = a.mat_entries[0];
        auto a_b = a.mat_entries[1];
        auto a_c = a.mat_entries[2];
        auto a_d = a.mat_entries[3];
        auto b_a = b.mat_entries[0];
        auto b_b = b.mat_entries[1];
        auto b_c = b.mat_entries[2];
        auto b_d = b.mat_entries[3];

        auto b_a_vec = b.vec_entries[0];
        auto b_b_vec = b.vec_entries[1];
        auto a_a_vec = a.vec_entries[0];
        auto a_b_vec = a.vec_entries[1];

        T new_mat_entries[4] = {b_a * a_a + b_b * a_c, b_a * a_b + b_b * a_d,
                                b_c * a_a + b_d * a_c, b_c * a_b + b_d * a_d};
        T new_vec_entries[2] = {b_a * a_a_vec + b_b * a_b_vec + b_a_vec,
                                b_c * a_a_vec + b_d * a_b_vec + b_b_vec};

        return sqm2_pair<T>{new_mat_entries, new_vec_entries};
    }
};

template <typename scalar_t>
void matrix_recurrence_second_order(const scalar_t *A, const scalar_t *x,
                                    scalar_t *out, int n_steps) {
    thrust::device_vector<sqm2_pair<scalar_t>> pairs(n_steps);

    thrust::counting_iterator<int> A_iter(0);
    // Initialize input_states and output_states
    thrust::transform(
        thrust::device, A_iter, A_iter + n_steps, pairs.begin(),
        [A, x] __host__ __device__(const int &i) {
            sqm2_pair<scalar_t> state;
            thrust::copy(A + i * 4, A + (i + 1) * 4, state.mat_entries);
            thrust::copy(x + i * 2, x + (i + 1) * 2, state.vec_entries);
            return state;
        });

    recur2_binary_op<scalar_t> binary_op;
    thrust::inclusive_scan(thrust::device, pairs.begin(), pairs.end(),
                           pairs.begin(), binary_op);
    thrust::for_each(thrust::device,
                     thrust::zip_iterator(A_iter, pairs.begin()),
                     thrust::zip_iterator(A_iter + n_steps, pairs.end()),
                     [out] __host__ __device__(
                         const thrust::tuple<int, sqm2_pair<scalar_t>> &t) {
                         int i = thrust::get<0>(t);
                         sqm2_pair<scalar_t> state = thrust::get<1>(t);
                         out[i * 2 + 0] = state.vec_entries[0];
                         out[i * 2 + 1] = state.vec_entries[1];
                     });
}
