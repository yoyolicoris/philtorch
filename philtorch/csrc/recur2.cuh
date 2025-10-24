#include <thrust/tuple.h>

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

template <typename T>
struct output_unary_op
{
    __host__ __device__ cuda::std::tuple<T, T> operator()(
        const sqm2_pair<T> &state) const
    {
        return thrust::make_tuple(thrust::get<4>(state), thrust::get<5>(state));
    }
};

template <typename T>
struct share_A_input_op
{
    const T *A;
    int n_steps;
    __host__ __device__ sqm2_pair<T> operator()(int i, const T &x_0,
                                                const T &x_1) const
    {
        int idx = i % n_steps;
        return thrust::make_tuple(A[idx], A[idx + n_steps],
                                  A[idx + n_steps * 2], A[idx + n_steps * 3],
                                  x_0, x_1);
    }
};

template <typename T>
struct lti_batch_A_input_op
{
    const T *A;
    int n_steps;
    __host__ __device__ sqm2_pair<T> operator()(int i, const T &x_0,
                                                const T &x_1) const
    {
        int idx = i / n_steps * 4;
        int offset = i % n_steps;
        if (offset > 0)
            return thrust::make_tuple(A[idx], A[idx + 1], A[idx + 2], A[idx + 3], x_0, x_1);
        return thrust::make_tuple(0, 0, 0, 0, x_0, x_1);
    }
};

template <typename T>
struct lti_share_A_input_op
{
    const T *A;
    int n_steps;
    __host__ __device__ sqm2_pair<T> operator()(int i, const T &x_0,
                                                const T &x_1) const
    {
        int offset = i % n_steps;
        if (offset > 0)
            return thrust::make_tuple(A[0], A[1], A[2], A[3], x_0, x_1);
        return thrust::make_tuple(0, 0, 0, 0, x_0, x_1);
    }
};

struct index2key
{
    int n_steps;
    __host__ __device__ int operator()(int i) const { return i / n_steps; }
};