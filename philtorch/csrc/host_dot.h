#pragma once

#include "stable_support.h"

#include <type_traits>

// sum_i left[i] right[i * stride], in right's opmath type. Only float and
// double from size 8 on are vectorized: in benchmarks, vectorizing smaller
// sizes or complex types made the recurrence and lpc kernels slower.
template <int stride = 1, typename left_t, typename right_t>
inline philtorch::opmath_type<right_t> host_dot(const left_t *left,
                                         const right_t *right, int size)
{
    using acc_t = philtorch::opmath_type<right_t>;
    constexpr int minimum_simd_size = 8;
    acc_t sum = 0;
    if constexpr (std::is_same_v<acc_t, float> ||
                  std::is_same_v<acc_t, double>)
    {
        if (size >= minimum_simd_size)
        {
#pragma omp simd reduction(+ : sum)
            for (int index = 0; index < size; ++index)
                sum += static_cast<acc_t>(left[index]) *
                       static_cast<acc_t>(right[index * stride]);
            return sum;
        }
    }
    for (int index = 0; index < size; ++index)
        sum += static_cast<acc_t>(left[index]) *
               static_cast<acc_t>(right[index * stride]);
    return sum;
}
