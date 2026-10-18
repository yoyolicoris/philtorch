#pragma once

#include <type_traits>

template <typename scalar_t, typename acc_t>
inline acc_t host_dot(const scalar_t *left, const acc_t *right, int size)
{
    constexpr int minimum_simd_size = 8;
    acc_t sum = 0;
    if constexpr (std::is_same_v<acc_t, float> ||
                  std::is_same_v<acc_t, double>)
    {
        if (size >= minimum_simd_size)
        {
#pragma omp simd reduction(+ : sum)
            for (int index = 0; index < size; ++index)
                sum += left[index] * right[index];
            return sum;
        }
    }
    for (int index = 0; index < size; ++index)
        sum += left[index] * right[index];
    return sum;
}