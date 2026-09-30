#pragma once

#include <type_traits>

template <typename scalar_t>
inline scalar_t host_dot(const scalar_t *left, const scalar_t *right, int size)
{
    constexpr int minimum_simd_size = 8;
    scalar_t sum = 0;
    if constexpr (std::is_same_v<scalar_t, float> ||
                  std::is_same_v<scalar_t, double>)
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