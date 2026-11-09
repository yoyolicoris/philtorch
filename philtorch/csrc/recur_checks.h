#pragma once

#include "stable_support.h"

// The shape checks of recurN and lti_recurN (the CPU and CUDA kernels): A is
// (M, M) or (B, M, M), or when varying (T, M, M) or (B, T, M, M).
inline void check_recurN_shapes(const torch::stable::Tensor &A,
                                const torch::stable::Tensor &zi,
                                const torch::stable::Tensor &x, bool varying)
{
    STD_TORCH_CHECK(varying ? (A.dim() == 3 || A.dim() == 4)
                            : (A.dim() == 2 || A.dim() == 3),
                    varying ? "A must be a 3D or 4D tensor"
                            : "A must be a 2D or 3D tensor");
    STD_TORCH_CHECK(x.dim() == 3 && zi.dim() == 2 && zi.size(0) == x.size(0) &&
                        zi.size(1) == A.size(-1),
                    "x must be (B, T, M) and zi (B, M)");
    STD_TORCH_CHECK(x.size(2) == A.size(-1),
                    "Last dimension of x must match last dimension of A");
    STD_TORCH_CHECK(A.size(-1) == A.size(-2),
                    "Last two dimensions of A must be equal");
    if (varying)
        STD_TORCH_CHECK(A.size(-3) == x.size(1),
                        "A must have one matrix per time step of x");
    STD_TORCH_CHECK(A.dim() == (varying ? 3 : 2) || A.size(0) == x.size(0),
                    "A must have the batch size of x");
}

// The checks of lpc (the CPU and CUDA kernels): x is (B, T), a (B, T, M) and
// zi (B, M), all of one dtype. The dispatch rejects dtypes it doesn't take.
inline void check_lpc_inputs(const torch::stable::Tensor &x,
                             const torch::stable::Tensor &a,
                             const torch::stable::Tensor &zi)
{
    STD_TORCH_CHECK(a.scalar_type() == x.scalar_type(),
                    "Coefficients must have the same scalar type as input");
    STD_TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                    "Initial conditions must have the same scalar type as input");
    STD_TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    STD_TORCH_CHECK(zi.dim() == 2, "Initial conditions must be 2D");
    STD_TORCH_CHECK(x.size(0) == zi.size(0),
                    "Batch size of input and initial conditions must match");
    STD_TORCH_CHECK(a.dim() == 3, "a must be 3-dimensional");
    STD_TORCH_CHECK(a.size(0) == x.size(0) && a.size(1) == x.size(1),
                    "Batch and time dimensions of a and x must match");
    STD_TORCH_CHECK(zi.size(1) == a.size(2),
                    "Initial conditions must have one state per coefficient of a");
}
