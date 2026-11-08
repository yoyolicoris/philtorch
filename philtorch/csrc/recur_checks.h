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
