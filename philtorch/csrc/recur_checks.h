#pragma once

#include "stable_support.h"

// The dtype and device checks of the recurrence ops (CPU and CUDA): A, zi and
// x of one dtype, on x's device.
inline void check_types_devices(const torch::stable::Tensor &A,
                                const torch::stable::Tensor &zi,
                                const torch::stable::Tensor &x)
{
    STD_TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                    "zi must have the same scalar type as input");
    STD_TORCH_CHECK(A.scalar_type() == x.scalar_type(),
                    "A must have the same scalar type as input");
    STD_TORCH_CHECK(A.device() == x.device() && zi.device() == x.device(),
                    "A and zi must be on the same device as input");
}

// The shape checks of lti_recur: a is a scalar or (B,), x (B, T) with T > 0,
// zi (B,). Returns whether a is one per sequence.
inline bool lti_recur_per_item(const torch::stable::Tensor &a,
                               const torch::stable::Tensor &zi,
                               const torch::stable::Tensor &x)
{
    STD_TORCH_CHECK(a.dim() <= 1, "A must be a vector or a scalar");
    STD_TORCH_CHECK(x.dim() == 2 && zi.dim() == 1 && zi.size(0) == x.size(0),
                    "x must be (B, T) and zi (B,)");
    STD_TORCH_CHECK(x.size(1) > 0, "x must contain at least one time step");
    const bool per_item = a.dim() == 1 && a.numel() == x.size(0);
    STD_TORCH_CHECK(per_item || a.numel() == 1, "A must be a scalar or (B,)");
    return per_item;
}

// The checks of scan: impulse and decay (B, T) and init (B,), in torchlpc's
// dtypes, which the recurrence kernels it runs exceed.
inline void check_scan_inputs(const torch::stable::Tensor &impulse,
                              const torch::stable::Tensor &decay,
                              const torch::stable::Tensor &init)
{
    STD_TORCH_CHECK(impulse.dim() == 2 && decay.dim() == 2 &&
                        decay.size(0) == impulse.size(0) &&
                        decay.size(1) == impulse.size(1) &&
                        init.dim() == 1 && init.size(0) == impulse.size(0),
                    "impulse and decay must be (B, T) and init (B,)");
    using S = torch::headeronly::ScalarType;
    const auto type = impulse.scalar_type();
    STD_TORCH_CHECK(type == S::Float || type == S::Double ||
                        type == S::ComplexFloat || type == S::ComplexDouble,
                    "Input must be float32, float64, complex64 or complex128");
}

// The order check of the ops of one order M (recur2, recur3 and their lti_
// forms), ahead of check_recurN_shapes.
inline void check_order(const torch::stable::Tensor &x, int64_t M)
{
    STD_TORCH_CHECK(x.dim() == 3 && x.size(2) == M, "x must be of shape (B, T, ", M, ")");
}

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
