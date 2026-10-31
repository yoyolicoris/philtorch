#include <ATen/Dispatch_v2.h>

#include "recur_scan.cuh"

// float64 and int64 run faster in the Triton kernels on GPUs measured, and
// complex128's 24 values overflow CUB's default shared memory.
struct recur3_dispatch
{
    template <typename Fn>
    void operator()(at::ScalarType type, Fn f) const
    {
        AT_DISPATCH_V2(
            type, "recur3", AT_WRAP([&] { f(scalar_t{}); }),
            at::kFloat, at::kHalf, at::kBFloat16, at::kComplexFloat);
    }
};

at::Tensor mat_recur_third_order_cuda_impl(const at::Tensor &A, const at::Tensor &zi,
                                           const at::Tensor &x)
{
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    check_recur_inputs<3>(A, zi, x);
    TORCH_CHECK(A.size(-3) == x.size(1), "A must have one matrix per step of x");
    TORCH_CHECK(A.dim() == 3 || A.size(0) == x.size(0), "A must have x's batch size");
    return recur_small_launch<3>(A, zi, x, A.dim() == 4 ? x.size(1) * 9 : 0, 9,
                                 recur3_dispatch{});
}

at::Tensor lti_mat_recur_third_order_cuda_impl(const at::Tensor &A, const at::Tensor &zi,
                                               const at::Tensor &x)
{
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be a 2D or 3D tensor");
    check_recur_inputs<3>(A, zi, x);
    TORCH_CHECK(A.dim() == 2 || A.size(0) == x.size(0), "A must have x's batch size");
    return recur_small_launch<3>(A, zi, x, A.dim() == 3 ? 9 : 0, 0, recur3_dispatch{});
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("recur3", &mat_recur_third_order_cuda_impl);
    m.impl("lti_recur3", &lti_mat_recur_third_order_cuda_impl);
}
