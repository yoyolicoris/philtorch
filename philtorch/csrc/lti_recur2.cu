#include <ATen/Dispatch.h>
#include <torch/torch.h>

#include "recur2.cuh"

struct lti_recur2_dispatch
{
    template <typename Fn>
    void operator()(at::ScalarType type, Fn f) const
    {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
            at::kLong, at::kHalf, at::kBFloat16, type, "lti_recur2", [&] { f(scalar_t{}); });
    }
};

at::Tensor lti_mat_recur_second_order_cuda_impl(const at::Tensor &A,
                                                const at::Tensor &zi,
                                                const at::Tensor &x)
{
    TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be a 2D or 3D tensor");
    check_recur_inputs<2>(A, zi, x);
    TORCH_CHECK(A.dim() == 2 || A.size(0) == x.size(0), "A must have x's batch size");
    return recur_small_launch<2>(A, zi, x, A.dim() == 3 ? 4 : 0, 0,
                                 lti_recur2_dispatch{});
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("lti_recur2", &lti_mat_recur_second_order_cuda_impl);
}
