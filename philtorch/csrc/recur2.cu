#include <ATen/Dispatch.h>
#include <torch/torch.h>

#include "recur_scan.cuh"

struct recur2_dispatch
{
    template <typename Fn>
    void operator()(at::ScalarType type, Fn f) const
    {
        AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
            at::kLong, at::kHalf, at::kBFloat16, type, "recur2", [&] { f(scalar_t{}); });
    }
};

at::Tensor mat_recur_second_order_cuda_impl(const at::Tensor &A,
                                            const at::Tensor &zi,
                                            const at::Tensor &x)
{
    TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    check_recur_inputs<2>(A, zi, x);
    TORCH_CHECK(A.size(-3) == x.size(1), "A must have one matrix per step of x");
    TORCH_CHECK(A.dim() == 3 || A.size(0) == x.size(0), "A must have x's batch size");
    return recur_small_launch<2>(A, zi, x, A.dim() == 4 ? x.size(1) * 4 : 0, 4,
                                 recur2_dispatch{});
}

TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("recur2", &mat_recur_second_order_cuda_impl);
}
