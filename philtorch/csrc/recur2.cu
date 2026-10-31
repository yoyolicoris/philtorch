#include "recur_scan.cuh"

struct recur2_dispatch
{
    template <typename Fn>
    void operator()(torch::headeronly::ScalarType type, Fn f) const
    {
        using S = torch::headeronly::ScalarType;
        THO_DISPATCH_V2(
            type, "recur2", AT_WRAP([&] { f(scalar_t{}); }),
            AT_FLOATING_TYPES, AT_COMPLEX_TYPES, S::Long, S::Half, S::BFloat16);
    }
};

Tensor mat_recur_second_order_cuda_impl(const Tensor &A,
                                        const Tensor &zi,
                                        const Tensor &x)
{
    STD_TORCH_CHECK(A.dim() == 3 || A.dim() == 4, "A must be a 3D or 4D tensor");
    check_recur_inputs<2>(A, zi, x);
    STD_TORCH_CHECK(A.size(-3) == x.size(1), "A must have one matrix per step of x");
    STD_TORCH_CHECK(A.dim() == 3 || A.size(0) == x.size(0), "A must have x's batch size");
    return recur_small_launch<2>(A, zi, x, A.dim() == 4 ? x.size(1) * 4 : 0, 4,
                                 recur2_dispatch{});
}

Tensor lti_mat_recur_second_order_cuda_impl(const Tensor &A,
                                            const Tensor &zi,
                                            const Tensor &x)
{
    STD_TORCH_CHECK(A.dim() == 2 || A.dim() == 3, "A must be a 2D or 3D tensor");
    check_recur_inputs<2>(A, zi, x);
    STD_TORCH_CHECK(A.dim() == 2 || A.size(0) == x.size(0), "A must have x's batch size");
    return recur_small_launch<2>(A, zi, x, A.dim() == 3 ? 4 : 0, 0, recur2_dispatch{});
}

STABLE_TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("recur2", TORCH_BOX(&mat_recur_second_order_cuda_impl));
    m.impl("lti_recur2", TORCH_BOX(&lti_mat_recur_second_order_cuda_impl));
}
