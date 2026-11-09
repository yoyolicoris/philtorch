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

STABLE_TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("recur2", TORCH_BOX((&recur_small<2, true, recur2_dispatch>)));
    m.impl("lti_recur2", TORCH_BOX((&recur_small<2, false, recur2_dispatch>)));
}
