#include "recur_scan.cuh"

// float64 and int64 are left to recurN, and complex128's 24 values overflow
// CUB's default shared memory.
struct recur3_dispatch
{
    template <typename Fn>
    void operator()(torch::headeronly::ScalarType type, Fn f) const
    {
        using S = torch::headeronly::ScalarType;
        THO_DISPATCH_V2(
            type, "recur3", AT_WRAP([&] { f(scalar_t{}); }),
            S::Float, S::Half, S::BFloat16, S::ComplexFloat);
    }
};

STABLE_TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("recur3", TORCH_BOX((&recur_small<3, true, recur3_dispatch>)));
    m.impl("lti_recur3", TORCH_BOX((&recur_small<3, false, recur3_dispatch>)));
}
