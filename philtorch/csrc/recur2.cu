#include "recur_scan.cuh"

STABLE_TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("recur2", TORCH_BOX((&recur_small<2, true, full_dispatch>)));
    m.impl("lti_recur2", TORCH_BOX((&recur_small<2, false, full_dispatch>)));
}
