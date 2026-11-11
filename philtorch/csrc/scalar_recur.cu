#include "recur_scan.cuh"

// h[t] = a h[t - 1] + x[b, t], h[-1] = zi[b]: the scan of M = 1, with a shared
// (strides 0, 0) or one per sequence (strides 1, 0).
Tensor lti_recur_cuda_impl(const Tensor &a,
                           const Tensor &zi, const Tensor &x)
{
    const bool per_item = lti_recur_per_item(a, zi, x);
    check_types_devices(a, zi, x);
    return recur_small_launch<1>(a, zi, x, per_item ? 1 : 0, 0, full_dispatch{});
}

// h[t] = decay[t] h[t - 1] + impulse[t]: the scalar recurrence with a
// (B, T) A, one coefficient per step.
Tensor scan_cuda(const Tensor &impulse, const Tensor &decay,
                 const Tensor &init)
{
    check_scan_inputs(impulse, decay, init);
    check_types_devices(decay, init, impulse);
    return recur_small_launch<1>(decay, init, impulse, impulse.size(1), 1, full_dispatch{});
}

STABLE_TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("lti_recur", TORCH_BOX(&lti_recur_cuda_impl));
    m.impl("scan", TORCH_BOX(&scan_cuda));
}
