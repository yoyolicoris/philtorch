// CUDA implementation of recurrence relation x[t]=x_{t+1}
// for x_{t+1} = M_t * x_t + v_t

#include <torch/extension.h>

__global__ void assoc_recur_mv_cuda(float* Ms, float* vs, float* x, int T, int B, int D) {
    int t = blockIdx.x;
    int b = blockIdx.y;

    // Check if within bounds
    if (t < T && b < B) {
        for (int d = 0; d < D; d++) {
            float sum = 0;
            for (int d2 = 0; d2 < D; d2++) {
                sum += Ms[t * B * D * D + b * D * D + d * D + d2] * x[(t * B + b) * D + d2];
            }
            x[(t * B + b) * D + d] = sum + vs[t * B + b];
        }
    }
}

TORCH_LIBRARY(philtorch, m) {
    m.def("assoc_recur_mv", &assoc_recur_mv_cuda);
}