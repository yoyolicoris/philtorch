#include <assert.h>
#include <stdio.h>

// The includer, philtorch/csrc/torchlpc_cuda.cu, includes stable_support.h
// first: it sets the stable ABI version before any torch header is read.
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>

using torch::stable::Tensor;

// atomicAdd for real types; a complex one adds its parts, which are laid out
// as two scalars.
template <typename scalar_t>
__device__ void atomic_add(scalar_t* address, scalar_t val) {
    atomicAdd(address, val);
}

template <typename scalar_t>
__device__ void atomic_add(c10::complex<scalar_t>* address,
                           c10::complex<scalar_t> val) {
    auto parts = reinterpret_cast<scalar_t*>(address);
    atomicAdd(parts, val.real());
    atomicAdd(parts + 1, val.imag());
}

// CUDA kernel for LPC computation, real or complex
template <typename scalar_t>
__global__ void lpc_cuda_kernel(scalar_t* y,           // [B, T]
                                const scalar_t* x,     // [B, T]
                                const scalar_t* zi,    // [B, order]
                                const scalar_t* A,     // [B, T, order]
                                int64_t B, int64_t T, int64_t order) {
    extern __shared__ char smem[];
    scalar_t* sm = reinterpret_cast<scalar_t*>(smem);

    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= order) return;

    // Initialize shared memory with the initial conditions, reversed
    sm[i] = zi[b * order + order - 1 - i];
    __syncthreads();

    int circular_idx = 0;
    for (int t = 0; t < T; ++t) {
        circular_idx = t % order;
        scalar_t a = -A[((b * T + t) * order) + i];

        // Compute s as in the Python code
        int idx_offset = circular_idx - i - 1;
        if (i > circular_idx - 1) {
            idx_offset += order;
        }
        scalar_t s = sm[(idx_offset + order) % order];

        scalar_t v = a * s;

        if (i == order - 1) {
            sm[circular_idx] = v;
            v = x[b * T + t];
        }
        __syncthreads();

        // Atomic add to shared memory
        atomic_add(&sm[circular_idx], v);
        __syncthreads();

        if (i == order - 1) {
            y[b * T + t] = sm[circular_idx];
        }
        __syncthreads();
    }
}

// The launch is outside the dispatch macro, inside which it did not compile.
template <typename scalar_t>
void lpc_launch(Tensor& y, const Tensor& x, const Tensor& zi, const Tensor& a,
                int64_t order, cudaStream_t stream) {
    lpc_cuda_kernel<scalar_t>
        <<<x.size(0), order, order * sizeof(scalar_t), stream>>>(
            y.mutable_data_ptr<scalar_t>(), x.const_data_ptr<scalar_t>(),
            zi.const_data_ptr<scalar_t>(), a.const_data_ptr<scalar_t>(),
            x.size(0), x.size(1), order);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor lpc_cuda_wrapper(const Tensor& x, const Tensor& a, const Tensor& zi) {
    namespace ph = philtorch;
    STD_TORCH_CHECK(
        ph::is_floating(x.scalar_type()) || ph::is_complex(x.scalar_type()),
        "Input must be floating point or complex");
    STD_TORCH_CHECK(a.scalar_type() == x.scalar_type(),
                    "Coefficients must have the same scalar type as input");
    STD_TORCH_CHECK(a.device() == x.device(),
                    "Coefficients must be on the same device as input");
    STD_TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                    "Initial conditions must have the same scalar type as input");

    STD_TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    STD_TORCH_CHECK(zi.dim() == 2, "Initial conditions must be 2D");
    STD_TORCH_CHECK(x.size(0) == zi.size(0),
                    "Batch size of input and initial conditions must match");

    const torch::stable::accelerator::DeviceGuard device_guard(
        x.get_device_index());

    auto a_contiguous = torch::stable::contiguous(a);
    auto x_contiguous = torch::stable::contiguous(x);
    auto zi_contiguous = torch::stable::contiguous(zi);

    auto order = a_contiguous.size(2);
    STD_TORCH_CHECK(order <= 1024, "LPC order must be less than or equal to 1024");
    STD_TORCH_CHECK(zi.size(1) == order,
                    "Initial conditions must have one state per coefficient of a");
    auto stream = ph::current_cuda_stream(x.get_device_index());

    auto out = torch::stable::empty_like(x_contiguous);
    THO_DISPATCH_V2(
        x.scalar_type(), "lpc_cuda",
        AT_WRAP([&] {
            lpc_launch<scalar_t>(out, x_contiguous, zi_contiguous, a_contiguous,
                                 order, stream);
        }),
        AT_FLOATING_TYPES, AT_COMPLEX_TYPES);
    return out;
}

STABLE_TORCH_LIBRARY_IMPL(torchlpc, CUDA, m) {
    m.impl("lpc", TORCH_BOX(&lpc_cuda_wrapper));
}
