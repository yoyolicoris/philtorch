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

// CUDA kernel for LPC computation
template <typename scalar_t>
__global__ void lpc_cuda_kernel(scalar_t* padded_y,  // [B, T + order]
                                const scalar_t* A,   // [B, T, order]
                                int64_t B, int64_t T, int64_t order) {
    extern __shared__ char smem[];
    scalar_t* sm = reinterpret_cast<scalar_t*>(smem);

    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= order) return;

    // Initialize shared memory with the first 'order' elements
    sm[i] = padded_y[b * (T + order) + i];
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
            v = padded_y[b * (T + order) + t + order];
        }
        __syncthreads();

        // Atomic add to shared memory
        atomicAdd(&sm[circular_idx], v);
        __syncthreads();

        if (i == order - 1) {
            padded_y[b * (T + order) + t + order] = sm[circular_idx];
        }
        __syncthreads();
    }
}
// CUDA kernel for complex LPC computation
template <typename scalar_t>
__global__ void lpc_cuda_kernel_complex(
    scalar_t* padded_y_real,  // [B, T + order]
    scalar_t* padded_y_imag,  // [B, T + order]
    const scalar_t* A_real,   // [B, T, order]
    const scalar_t* A_imag,   // [B, T, order]
    int64_t B, int64_t T, int64_t order) {
    extern __shared__ char smem[];
    scalar_t* sm_real = reinterpret_cast<scalar_t*>(smem);
    scalar_t* sm_imag = sm_real + order;

    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= order) return;

    // Initialize shared memory with the first 'order' elements
    sm_real[i] = padded_y_real[b * (T + order) + i];
    sm_imag[i] = padded_y_imag[b * (T + order) + i];
    __syncthreads();

    int circular_idx = 0;
    for (int t = 0; t < T; ++t) {
        circular_idx = t % order;
        scalar_t a_real = -A_real[((b * T + t) * order) + i];
        scalar_t a_imag = -A_imag[((b * T + t) * order) + i];

        int idx_offset = circular_idx - i - 1;
        if (i > circular_idx - 1) {
            idx_offset += order;
        }
        int s_idx = (idx_offset + order) % order;
        scalar_t s_real = sm_real[s_idx];
        scalar_t s_imag = sm_imag[s_idx];

        // Complex multiply: v = a * s
        scalar_t v_real = a_real * s_real - a_imag * s_imag;
        scalar_t v_imag = a_real * s_imag + a_imag * s_real;

        if (i == order - 1) {
            sm_real[circular_idx] = v_real;
            sm_imag[circular_idx] = v_imag;
            v_real = padded_y_real[b * (T + order) + t + order];
            v_imag = padded_y_imag[b * (T + order) + t + order];
        }
        __syncthreads();

        atomicAdd(&sm_real[circular_idx], v_real);
        atomicAdd(&sm_imag[circular_idx], v_imag);
        __syncthreads();

        if (i == order - 1) {
            padded_y_real[b * (T + order) + t + order] = sm_real[circular_idx];
            padded_y_imag[b * (T + order) + t + order] = sm_imag[circular_idx];
        }
        __syncthreads();
    }
}

// The launches are outside the dispatch macro, inside which they did not compile.
template <typename scalar_t>
void lpc_launch(Tensor& padded_y, const Tensor& a, int64_t B, int64_t T,
                int64_t order, cudaStream_t stream) {
    lpc_cuda_kernel<scalar_t><<<B, order, order * sizeof(scalar_t), stream>>>(
        padded_y.mutable_data_ptr<scalar_t>(), a.const_data_ptr<scalar_t>(), B,
        T, order);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void lpc_launch_complex(Tensor& padded_y_real, Tensor& padded_y_imag,
                        const Tensor& a_real, const Tensor& a_imag, int64_t B,
                        int64_t T, int64_t order, cudaStream_t stream) {
    lpc_cuda_kernel_complex<scalar_t>
        <<<B, order, 2 * order * sizeof(scalar_t), stream>>>(
            padded_y_real.mutable_data_ptr<scalar_t>(),
            padded_y_imag.mutable_data_ptr<scalar_t>(),
            a_real.const_data_ptr<scalar_t>(),
            a_imag.const_data_ptr<scalar_t>(), B, T, order);
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

    auto order = a_contiguous.size(2);
    STD_TORCH_CHECK(order <= 1024, "LPC order must be less than or equal to 1024");
    auto stream = ph::current_cuda_stream(x.get_device_index());
    // cat, flip, real, imag, stack and view_as_complex have no stable wrapper.
    const std::vector<int64_t> flip_dims{1};
    auto flip = [&](const Tensor& t) {
        return ph::call_aten("aten::flip", "", t, flip_dims);
    };
    auto cat = [&](const Tensor& first, const Tensor& second) {
        return ph::call_aten("aten::cat", "",
                             std::vector<Tensor>{first, second}, int64_t(1));
    };

    if (ph::is_floating(x.scalar_type())) {
        Tensor out = torch::stable::contiguous(cat(flip(zi), x));
        THO_DISPATCH_V2(
            x.scalar_type(), "lpc_cuda",
            AT_WRAP([&] {
                lpc_launch<scalar_t>(out, a_contiguous, x.size(0), x.size(1),
                                     order, stream);
            }),
            AT_FLOATING_TYPES);
        return torch::stable::contiguous(
            torch::stable::narrow(out, 1, order, out.size(1) - order));
    }
    auto real = [&](const Tensor& t) { return ph::call_aten("aten::real", "", t); };
    auto imag = [&](const Tensor& t) { return ph::call_aten("aten::imag", "", t); };
    auto out_real = torch::stable::contiguous(cat(flip(real(zi)), real(x)));
    auto out_imag = torch::stable::contiguous(cat(flip(imag(zi)), imag(x)));
    auto a_real = torch::stable::contiguous(real(a_contiguous));
    auto a_imag = torch::stable::contiguous(imag(a_contiguous));
    THO_DISPATCH_V2(
        out_real.scalar_type(), "lpc_cuda_complex",
        AT_WRAP([&] {
            lpc_launch_complex<scalar_t>(out_real, out_imag, a_real, a_imag,
                                         x.size(0), x.size(1), order, stream);
        }),
        AT_FLOATING_TYPES);
    auto out = ph::call_aten(
        "aten::view_as_complex", "",
        ph::call_aten("aten::stack", "",
                      std::vector<Tensor>{out_real, out_imag}, int64_t(-1)));
    return torch::stable::contiguous(
        torch::stable::narrow(out, 1, order, out.size(1) - order));
}

STABLE_TORCH_LIBRARY_IMPL(torchlpc, CUDA, m) {
    m.impl("lpc", TORCH_BOX(&lpc_cuda_wrapper));
}
