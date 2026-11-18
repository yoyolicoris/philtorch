// The includer, philtorch/csrc/torchlpc_cuda.cu, includes stable_support.h
// first, which holds the torch headers this file uses, and recur_checks.h.

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

// CUDA kernel for LPC computation, real or complex. The history in shared
// memory and the sums are in the accumulation type, as the recurrence kernels'
// are; atomicAdd has no overload for c10's Half or BFloat16 either.
template <typename scalar_t>
__global__ void lpc_cuda_kernel(scalar_t* y,           // [B, T]
                                const scalar_t* x,     // [B, T]
                                const scalar_t* zi,    // [B, order]
                                const scalar_t* A,     // [B, T, order]
                                int64_t B, int64_t T, int64_t order) {
    using acc_t = philtorch::opmath_type<scalar_t>;
    extern __shared__ char smem[];
    acc_t* sm = reinterpret_cast<acc_t*>(smem);

    int b = blockIdx.x;
    int i = threadIdx.x;

    if (b >= B || i >= order) return;

    // Initialize shared memory with the initial conditions, reversed
    sm[i] = static_cast<acc_t>(zi[b * order + order - 1 - i]);
    __syncthreads();

    int circular_idx = 0;
    for (int t = 0; t < T; ++t) {
        circular_idx = t % order;
        acc_t a = -static_cast<acc_t>(A[((b * T + t) * order) + i]);

        // Compute s as in the Python code
        int idx_offset = circular_idx - i - 1;
        if (i > circular_idx - 1) {
            idx_offset += order;
        }
        acc_t s = sm[(idx_offset + order) % order];

        acc_t v = a * s;

        if (i == order - 1) {
            sm[circular_idx] = v;
            v = static_cast<acc_t>(x[b * T + t]);
        }
        __syncthreads();

        // Atomic add to shared memory
        atomic_add(&sm[circular_idx], v);
        __syncthreads();

        if (i == order - 1) {
            y[b * T + t] = static_cast<scalar_t>(sm[circular_idx]);
        }
        __syncthreads();
    }
}

// The launch is outside the dispatch macro, inside which it did not compile.
template <typename scalar_t>
void lpc_launch(Tensor& y, const Tensor& x, const Tensor& zi, const Tensor& a,
                int64_t order, cudaStream_t stream) {
    using acc_t = philtorch::opmath_type<scalar_t>;
    lpc_cuda_kernel<scalar_t>
        <<<x.size(0), order, order * sizeof(acc_t), stream>>>(
            y.mutable_data_ptr<scalar_t>(), x.const_data_ptr<scalar_t>(),
            zi.const_data_ptr<scalar_t>(), a.const_data_ptr<scalar_t>(),
            x.size(0), x.size(1), order);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor lpc_cuda_wrapper(const Tensor& x, const Tensor& a, const Tensor& zi) {
    namespace ph = philtorch;
    check_lpc_inputs(x, a, zi);

    const torch::stable::accelerator::DeviceGuard device_guard(
        x.get_device_index());

    auto a_contiguous = torch::stable::contiguous(a);
    auto x_contiguous = torch::stable::contiguous(x);
    auto zi_contiguous = torch::stable::contiguous(zi);

    auto order = a_contiguous.size(2);
    check_lpc_order(order);
    auto stream = ph::current_cuda_stream(x.get_device_index());

    auto out = torch::stable::empty_like(x_contiguous);
    using S = torch::headeronly::ScalarType;
    THO_DISPATCH_V2(
        x.scalar_type(), "lpc_cuda",
        AT_WRAP([&] {
            lpc_launch<scalar_t>(out, x_contiguous, zi_contiguous, a_contiguous,
                                 order, stream);
        }),
        AT_FLOATING_TYPES, AT_COMPLEX_TYPES, S::Half, S::BFloat16,
        S::ComplexHalf);
    return out;
}

STABLE_TORCH_LIBRARY_IMPL(torchlpc, CUDA, m) {
    m.impl("lpc", TORCH_BOX(&lpc_cuda_wrapper));
}
