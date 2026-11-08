#include "recurN_scan.cuh"

#include "recur_checks.h"

#include <cstdint>
#include <limits>

using torch::stable::Tensor;
namespace rn = philtorch::recur_n;

// philtorch::recurN and philtorch::lti_recurN on CUDA, for any state size M up
// to 128 (64 for complex): see recurN_scan.cuh. No host-to-device copy or
// synchronization, so calls can be captured in CUDA graphs.

namespace
{
    // The largest state size of S, a complex state counting twice.
    template <class S>
    constexpr int max_states = rn::is_cx<rn::acc_t<S>> ? rn::MAX_STATES / 2 : rn::MAX_STATES;

    // log2 of the steps per chunk, a power of 2 (the power's squarings: its
    // log2), a complex state counting twice: longer chunks mean fewer
    // look-backs, shorter ones more blocks.
    int chunk_log2(int BM, bool complex, bool varying)
    {
        const int states = BM * (complex ? 2 : 1);
        if (varying)
            return states <= 16 ? 7 : 6;
        return states <= 16 ? 8 : 7;
    }

    // f(std::integral_constant<int, BM>) for the BM of the padded size bm.
    template <class S, int BM = 2, class F>
    void with_bm(int bm, F &&f)
    {
        if (bm == BM)
            f(std::integral_constant<int, BM>{});
        else if constexpr (BM < max_states<S>)
            with_bm<S, BM * 2>(bm, f);
    }

    // The matrices' A^chunk in P, for the time-invariant scan: A (Ba, M, M)
    // squared log2(chunk) times, in one launch when a matrix is one tile, else
    // one launch per squaring, ping-ponging between P and tmp so that the last
    // lands in P.
    template <class S, int BM>
    void chunk_power(const S *A, int64_t Ba, rn::acc_t<S> *P, rn::acc_t<S> *tmp, int M,
                     int squarings, cudaStream_t stream)
    {
        using PC = rn::PowerCfg<BM>;
        const dim3 grid(static_cast<unsigned>(Ba), PC::TILES * PC::TILES);
        if constexpr (PC::TILES == 1)
        {
            rn::power_kernel<S, BM><<<grid, PC::NT, 0, stream>>>(A, nullptr, P, tmp, M, squarings);
            STD_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else
        {
            const rn::acc_t<S> *src = nullptr;
            for (int q = 0; q < squarings; ++q)
            {
                rn::acc_t<S> *dst = (squarings - 1 - q) % 2 == 0 ? P : tmp;
                rn::power_kernel<S, BM><<<grid, PC::NT, 0, stream>>>(q == 0 ? A : nullptr, src, dst,
                                                                      nullptr, M, 1);
                STD_CUDA_KERNEL_LAUNCH_CHECK();
                src = dst;
            }
        }
    }

    // The inputs are contiguous and checked. A is (Ba, T, M, M), Ba = 1 or B,
    // when VARYING, else (Ba, M, M).
    template <class S, bool VARYING>
    void run(const Tensor &A, const Tensor &zi, const Tensor &x, Tensor &out)
    {
        using Acc = rn::acc_t<S>;
        constexpr bool complex = rn::is_cx<Acc>;
        const int64_t B = x.size(0), T = x.size(1), M = x.size(2);
        STD_TORCH_CHECK(M <= max_states<S>,
                        "the CUDA recurrence takes M up to ", rn::MAX_STATES, " (", rn::MAX_STATES / 2,
                        " complex), got ", M);
        STD_TORCH_CHECK(T <= std::numeric_limits<int>::max(),
                        "the input is too large for the CUDA recurrence");
        int BM = 2;
        while (BM < M)
            BM *= 2;
        const int squarings = chunk_log2(BM, complex, VARYING);
        const int chunk = 1 << squarings;
        const int64_t C = philtorch::ceil_div<int64_t>(T, chunk);
        STD_TORCH_CHECK(B * C < std::numeric_limits<int>::max(),
                        "the input is too large for the CUDA recurrence");
        const int64_t Ba = A.dim() == (VARYING ? 4 : 3) ? A.size(0) : 1;
        const int64_t stride_ab = Ba > 1 ? (VARYING ? T : 1) * M * M : 0;
        const int64_t stride_pb = Ba > 1 ? M * M : 0;

        // The scratch: each chunk's end state, offset and map; A^chunk and a
        // buffer for the squarings; the flags and the ticket counter.
        size_t bytes = 0;
        auto reserve = [&](int64_t n_bytes)
        {
            const size_t offset = bytes;
            bytes += philtorch::ceil_div<size_t>(n_bytes, 256) * 256;
            return offset;
        };
        const int64_t BC = B * C;
        const bool scan = C > 1;
        const size_t o_states = reserve(BC * M * sizeof(Acc));
        const size_t o_offsets = reserve(BC * M * sizeof(Acc));
        const size_t o_maps = reserve(VARYING ? BC * M * M * sizeof(Acc) : 0);
        const size_t o_power = reserve(VARYING ? 0 : Ba * M * M * sizeof(Acc));
        const size_t o_tmp = reserve(VARYING ? 0 : Ba * M * M * sizeof(Acc));
        const size_t o_flags = reserve((BC + 1) * sizeof(int));
        auto work = torch::stable::empty({static_cast<int64_t>(bytes)}, torch::headeronly::ScalarType::Byte,
                                         std::nullopt, x.device());
        char *base = static_cast<char *>(work.mutable_data_ptr());
        auto at = [&](size_t offset)
        { return reinterpret_cast<Acc *>(base + offset); };
        int *flags = reinterpret_cast<int *>(base + o_flags);

        const auto stream = philtorch::current_cuda_stream(x.get_device_index());
        STD_CUDA_CHECK(cudaMemsetAsync(flags, 0, (BC + 1) * sizeof(int), stream));
        const S *A_ptr = A.const_data_ptr<S>();
        with_bm<S>(BM, [&](auto bm)
                   {
            constexpr int BM_ = decltype(bm)::value;
            if constexpr (!VARYING)
                if (scan)
                    chunk_power<S, BM_>(A_ptr, Ba, at(o_power), at(o_tmp), static_cast<int>(M), squarings, stream);
            rn::scan_kernel<S, BM_, VARYING>
                <<<static_cast<unsigned>(BC), rn::Cfg<Acc, BM_, VARYING>::NT, 0, stream>>>(
                    A_ptr, x.const_data_ptr<S>(), zi.const_data_ptr<S>(), at(o_power),
                    out.mutable_data_ptr<S>(), at(o_offsets), at(o_maps), at(o_states), flags,
                    static_cast<int>(T), static_cast<int>(C), chunk, stride_ab, stride_pb,
                    static_cast<int>(M), reinterpret_cast<uintptr_t>(A_ptr) % 16 == 0);
            STD_CUDA_KERNEL_LAUNCH_CHECK(); });
    }

    template <bool VARYING>
    Tensor recur_cuda(const Tensor &A, const Tensor &zi, const Tensor &x)
    {
        STD_TORCH_CHECK(zi.scalar_type() == x.scalar_type(),
                        "zi must have the same scalar type as input");
        STD_TORCH_CHECK(A.scalar_type() == x.scalar_type(),
                        "A must have the same scalar type as input");
        STD_TORCH_CHECK(A.device() == x.device() && zi.device() == x.device(),
                        "A and zi must be on the same device as input");
        check_recurN_shapes(A, zi, x, VARYING);
        const auto A_contiguous = torch::stable::contiguous(A);
        const auto zi_contiguous = torch::stable::contiguous(zi);
        const auto x_contiguous = torch::stable::contiguous(x);
        auto out = torch::stable::empty_like(x_contiguous);
        if (x_contiguous.numel() == 0)
            return out;
        const torch::stable::accelerator::DeviceGuard device_guard(x.get_device_index());
        // A macro argument takes no comma: run's template arguments are in go.
        auto go = [&](auto tag)
        { run<decltype(tag), VARYING>(A_contiguous, zi_contiguous, x_contiguous, out); };
        using S = torch::headeronly::ScalarType;
        THO_DISPATCH_V2(
            x.scalar_type(), "recurN", AT_WRAP([&] { go(scalar_t{}); }),
            AT_FLOATING_TYPES, AT_COMPLEX_TYPES, S::Long, S::Half, S::BFloat16);
        return out;
    }
} // namespace

STABLE_TORCH_LIBRARY_IMPL(philtorch, CUDA, m)
{
    m.impl("recurN", TORCH_BOX(&recur_cuda<true>));
    m.impl("lti_recurN", TORCH_BOX(&recur_cuda<false>));
}
