// The Metal kernels of lpc, lpc_mps.mm's: setup.py embeds this file in the
// extension, which compiles it after mps_common.metal on first use.
//
// y[b, t] = x[b, t] - sum_i a[b, t, i] y[b, t - 1 - i], y[b, -1 - k] =
// zi[b, k], serial over t, as lpc is on every device: one threadgroup per
// sequence, one thread per coefficient i < M. The last M outputs live in a
// ring buffer in threadgroup memory, slot j holding y[s] for s mod M == j.
// Each step, thread i multiplies a[b, t, i] by y[t - 1 - i], the products are
// summed over the group (simd_sum, then the SIMD groups' partials through
// threadgroup memory), and thread 0 computes y[t], which it keeps in a
// register for the next step.

// The most coefficients, as on CUDA: one thread each.
constant constexpr uint MAX_ORDER = 1024;
constant constexpr uint MAX_SIMD_GROUPS = MAX_ORDER / SIMD_WIDTH;

template <typename T, typename A = typename Acc<T>::type>
inline void lpc_sequence(const device T *x, const device T *a,
                         const device T *zi, device T *out, uint steps,
                         uint order, uint lid, uint lane, uint simd_group,
                         threadgroup A *ring, threadgroup A *partials)
{
    const uint simd_groups = (order + SIMD_WIDTH - 1) / SIMD_WIDTH;
    if (lid < order)
        ring[order - 1 - lid] = A(zi[lid]);
    // y[t - 1], which thread 0 reads from its register, not the ring.
    A previous = A(0.0f);
    if (lid == 0)
        previous = A(zi[0]);
    // The slot of y[t], and of y[t - 1 - lid] at step t, one less, wrapped.
    uint slot = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < steps; ++t)
    {
        A product = A(0.0f);
        if (lid < order)
        {
            const A coefficient = A(*a);
            if (lid == 0)
                product = mul(coefficient, previous);
            else
            {
                const uint back = lid + 1;
                product = mul(coefficient, ring[slot >= back ? slot - back
                                                             : slot + order - back]);
            }
        }
        a += order;
        const A partial = simd_sum(product);
        // One buffer per step parity: a SIMD group may write step t + 1's
        // partial before another thread has read step t's.
        threadgroup A *sums = partials + (t & 1) * MAX_SIMD_GROUPS;
        if (lane == 0)
            sums[simd_group] = partial;
        // The reads of y[t - M]'s slot, by thread M - 1, precede this barrier,
        // the write of y[t] to it follows; and the write of y[t - 1 - i] to
        // the ring is visible to thread i by this barrier of step t.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0)
        {
            A total = A(0.0f);
            for (uint g = 0; g < simd_groups; ++g)
                total += sums[g];
            previous = A(x[t]) - total;
            out[t] = T(previous);
            ring[slot] = previous;
        }
        slot = slot + 1 == order ? 0 : slot + 1;
    }
}

#define DEFINE_LPC(T, NAME)                                                   \
    [[max_total_threads_per_threadgroup(MAX_ORDER)]] kernel void              \
    lpc_##NAME(const device T *x [[buffer(0)]],                               \
               const device T *a [[buffer(1)]],                               \
               const device T *zi [[buffer(2)]], device T *out [[buffer(3)]], \
               constant long &steps [[buffer(4)]],                            \
               constant long &order [[buffer(5)]],                            \
               uint lid [[thread_index_in_threadgroup]],                      \
               uint lane [[thread_index_in_simdgroup]],                       \
               uint simd_group [[simdgroup_index_in_threadgroup]],            \
               uint2 group [[threadgroup_position_in_grid]])                  \
    {                                                                         \
        threadgroup Acc<T>::type ring[MAX_ORDER];                             \
        threadgroup Acc<T>::type partials[2 * MAX_SIMD_GROUPS];               \
        const uint b = group.y;                                               \
        lpc_sequence<T>(x + b * steps, a + b * steps * order + lid,           \
                        zi + b * order, out + b * steps, uint(steps),         \
                        uint(order), lid, lane, simd_group, ring, partials);  \
    }

DEFINE_LPC(float, float)
DEFINE_LPC(half, half)
DEFINE_LPC(bfloat, bfloat)
DEFINE_LPC(float2, float2)
DEFINE_LPC(half2, half2)
