// The Metal kernels of the scalar recurrence, scalar_recur_mps.mm's: setup.py
// embeds this file in the extension, which torch's Metal shim compiles on
// first use.
//
// h[b, t] = a h[b, t - 1] + x[b, t], h[b, -1] = zi[b], with a one coefficient
// per step (time-varying, lpv) or per sequence (time-invariant, lti), as a
// scan of the affine maps f(h) = m h + v, in three passes over tiles of
// TILE = THREADS * K steps, one threadgroup each (see README.md): reduce each
// tile to its map, scan the tiles' maps of each sequence into the state at
// each tile's end, and replay each tile from the state before it. Every map
// from a sequence's start has m = 0, as zi is folded into the first step,
// (0, a zi + x[b, 0]), so a tile needs only the previous tile's end state, and
// tile 0 none: a sequence of one tile runs the replay alone.
#include <metal_stdlib>
using namespace metal;

// Every kernel runs THREADS threads per threadgroup, which each declares as
// its maximum so that the compiler guarantees it, in SIMD groups of 32 lanes,
// the width of every Apple GPU, which torch's Metal shim can't query.
constant constexpr uint THREADS = 256;
constant constexpr uint SIMD_GROUPS = THREADS / 32;
// Consecutive steps per thread, read and written as one vector.
constant constexpr uint K = 4;
constant constexpr uint TILE = THREADS * K;
// Tile totals per thread in the scan of the tiles.
constant constexpr uint TOTALS = 2;

// The accumulation type: float for float, half and bfloat; float2, a complex
// number, for complex64.
template <typename T> struct Acc
{
    using type = float;
    static float one() { return 1.0f; }
};
template <> struct Acc<float2>
{
    using type = float2;
    static float2 one() { return float2(1.0f, 0.0f); }
};

inline float mul(float a, float b) { return a * b; }
inline float2 mul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// The map f(h) = m h + v.
template <typename A> struct Pair
{
    A m;
    A v;
};

template <typename A> inline Pair<A> identity()
{
    return {Acc<A>::one(), A(0.0f)};
}

// right(left(h)): composition isn't commutative.
template <typename A> inline Pair<A> compose(Pair<A> left, Pair<A> right)
{
    return {mul(left.m, right.m), mul(left.v, right.m) + right.v};
}

// The map of a thread's K steps.
template <typename A> inline Pair<A> compose_steps(thread const Pair<A> *items)
{
    Pair<A> total = items[0];
    for (uint k = 1; k < K; ++k)
        total = compose(total, items[k]);
    return total;
}

template <typename A> inline Pair<A> shuffle_up(Pair<A> p, ushort delta)
{
    return {simd_shuffle_up(p.m, delta), simd_shuffle_up(p.v, delta)};
}

template <typename A> inline Pair<A> simd_inclusive(Pair<A> p, uint lane)
{
    for (ushort offset = 1; offset < 32; offset <<= 1)
    {
        const Pair<A> left = shuffle_up(p, offset);
        if (lane >= offset)
            p = compose(left, p);
    }
    return p;
}

// One sequence's inputs. Its pointers are computed once per thread, and the
// steps within it indexed in 32 bits: 64-bit integer arithmetic is emulated.
// Time-varying coefficients are one per step, a[b * steps + t]; time-invariant
// ones a[b * stride_b], shared (stride_b = 0) or one per sequence.
template <typename T> struct Sequence
{
    const device T *decays;
    const device T *zi;
    const device T *x;
    uint steps;
};

template <typename T>
inline Sequence<T> sequence(const device T *decays, const device T *zi,
                            const device T *x, uint b, long steps, long stride_b)
{
    return {decays + b * stride_b, zi + b, x + b * steps, uint(steps)};
}

// Four consecutive steps with one vector load or store, which need only the
// element's alignment: a SIMD group reads 32 * 4 consecutive steps.
inline void load4(const device float *p, thread float *v)
{
    const packed_float4 q = *(const device packed_float4 *)p;
    v[0] = q[0]; v[1] = q[1]; v[2] = q[2]; v[3] = q[3];
}
inline void store4(device float *p, thread const float *v)
{
    *(device packed_float4 *)p = packed_float4(v[0], v[1], v[2], v[3]);
}
inline void load4(const device half *p, thread half *v)
{
    const packed_half4 q = *(const device packed_half4 *)p;
    v[0] = q[0]; v[1] = q[1]; v[2] = q[2]; v[3] = q[3];
}
inline void store4(device half *p, thread const half *v)
{
    *(device packed_half4 *)p = packed_half4(v[0], v[1], v[2], v[3]);
}
inline void load4(const device float2 *p, thread float2 *v)
{
    const float4 a = float4(((const device packed_float4 *)p)[0]);
    const float4 b = float4(((const device packed_float4 *)p)[1]);
    v[0] = a.xy; v[1] = a.zw; v[2] = b.xy; v[3] = b.zw;
}
inline void store4(device float2 *p, thread const float2 *v)
{
    ((device packed_float4 *)p)[0] = packed_float4(v[0].x, v[0].y, v[1].x, v[1].y);
    ((device packed_float4 *)p)[1] = packed_float4(v[2].x, v[2].y, v[3].x, v[3].y);
}
inline void load4(const device bfloat *p, thread bfloat *v)
{
    const packed_bfloat4 q = *(const device packed_bfloat4 *)p;
    v[0] = q[0]; v[1] = q[1]; v[2] = q[2]; v[3] = q[3];
}
inline void store4(device bfloat *p, thread const bfloat *v)
{
    *(device packed_bfloat4 *)p = packed_bfloat4(v[0], v[1], v[2], v[3]);
}

template <typename T, bool VARYING, typename A = typename Acc<T>::type>
inline void load_steps(Sequence<T> s, uint first, uint lid,
                       thread Pair<A> *items)
{
    const uint t0 = first + lid * K;
    T xv[K], av[K];
    if (t0 + K <= s.steps)
    {
        load4(s.x + t0, xv);
        if (VARYING)
            load4(s.decays + t0, av);
    }
    else
    {
        for (uint k = 0; k < K; ++k)
        {
            xv[k] = t0 + k < s.steps ? s.x[t0 + k] : T(0.0f);
            if (VARYING)
                av[k] = t0 + k < s.steps ? s.decays[t0 + k] : T(0.0f);
        }
    }
    const A shared = VARYING ? A(0.0f) : A(*s.decays);
    for (uint k = 0; k < K; ++k)
    {
        const uint t = t0 + k;
        if (t >= s.steps)
        {
            items[k] = identity<A>();
            continue;
        }
        const A a = VARYING ? A(av[k]) : shared;
        const A value = A(xv[k]);
        items[k] = t == 0 ? Pair<A>{A(0.0f), mul(a, A(*s.zi)) + value}
                          : Pair<A>{a, value};
    }
}

// The map of a threadgroup's thread totals before each thread's.
template <typename A>
inline Pair<A> group_scan(Pair<A> total, uint lane, uint simd_group,
                          threadgroup Pair<A> *sums)
{
    const Pair<A> inclusive = simd_inclusive(total, lane);
    if (lane == 31)
        sums[simd_group] = inclusive;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0)
    {
        Pair<A> group = lane < SIMD_GROUPS ? sums[lane] : identity<A>();
        group = simd_inclusive(group, lane);
        // Stored once every lane has read its sum.
        const Pair<A> prior = shuffle_up(group, 1);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < SIMD_GROUPS)
            sums[lane] = lane == 0 ? identity<A>() : prior;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const Pair<A> prior_thread = shuffle_up(inclusive, 1);
    return compose(sums[simd_group], lane == 0 ? identity<A>() : prior_thread);
}

template <typename T, bool VARYING, typename A = typename Acc<T>::type>
inline void reduce_tiles(Sequence<T> s, device Pair<A> *totals, long n_tiles,
                         uint lid, uint lane, uint simd_group, uint2 group,
                         threadgroup Pair<A> *sums)
{
    Pair<A> items[K];
    load_steps<T, VARYING>(s, group.x * TILE, lid, items);
    const Pair<A> total = compose_steps(items);
    const Pair<A> exclusive = group_scan(total, lane, simd_group, sums);
    if (lid == THREADS - 1)
        totals[group.y * n_tiles + group.x] = compose(exclusive, total);
}

// One threadgroup per sequence, over its tiles THREADS * TOTALS at a time,
// carrying the last state between them.
template <typename A>
inline void scan_tiles(const device Pair<A> *totals, device A *states,
                       long n_tiles, uint lid, uint lane, uint simd_group,
                       uint2 group, threadgroup Pair<A> *sums,
                       threadgroup Pair<A> *carry)
{
    totals += group.y * n_tiles;
    states += group.y * n_tiles;
    const uint tiles = uint(n_tiles);
    if (lid == 0)
        *carry = identity<A>();
    for (uint chunk = 0; chunk < tiles; chunk += THREADS * TOTALS)
    {
        Pair<A> items[TOTALS];
        Pair<A> total = identity<A>();
        for (uint k = 0; k < TOTALS; ++k)
        {
            const uint tile = chunk + lid * TOTALS + k;
            items[k] = tile < tiles ? totals[tile] : identity<A>();
            total = compose(total, items[k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const Pair<A> exclusive = group_scan(total, lane, simd_group, sums);
        Pair<A> prefix = compose(*carry, exclusive);
        for (uint k = 0; k < TOTALS; ++k)
        {
            prefix = compose(prefix, items[k]);
            const uint tile = chunk + lid * TOTALS + k;
            if (tile < tiles)
                states[tile] = prefix.v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == THREADS - 1)
            *carry = prefix;
    }
}

template <typename T, bool VARYING, typename A = typename Acc<T>::type>
inline void replay_tiles(Sequence<T> s, const device A *states, device T *out,
                         long n_tiles, uint lid, uint lane, uint simd_group,
                         uint2 group, threadgroup Pair<A> *sums)
{
    const uint first = group.x * TILE;
    Pair<A> items[K];
    load_steps<T, VARYING>(s, first, lid, items);
    const Pair<A> total = compose_steps(items);
    const Pair<A> exclusive = group_scan(total, lane, simd_group, sums);
    A h = group.x == 0 ? A(0.0f) : states[group.y * n_tiles + group.x - 1];
    h = mul(h, exclusive.m) + exclusive.v;
    const uint t0 = first + lid * K;
    T hv[K];
    for (uint k = 0; k < K; ++k)
    {
        h = mul(h, items[k].m) + items[k].v;
        hv[k] = T(h);
    }
    out += group.y * long(s.steps);
    if (t0 + K <= s.steps)
        store4(out + t0, hv);
    else
        for (uint k = 0; k < K && t0 + k < s.steps; ++k)
            out[t0 + k] = hv[k];
}

#define THREAD_ARGUMENTS                                                      \
    uint lid [[thread_index_in_threadgroup]],                                 \
        uint lane [[thread_index_in_simdgroup]],                              \
        uint simd_group [[simdgroup_index_in_threadgroup]],                   \
        uint2 group [[threadgroup_position_in_grid]]

#define DEFINE_PASSES(T, NAME, KIND, VARYING)                                 \
    [[max_total_threads_per_threadgroup(THREADS)]] kernel void                \
    scalar_recur_reduce_##NAME##_##KIND(                                      \
        const device T *decays [[buffer(0)]],                                 \
        const device T *zi [[buffer(1)]], const device T *x [[buffer(2)]],    \
        device Pair<Acc<T>::type> *totals [[buffer(3)]],                      \
        constant long &steps [[buffer(4)]],                                   \
        constant long &stride_b [[buffer(5)]],                                \
        constant long &n_tiles [[buffer(6)]], THREAD_ARGUMENTS)               \
    {                                                                         \
        threadgroup Pair<Acc<T>::type> sums[SIMD_GROUPS];                     \
        reduce_tiles<T, VARYING>(                                             \
            sequence(decays, zi, x, group.y, steps, stride_b), totals,        \
            n_tiles, lid, lane, simd_group, group, sums);                     \
    }                                                                         \
    [[max_total_threads_per_threadgroup(THREADS)]] kernel void                \
    scalar_recur_replay_##NAME##_##KIND(                                      \
        const device T *decays [[buffer(0)]],                                 \
        const device T *zi [[buffer(1)]], const device T *x [[buffer(2)]],    \
        const device Acc<T>::type *states [[buffer(3)]],                      \
        device T *out [[buffer(4)]], constant long &steps [[buffer(5)]],      \
        constant long &stride_b [[buffer(6)]],                                \
        constant long &n_tiles [[buffer(7)]], THREAD_ARGUMENTS)               \
    {                                                                         \
        threadgroup Pair<Acc<T>::type> sums[SIMD_GROUPS];                     \
        replay_tiles<T, VARYING>(                                             \
            sequence(decays, zi, x, group.y, steps, stride_b), states, out,   \
            n_tiles, lid, lane, simd_group, group, sums);                     \
    }

#define DEFINE_KERNELS(T, NAME)                                               \
    DEFINE_PASSES(T, NAME, lti, false)                                        \
    DEFINE_PASSES(T, NAME, lpv, true)

// The scan of tile totals depends on the accumulation type alone.
#define DEFINE_SCAN(A, NAME)                                                  \
    [[max_total_threads_per_threadgroup(THREADS)]] kernel void                \
    scalar_recur_scan_##NAME(const device Pair<A> *totals [[buffer(0)]],      \
                             device A *states [[buffer(1)]],                  \
                             constant long &n_tiles [[buffer(2)]],            \
                             THREAD_ARGUMENTS)                                \
    {                                                                         \
        threadgroup Pair<A> sums[SIMD_GROUPS];                                \
        threadgroup Pair<A> carry;                                            \
        scan_tiles(totals, states, n_tiles, lid, lane, simd_group, group,     \
                   sums, &carry);                                             \
    }

DEFINE_KERNELS(float, float)
DEFINE_KERNELS(half, half)
DEFINE_KERNELS(bfloat, bfloat)
DEFINE_KERNELS(float2, cfloat)
DEFINE_SCAN(float, float)
DEFINE_SCAN(float2, cfloat)
