// The Metal kernels of recur_mps.mm, compiled after mps_common.metal once
// per state size and dtype, with STATES (M), SCALAR (float, half, bfloat, or
// float2 for complex64), THREADS (per threadgroup) and CHUNK (steps per
// chunk) defined, so that loops over the states unroll. README.md describes
// the algorithm: recur_reduce reduces each chunk but the last to its map
// h -> P h + o, the host scans the maps as the same recurrence, and
// recur_run runs each chunk from the state before it. A chunk's vector of
// states runs in a block of 8 by 8 x 8 matrix products (run_block) or on its
// own lanes (run_vector), and a map's P in blocks of 8 columns (map_block).

using T = SCALAR;
using A = Acc<T>::type;
constant constexpr uint M = STATES;
constant constexpr uint CHUNK_STEPS = CHUNK;
constant constexpr bool COMPLEX = sizeof(A) == 2 * sizeof(float);

// Loops over states, lanes and tiles unroll whole, or their arrays leave
// registers.
#define UNROLL _Pragma("clang loop unroll(full)")

constexpr uint power_of_2_from(uint n)
{
    uint p = 1;
    while (p < n)
        p *= 2;
    return p;
}

constant constexpr uint LANES =
    power_of_2_from(M) < SIMD_WIDTH ? power_of_2_from(M) : SIMD_WIDTH;
constant constexpr uint SLOTS = (M + LANES - 1) / LANES;
constant constexpr uint VECTORS = SIMD_WIDTH / LANES;
constant constexpr uint HALF = LANES / 2;
// Whether vectors hold states past M, which every access then checks.
constant constexpr bool PADDED = LANES * SLOTS != M;

// A lane's vector in the grid, and its lane in the vector, whose state in
// slot i is lane + LANES * i, live below M.
struct Lane
{
    uint vector, lane;
};

// The kernels run one-dimensional grids in threadgroups of a multiple of
// SIMD_WIDTH threads, so that thread `index` is lane `lane` of SIMD group
// index / SIMD_WIDTH.
inline Lane lane_of(uint index, uint lane)
{
    return {index / SIMD_WIDTH * VECTORS + lane / LANES, lane % LANES};
}

inline uint state_of(Lane l, uint slot) { return l.lane + LANES * slot; }

// Index i, below M, of states, rows or columns, or M - 1 past it, so that
// no address leaves the matrix or vector.
inline uint clamp(uint i) { return PADDED ? min(i, M - 1) : i; }

// Slot i's index past the lane's pointers, which are offset once by its
// first state, clamped.
inline uint column(Lane l, uint i)
{
    return PADDED ? clamp(state_of(l, i)) - clamp(l.lane) : LANES * i;
}

// The lane's share of row `row` of a step's matrix, at m, offset by the
// lane, by the states h: the products of its states' columns, zero past M.
inline A row_share(const device T *m, uint row, thread const A *h, Lane l)
{
    A sum = A(0.0f);
    UNROLL
    for (uint i = 0; i < SLOTS; ++i)
    {
        const A a = A(m[clamp(row) * M + column(l, i)]);
        if (!PADDED || (row < M && state_of(l, i) < M))
            sum += mul(a, h[i]);
    }
    return sum;
}

// A reduce-scatter over the vector's lanes: each round halves the rows a
// lane holds, keeping the half its lane's bit selects and adding its
// partner's, so that v[0] ends as the sum over the lanes of row `lane`'s
// shares.
template <uint SPAN> inline void butterfly(thread A *v, uint lane)
{
    const bool upper = lane & SPAN;
    UNROLL
    for (uint q = 0; q < SPAN; ++q)
    {
        const A keep = upper ? v[q + SPAN] : v[q];
        const A give = upper ? v[q] : v[q + SPAN];
        v[q] = keep + simd_shuffle_xor(give, ushort(SPAN));
    }
    butterfly<SPAN / 2>(v, lane);
}
template <> inline void butterfly<0>(thread A *, uint) {}

// The vector's states after a step, whose matrix is at m and inputs at x,
// offset by the lane: the matrix by the states, LANES rows at a time, whose
// shares the butterfly sums, then x, added last so that cancelling products
// meet first. The butterfly's first round is taken as the shares are
// formed, so that a lane holds half of them at most, in registers.
inline void step(const device T *m, const device T *x, thread A *h, Lane l)
{
    A next[SLOTS];
    UNROLL
    for (uint block = 0; block < SLOTS; ++block)
    {
        const uint first = block * LANES;
        A v[HALF > 0 ? HALF : 1];
        if (HALF == 0)
            v[0] = row_share(m, first, h, l);
        const uint kept = l.lane & HALF;
        UNROLL
        for (uint q = 0; q < HALF; ++q)
            v[q] = row_share(m, first + q + kept, h, l) +
                   simd_shuffle_xor(row_share(m, first + q + HALF - kept, h, l),
                                    ushort(HALF));
        butterfly<HALF / 2>(v, l.lane);
        const uint s = state_of(l, block);
        next[block] = s < M ? v[0] + A(x[column(l, block)]) : v[0];
    }
    UNROLL
    for (uint i = 0; i < SLOTS; ++i)
        h[i] = next[i];
}

// The sequence and chunk of vector v, and whether that chunk exists:
// vectors run `run` consecutive chunks of a sequence, those of each sequence
// in turn, then the next `run`, so that the vectors in flight read
// neighbouring memory.
inline bool sequence_chunk(uint v, uint n_batch, uint chunks, uint run,
                           thread uint &b, thread uint &c)
{
    const uint index = v / run;
    b = index % n_batch;
    c = index / n_batch * run + v % run;
    return c < chunks;
}

// The vector of thread `index` through its chunk (see sequence_chunk), as
// run_block's columns. Pointers advance a step at a time, as 64-bit integer
// arithmetic is emulated.
template <bool WRITE>
inline void run_vector(const device T *a, const device T *zi,
                       const device T *x, const device A *starts,
                       device T *out, device A *ends, uint index, uint lane,
                       uint n_batch, uint chunks, uint steps, long stride_b,
                       long stride_t, uint run)
{
    const Lane l = lane_of(index, lane);
    uint b, c;
    if (!sequence_chunk(l.vector, n_batch, chunks, run, b, c))
        return;
    A h[SLOTS];
    UNROLL
    for (uint i = 0; i < SLOTS; ++i)
    {
        const uint s = state_of(l, i);
        h[i] = A(0.0f);
        if (s < M && c > 0 && starts)
            h[i] = starts[(b * (chunks - 1) + c - 1) * M + s];
        else if (s < M && c == 0 && zi)
            h[i] = A(zi[b * M + s]);
    }
    const uint first = c * CHUNK_STEPS;
    const uint last = min(first + CHUNK_STEPS, steps);
    const long offset = (long(b) * steps + first) * M + clamp(l.lane);
    const device T *m = a + b * stride_b + first * stride_t + clamp(l.lane);
    x += offset;
    if (WRITE)
        out += offset;
    for (uint t = first; t < last; ++t, m += stride_t, x += M)
    {
        step(m, x, h, l);
        if (!WRITE)
            continue;
        UNROLL
        for (uint i = 0; i < SLOTS; ++i)
            if (state_of(l, i) < M)
                out[column(l, i)] = T(h[i]);
        out += M;
    }
    UNROLL
    for (uint i = 0; i < SLOTS; ++i)
        if (ends && state_of(l, i) < M)
            ends[(b * chunks + c) * M + state_of(l, i)] = h[i];
}

// The row and first column of a lane's two entries of an 8 x 8
// simdgroup_matrix, thread_elements(), in the layout of every Apple GPU,
// which MLX relies on too.
inline uint2 tile_entries(uint lane)
{
    const uint q = lane / 4;
    return uint2((q & 4) + (lane / 2) % 4, (q & 2) * 2 + (lane % 2) * 2);
}

inline float real_part(float v) { return v; }
inline float real_part(float2 v) { return v.x; }
inline float imag_part(float) { return 0.0f; }
inline float imag_part(float2 v) { return v.y; }

constant constexpr uint TILES = (M + 7) / 8;

// Entry (row, col) of a SIMD group's TILES x TILES tiles, which hold
// `packed` M x M matrices on their diagonal, 8 / M of them up to 4 states
// and 1 above: its matrix, its row and column there, and whether it lies in
// one.
struct Entry
{
    uint map, row, col;
    bool in;
};

inline Entry entry_of(uint row, uint col, uint packed)
{
    const uint g = row / M;
    return {g, row - g * M, col - (col / M) * M, g == col / M && g < packed};
}

// Tile (i, k) of a step's matrices, the lane's map's at m, in float, zero
// past the maps' blocks: its real part, its imaginary part and minus that.
struct Tile
{
    simdgroup_float8x8 re, im, minus_im;
};

inline Tile load_tile(const device T *m, uint i, uint k, uint2 e, uint packed)
{
    Tile t;
    UNROLL
    for (uint c = 0; c < 2; ++c)
    {
        const Entry n = entry_of(i * 8 + e.x, k * 8 + e.y + c, packed);
        const A v = A(m[n.row * M + n.col]);
        t.re.thread_elements()[c] = n.in ? real_part(v) : 0.0f;
        t.im.thread_elements()[c] = n.in ? imag_part(v) : 0.0f;
        t.minus_im.thread_elements()[c] = n.in ? -imag_part(v) : 0.0f;
    }
    return t;
}

template <typename P> inline void store_entry(device P *p, float re, float)
{
    *p = P(re);
}
inline void store_entry(device float2 *p, float re, float im)
{
    *p = float2(re, im);
}

// X <- A X for a SIMD group's block X, in tiles Xr and Xi, its real and
// imaginary parts, A the step's matrix at m (see load_tile); complex64 as
// four real products.
inline void multiply(const device T *m, thread simdgroup_float8x8 *Xr,
                     thread simdgroup_float8x8 *Xi, uint2 e, uint packed)
{
    simdgroup_float8x8 Yr[TILES], Yi[TILES];
    UNROLL
    for (uint i = 0; i < TILES; ++i)
    {
        Yr[i] = simdgroup_float8x8(0.0f);
        Yi[i] = simdgroup_float8x8(0.0f);
        UNROLL
        for (uint k = 0; k < TILES; ++k)
        {
            const Tile t = load_tile(m, i, k, e, packed);
            simdgroup_multiply_accumulate(Yr[i], t.re, Xr[k], Yr[i]);
            if (COMPLEX)
            {
                simdgroup_multiply_accumulate(Yr[i], t.minus_im, Xi[k], Yr[i]);
                simdgroup_multiply_accumulate(Yi[i], t.re, Xi[k], Yi[i]);
                simdgroup_multiply_accumulate(Yi[i], t.im, Xr[k], Yi[i]);
            }
        }
    }
    UNROLL
    for (uint i = 0; i < TILES; ++i)
    {
        Xr[i] = Yr[i];
        Xi[i] = Yi[i];
    }
}

// Columns 8 kb to 8 kb + 7 of maps first to first + packed - 1 of n_maps,
// map p * maps_per_a + j being A[p]'s chunk j's: the product of the chunk's
// matrices by those columns of the identity, zero for map 0 of
// time-varying A, whose chunk's offset holds zi. A lane's rows belong to one
// map, whose matrices it reads.
inline void map_block(const device T *a, device A *maps, uint first, uint kb,
                      uint packed, uint n_maps, uint maps_per_a,
                      long stride_b, long stride_t, uint lane)
{
    const uint2 e = tile_entries(lane);
    const uint map = min(first + min(e.x / M, packed - 1), n_maps - 1);
    const uint p = map / maps_per_a, j = map % maps_per_a;
    const bool zero = stride_t != 0 && j == 0;
    simdgroup_float8x8 Xr[TILES], Xi[TILES];
    UNROLL
    for (uint i = 0; i < TILES; ++i)
        UNROLL
        for (uint c = 0; c < 2; ++c)
        {
            const Entry n = entry_of(i * 8 + e.x, kb * 8 + e.y + c, packed);
            Xr[i].thread_elements()[c] =
                n.in && !zero && n.row == n.col ? 1.0f : 0.0f;
            Xi[i].thread_elements()[c] = 0.0f;
        }
    // Time-invariant A's map is that of any chunk: chunk 0's steps.
    const device T *m = a + p * stride_b + j * CHUNK_STEPS * stride_t;
    for (uint t = 0; t < CHUNK_STEPS; ++t, m += stride_t)
        multiply(m, Xr, Xi, e, packed);
    UNROLL
    for (uint i = 0; i < TILES; ++i)
        UNROLL
        for (uint c = 0; c < 2; ++c)
        {
            const Entry n = entry_of(i * 8 + e.x, kb * 8 + e.y + c, packed);
            if (n.in && first + n.map < n_maps)
            {
                const long at = (long(first + n.map) * M + n.row) * M + n.col;
                store_entry(maps + at, Xr[i].thread_elements()[c],
                            Xi[i].thread_elements()[c]);
            }
        }
}

// Blocks' columns are vectors in turn: the chunks of each sequence for
// time-invariant A, and the sequences of each chunk for time-varying A, so
// that a block's columns share their matrices. The inner count is padded to
// a multiple of 8 where a block must not span two.
struct Layout
{
    bool by_chunk;
    uint inner, padded, outer;
};

inline Layout layout(uint n_batch, uint chunks, long stride_t, long a_batch)
{
    const bool by_chunk = stride_t != 0;
    const uint inner = by_chunk ? n_batch : chunks;
    const uint padded = by_chunk || a_batch > 1 ? (inner + 7) / 8 * 8 : inner;
    return {by_chunk, inner, padded, by_chunk ? chunks : n_batch};
}

// The sequence and chunk of a block's column k, and whether that chunk
// exists.
inline bool block_column(Layout o, uint block, uint k, thread uint &b,
                         thread uint &c)
{
    const uint v = block * 8 + k, outer = v / o.padded, i = v % o.padded;
    b = o.by_chunk ? i : outer;
    c = o.by_chunk ? outer : i;
    return i < o.inner && outer < o.outer;
}

// The 8 vectors of block `block` through their chunks (see block_column),
// as the columns of X <- A[t] X + x[t]: from starts[b, c - 1], the state at
// the previous chunk's end, for c > 0, and from zi[b] for c = 0, or zero
// without them. With WRITE, their states go to out; with ends, their end
// states to ends[b, c].
template <bool WRITE>
inline void run_block(const device T *a, const device T *zi,
                      const device T *x, const device A *starts,
                      device T *out, device A *ends, uint block, uint lane,
                      uint n_batch, uint chunks, uint steps, long stride_b,
                      long stride_t, long a_batch)
{
    const Layout o = layout(n_batch, chunks, stride_t, a_batch);
    uint b0, c0;
    if (!block_column(o, block, 0, b0, c0))
        return;
    const uint2 e = tile_entries(lane);
    uint b[2], c[2];
    bool live[2];
    UNROLL
    for (uint k = 0; k < 2; ++k)
        live[k] = block_column(o, block, e.y + k, b[k], c[k]);
    simdgroup_float8x8 Xr[TILES], Xi[TILES];
    UNROLL
    for (uint i = 0; i < TILES; ++i)
        UNROLL
        for (uint k = 0; k < 2; ++k)
        {
            const uint row = min(i * 8 + e.x, M - 1);
            const bool in = live[k] && i * 8 + e.x < M;
            A h = A(0.0f);
            if (in && c[k] > 0 && starts)
                h = starts[(b[k] * (chunks - 1) + c[k] - 1) * M + row];
            else if (in && c[k] == 0 && zi)
                h = A(zi[b[k] * M + row]);
            Xr[i].thread_elements()[k] = in ? real_part(h) : 0.0f;
            Xi[i].thread_elements()[k] = in ? imag_part(h) : 0.0f;
        }
    // The columns share their matrices: of sequence b0, whose constant matrix
    // serves chunks of any length, or of chunk c0, whose steps end with the
    // sequences', so that no matrix past A's last is read.
    const device T *m = a + b0 * stride_b + c0 * CHUNK_STEPS * stride_t;
    const uint last = stride_t == 0
        ? CHUNK_STEPS
        : min(CHUNK_STEPS, steps - c0 * CHUNK_STEPS);
    for (uint t = 0; t < last; ++t, m += stride_t)
    {
        multiply(m, Xr, Xi, e, 1);
        UNROLL
        for (uint i = 0; i < TILES; ++i)
            UNROLL
            for (uint k = 0; k < 2; ++k)
            {
                const uint row = i * 8 + e.x, s = c[k] * CHUNK_STEPS + t;
                if (!live[k] || row >= M || s >= steps)
                    continue;
                const long at = (long(b[k]) * steps + s) * M + row;
                const A v = A(x[at]);
                const float re = Xr[i].thread_elements()[k] + real_part(v);
                const float im = Xi[i].thread_elements()[k] + imag_part(v);
                Xr[i].thread_elements()[k] = re;
                Xi[i].thread_elements()[k] = im;
                if (WRITE)
                    store_entry(out + at, re, im);
            }
    }
    if (ends)
        UNROLL
        for (uint i = 0; i < TILES; ++i)
            UNROLL
            for (uint k = 0; k < 2; ++k)
            {
                const uint row = i * 8 + e.x;
                if (live[k] && row < M)
                    store_entry(ends + (b[k] * chunks + c[k]) * M + row,
                                Xr[i].thread_elements()[k],
                                Xi[i].thread_elements()[k]);
            }
}

// Chunk c of sequence b runs from starts[b, c - 1], the state at the
// previous chunk's end, for c > 0, and for c = 0 from zi[b] if from_zi, and
// zero otherwise, in a block of 8 if blocked (see run_block), and otherwise
// as a vector (see run_vector). A[b] is a + b * stride_b, its step t's
// matrix t * stride_t after.
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void recur_run(const device T *a [[buffer(0)]],
               const device T *zi [[buffer(1)]],
               const device T *x [[buffer(2)]],
               const device A *starts [[buffer(3)]],
               device T *out [[buffer(4)]],
               constant long &batch [[buffer(5)]],
               constant long &steps [[buffer(6)]],
               constant long &stride_b [[buffer(7)]],
               constant long &stride_t [[buffer(8)]],
               constant long &a_batch [[buffer(9)]],
               constant long &from_zi [[buffer(10)]],
               constant long &run [[buffer(11)]],
               constant long &blocked [[buffer(12)]],
               uint index [[thread_position_in_grid]],
               uint lane [[thread_index_in_simdgroup]])
{
    const uint chunks = (uint(steps) - 1) / CHUNK_STEPS + 1;
    if (blocked)
        run_block<true>(a, from_zi ? zi : nullptr, x, starts, out, nullptr,
                        index / SIMD_WIDTH, lane, batch, chunks, steps,
                        stride_b, stride_t, a_batch);
    else
        run_vector<true>(a, from_zi ? zi : nullptr, x, starts, out, nullptr,
                         index, lane, batch, chunks, steps, stride_b,
                         stride_t, run);
}

// The maps of chunks 0 to chunks - 2. The first offset_groups SIMD groups
// write o of chunk c of sequence b to offsets[b, c], as recur_run runs it
// but from zero for c > 0. The SIMD groups after them write P to maps[p, j],
// one per chunk j of time-varying A[p] and one for time-invariant A[p]:
// SIMD group g * TILES + kb after them writes columns 8 kb to 8 kb + 7 of
// maps g * packed to g * packed + packed - 1 (see map_block).
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void recur_reduce(const device T *a [[buffer(0)]],
                  const device T *zi [[buffer(1)]],
                  const device T *x [[buffer(2)]],
                  device A *offsets [[buffer(3)]],
                  device A *maps [[buffer(4)]],
                  constant long &batch [[buffer(5)]],
                  constant long &steps [[buffer(6)]],
                  constant long &stride_b [[buffer(7)]],
                  constant long &stride_t [[buffer(8)]],
                  constant long &a_batch [[buffer(9)]],
                  constant long &from_zi [[buffer(10)]],
                  constant long &run [[buffer(11)]],
                  constant long &blocked [[buffer(12)]],
                  constant long &packed [[buffer(13)]],
                  constant long &offset_groups [[buffer(14)]],
                  uint index [[thread_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]])
{
    const uint maps_per_item = (uint(steps) - 1) / CHUNK_STEPS;
    const uint group = index / SIMD_WIDTH;
    if (group < offset_groups && blocked)
        return run_block<false>(a, from_zi ? zi : nullptr, x, nullptr, nullptr,
                                offsets, group, lane, batch, maps_per_item,
                                steps, stride_b, stride_t, a_batch);
    if (group < offset_groups)
        return run_vector<false>(a, from_zi ? zi : nullptr, x, nullptr,
                                 nullptr, offsets, index, lane, batch,
                                 maps_per_item, steps, stride_b, stride_t, run);
    const uint maps_per_a = stride_t == 0 ? 1 : maps_per_item;
    const uint n_maps = uint(a_batch) * maps_per_a;
    const uint block = group - uint(offset_groups);
    const uint kb = block % TILES, first = block / TILES * uint(packed);
    if (first < n_maps)
        map_block(a, maps, first, kb, uint(packed), n_maps, maps_per_a,
                  stride_b, stride_t, lane);
}
