// The Metal kernels of the recurrence of M states, recur_mps.mm's: setup.py
// embeds this file in the extension, and the host compiles it after
// mps_common.metal once per state size and dtype, defining STATES (M),
// SCALAR (the dtype: float, half, bfloat or float2, which is complex64) and
// THREADS (per threadgroup) before both, so that loops over the states
// unroll.
//
// h[b, t] = A[b, t] h[b, t - 1] + x[b, t], h[b, -1] = zi[b], in chunks of
// `chunk` steps, as the Triton kernels of _recur_kernels.py: the state at a
// chunk's end is a map of the state at its start, h -> P h + o, with P the
// product of the chunk's matrices, A^chunk for time-invariant A, and o the
// chunk's end state from a zero start. Chunk 0 starts from zi instead, so
// the scan of the maps starts from zero. Two kernels (see README.md): reduce
// each chunk but the last to its map, which the host scans, as the same
// recurrence over the chunks, recursively, into the state at each chunk's
// end, and run each chunk from the state before it.
//
// A vector of M states, a chunk's state, runs on LANES lanes of a SIMD group,
// M rounded up to a power of 2, 32 at most, each holding SLOTS states: lane
// j holds states j, j + LANES, ..., and reads their columns of each matrix,
// so that a vector's lanes read the matrix's rows in order. Each row's
// products are summed over the lanes by a butterfly of shuffles (see step).
// A map's P, whose columns are vectors of their own, runs 8 columns at a
// time on a SIMD group, by 8 x 8 matrix products (see map_block).

using T = SCALAR;
using A = Acc<T>::type;
constant constexpr uint M = STATES;
constant constexpr bool COMPLEX = sizeof(A) == 2 * sizeof(float);

// Loops over states, lanes and tiles unroll whole: left to the compiler, the
// larger ones keep a loop, whose arrays then live in memory, as much as 30
// times as slow.
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

// The lane's share of row `row` of a step's matrix, at m, by the states h:
// the products of its states' columns, zero past M, where it reads the
// matrix's last row or column, so that no address leaves it.
inline A row_share(const device T *m, uint row, thread const A *h, Lane l)
{
    A sum = A(0.0f);
    UNROLL
    for (uint i = 0; i < SLOTS; ++i)
    {
        const uint s = state_of(l, i);
        const A a = A(m[min(row, M - 1) * M + min(s, M - 1)]);
        if (row < M && s < M)
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
// added if ADD: the matrix by the states, a block of LANES rows at a time,
// whose shares the butterfly sums into the block's states, one per lane,
// then x, which, added to a partial sum, could be rounded away before
// cancelling products meet. The butterfly's first round is taken as the
// shares are formed, so that a lane holds half of a block's at most: with
// all of them, complex64 at M = 32 spilled its registers and ran six times
// as long. Every lane of the vector calls it.
template <bool ADD>
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
        next[block] = ADD && s < M ? v[0] + A(x[min(s, M - 1)]) : v[0];
    }
    UNROLL
    for (uint i = 0; i < SLOTS; ++i)
        h[i] = next[i];
}

// The vector's states h through steps first to last - 1 of a sequence,
// whose matrices are m + t * stride_t and inputs x + t * M, added if ADD;
// if WRITE, the states are written to out + t * M. Pointers advance a step
// at a time, as 64-bit integer arithmetic is emulated.
template <bool ADD, bool WRITE>
inline void run_steps(const device T *m, long stride_t, const device T *x,
                      device T *out, uint first, uint last, thread A *h,
                      Lane l)
{
    m += first * stride_t;
    x += long(first) * M;
    if (WRITE)
        out += long(first) * M;
    for (uint t = first; t < last; ++t)
    {
        step<ADD>(m, x, h, l);
        m += stride_t;
        x += M;
        if (WRITE)
        {
            UNROLL
            for (uint i = 0; i < SLOTS; ++i)
                if (state_of(l, i) < M)
                    out[state_of(l, i)] = T(h[i]);
            out += M;
        }
    }
}

// The vector's states from v[0], ..., v[M - 1], or zero if v is null.
template <typename V>
inline void load_states(thread A *h, const device V *v, Lane l)
{
    UNROLL
    for (uint i = 0; i < SLOTS; ++i)
    {
        const uint s = state_of(l, i);
        h[i] = v && s < M ? A(v[min(s, M - 1)]) : A(0.0f);
    }
}

// Vectors run `run` consecutive chunks of a sequence, those of each
// sequence in turn, then the next `run` chunks: the vectors in flight then
// read neighbouring inputs, and the matrices of time-varying A shared over
// the batch at once. sequence_chunk sets the sequence and chunk of vector v,
// one of n_batch * ceil(chunks / run) * run, and returns whether that chunk
// exists.
inline bool sequence_chunk(uint v, uint n_batch, uint chunks, uint run,
                           thread uint &b, thread uint &c)
{
    const uint index = v / run;
    b = index % n_batch;
    c = index / n_batch * run + v % run;
    return c < chunks;
}

// A vector runs chunk c of sequence b (see sequence_chunk), from starts[b,
// c - 1], the state at the previous chunk's end, for c > 0, and for c = 0
// from zi[b] if from_zi, and zero otherwise. A[b] is a + b * stride_b, its
// step t's matrix t * stride_t after.
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void recur_run(const device T *a [[buffer(0)]],
               const device T *zi [[buffer(1)]],
               const device T *x [[buffer(2)]],
               const device A *starts [[buffer(3)]],
               device T *out [[buffer(4)]],
               constant long &batch [[buffer(5)]],
               constant long &steps [[buffer(6)]],
               constant long &chunk [[buffer(7)]],
               constant long &stride_b [[buffer(8)]],
               constant long &stride_t [[buffer(9)]],
               constant long &from_zi [[buffer(10)]],
               constant long &run [[buffer(11)]],
               uint index [[thread_position_in_grid]],
               uint lane [[thread_index_in_simdgroup]])
{
    const Lane l = lane_of(index, lane);
    const uint n_batch = uint(batch), length = uint(chunk);
    const uint chunks = (uint(steps) - 1) / length + 1;
    uint b, c;
    if (!sequence_chunk(l.vector, n_batch, chunks, uint(run), b, c))
        return;
    A h[SLOTS];
    if (c > 0)
        load_states(h, starts + (b * (chunks - 1) + c - 1) * M, l);
    else
        load_states(h, from_zi ? zi + b * M : nullptr, l);
    const long offset = long(b) * steps * M;
    run_steps<true, true>(a + b * stride_b, stride_t, x + offset, out + offset,
                          c * length, min(c * length + length, uint(steps)),
                          h, l);
}

// The lane's two entries of an 8 x 8 tile of a SIMD group's matrix,
// thread_elements(): their row, and the first of their two columns. The
// layout is that of every Apple GPU, which MLX's matrix products rely on.
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

// Tile (i, k) of a step's matrix at m, in float, padded with zeros past M:
// its real part, its imaginary part and minus that.
struct Tile
{
    simdgroup_float8x8 re, im, minus_im;
};

inline Tile load_tile(const device T *m, uint i, uint k, uint2 e)
{
    Tile t;
    UNROLL
    for (uint c = 0; c < 2; ++c)
    {
        const uint row = i * 8 + e.x, col = k * 8 + e.y + c;
        const A v = A(m[min(row, M - 1) * M + min(col, M - 1)]);
        const bool in = row < M && col < M;
        t.re.thread_elements()[c] = in ? real_part(v) : 0.0f;
        t.im.thread_elements()[c] = in ? imag_part(v) : 0.0f;
        t.minus_im.thread_elements()[c] = in ? -imag_part(v) : 0.0f;
    }
    return t;
}

inline void store_entry(device float *p, float re, float) { *p = re; }
inline void store_entry(device float2 *p, float re, float im)
{
    *p = float2(re, im);
}

// Columns 8 kb to 8 kb + 7 of map j of A[p], a SIMD group's: the product of
// the chunk's matrices, steps j * length on, by those columns of the identity,
// an M x 8 block X <- A[t] X in tiles of 8 x 8, its real and imaginary parts
// apart; zero for map 0 of time-varying A, whose chunk's offset holds zi.
// X's tiles live in registers, 2 * TILES floats per lane, whatever the
// chunk, where Triton's maps from M = 32 on alternate between two scratch
// buffers.
inline void map_block(const device T *a, device A *maps, uint p, uint j,
                      uint kb, uint maps_per_a, uint length, long stride_b,
                      long stride_t, uint lane)
{
    const uint2 e = tile_entries(lane);
    const bool zero = stride_t != 0 && j == 0;
    simdgroup_float8x8 Xr[TILES], Xi[TILES];
    UNROLL
    for (uint i = 0; i < TILES; ++i)
        UNROLL
        for (uint c = 0; c < 2; ++c)
        {
            const uint row = i * 8 + e.x, col = kb * 8 + e.y + c;
            Xr[i].thread_elements()[c] =
                !zero && row == col && row < M ? 1.0f : 0.0f;
            Xi[i].thread_elements()[c] = 0.0f;
        }
    if (!zero)
    {
        // Time-invariant A's map is that of any chunk: chunk 0's steps.
        const device T *m = a + p * stride_b + j * length * stride_t;
        for (uint t = 0; t < length; ++t, m += stride_t)
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
                    const Tile t = load_tile(m, i, k, e);
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
    }
    device A *P = maps + long(p * maps_per_a + j) * M * M;
    UNROLL
    for (uint i = 0; i < TILES; ++i)
        UNROLL
        for (uint c = 0; c < 2; ++c)
        {
            const uint row = i * 8 + e.x, col = kb * 8 + e.y + c;
            if (row < M && col < M)
                store_entry(P + row * M + col, Xr[i].thread_elements()[c],
                            Xi[i].thread_elements()[c]);
        }
}

// The maps of chunks 0 to chunks - 2, each of chunk steps. Its first SIMD
// groups hold n_batch * ceil((chunks - 1) / run) * run vectors, which write
// o of chunk c of sequence b (see sequence_chunk) to offsets[b, c]: its end
// state from zero, or for c = 0 from zi[b] if from_zi. The SIMD groups after
// write P, one per chunk of time-varying A and one for time-invariant A, all
// of whose chunks share it: SIMD group (p * maps_per_a + j) * TILES + kb
// after them writes columns 8 kb to 8 kb + 7 of map j of A[p], a_batch of
// them, to maps[p, j] (see map_block). Chunk 0's o holds zi, so its P is
// zero for time-varying A, and the scan of the maps starts from zero, which
// time-invariant A's P, shared by chunk 0, then leaves at zero.
[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void recur_reduce(const device T *a [[buffer(0)]],
                  const device T *zi [[buffer(1)]],
                  const device T *x [[buffer(2)]],
                  device A *offsets [[buffer(3)]],
                  device A *maps [[buffer(4)]],
                  constant long &batch [[buffer(5)]],
                  constant long &steps [[buffer(6)]],
                  constant long &chunk [[buffer(7)]],
                  constant long &stride_b [[buffer(8)]],
                  constant long &stride_t [[buffer(9)]],
                  constant long &from_zi [[buffer(10)]],
                  constant long &a_batch [[buffer(11)]],
                  constant long &run [[buffer(12)]],
                  uint index [[thread_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]])
{
    const uint n_batch = uint(batch), length = uint(chunk);
    const uint maps_per_item = (uint(steps) - 1) / length;
    const uint n_offsets =
        n_batch * ((maps_per_item - 1) / uint(run) + 1) * uint(run);
    const uint offset_groups = (n_offsets - 1) / VECTORS + 1;
    const uint group = index / SIMD_WIDTH;
    if (group < offset_groups)
    {
        const Lane l = lane_of(index, lane);
        uint b, c;
        if (!sequence_chunk(l.vector, n_batch, maps_per_item, uint(run), b, c))
            return;
        A h[SLOTS];
        load_states(h, c == 0 && from_zi ? zi + b * M : nullptr, l);
        run_steps<true, false>(a + b * stride_b, stride_t,
                               x + long(b) * steps * M, nullptr, c * length,
                               c * length + length, h, l);
        UNROLL
        for (uint i = 0; i < SLOTS; ++i)
            if (state_of(l, i) < M)
                offsets[(b * maps_per_item + c) * M + state_of(l, i)] = h[i];
        return;
    }
    const uint maps_per_a = stride_t == 0 ? 1 : maps_per_item;
    const uint block = group - offset_groups;
    const uint kb = block % TILES, item = block / TILES;
    if (item >= uint(a_batch) * maps_per_a)
        return;
    map_block(a, maps, item / maps_per_a, item % maps_per_a, kb, maps_per_a,
              length, stride_b, stride_t, lane);
}
