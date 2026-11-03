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
// A vector of M states, a chunk's state or a column of its P, runs on LANES
// lanes of a SIMD group, M rounded up to a power of 2, the lanes past M
// holding zeros. Lane j holds state j and reads column j of each matrix, so
// that a vector's lanes read a matrix in order, and the products of its
// rows are summed over the lanes by a butterfly of shuffles (see step).

using T = SCALAR;
using A = Acc<T>::type;
constant constexpr uint M = STATES;

constexpr uint power_of_2_from(uint n)
{
    uint p = 1;
    while (p < n)
        p *= 2;
    return p;
}

constant constexpr uint LANES = power_of_2_from(M);
constant constexpr uint VECTORS = SIMD_WIDTH / LANES;

// A lane's vector in the grid, and its state, live below M.
struct Lane
{
    uint vector, state;
};

// The kernels run one-dimensional grids in threadgroups of a multiple of
// SIMD_WIDTH threads, so that thread `index` is lane `lane` of SIMD group
// index / SIMD_WIDTH.
inline Lane lane_of(uint index, uint lane)
{
    return {index / SIMD_WIDTH * VECTORS + lane / LANES, lane % LANES};
}

// Column j of a step's matrix, padded to LANES rows, and input j if ADD; a
// lane past M reads zeros, from state M - 1's addresses, which it holds.
template <bool ADD> struct Column
{
    A m[LANES];
    A x;
};

template <bool ADD>
inline Column<ADD> load_column(const device T *m, const device T *x, Lane l)
{
    Column<ADD> c;
    const bool live = l.state < M;
    for (uint q = 0; q < LANES; ++q)
        c.m[q] = live && q < M ? A(m[q * M]) : A(0.0f);
    c.x = ADD && live ? A(*x) : A(0.0f);
    return c;
}

// A reduce-scatter over the vector's lanes, from v, each lane's products of
// its column by its state: each round halves the rows a lane holds, keeping
// the half its state's bit selects and adding its partner's, so that v[0]
// ends as the sum over the lanes of row `state`'s products.
template <uint SPAN> inline void butterfly(thread A *v, uint state)
{
    const bool upper = state & SPAN;
    for (uint q = 0; q < SPAN; ++q)
    {
        const A keep = upper ? v[q + SPAN] : v[q];
        const A give = upper ? v[q] : v[q + SPAN];
        v[q] = keep + simd_shuffle_xor(give, ushort(SPAN));
    }
    butterfly<SPAN / 2>(v, state);
}
template <> inline void butterfly<0>(thread A *, uint) {}

// The vector's state after a step: the matrix by the states, then x, which,
// added to a partial sum, could be rounded away before cancelling products
// meet. Every lane of the vector calls it.
template <bool ADD> inline A step(Column<ADD> column, A h, Lane l)
{
    A v[LANES];
    for (uint q = 0; q < LANES; ++q)
        v[q] = mul(column.m[q], h);
    butterfly<LANES / 2>(v, l.state);
    return v[0] + column.x;
}

// The vector's state through steps first to last - 1 of a sequence, whose
// matrices are m + t * stride_t and inputs x + t * M, added if ADD; if WRITE,
// the states are written to out + t * M. Pointers advance a step at a time,
// as 64-bit integer arithmetic is emulated.
template <bool ADD, bool WRITE>
inline A run_steps(const device T *m, long stride_t, const device T *x,
                   device T *out, uint first, uint last, A h, Lane l)
{
    const uint state = min(l.state, M - 1);
    m += first * stride_t + state;
    x += long(first) * M + state;
    if (WRITE)
        out += long(first) * M + state;
    for (uint t = first; t < last; ++t)
    {
        h = step(load_column<ADD>(m, x, l), h, l);
        m += stride_t;
        x += M;
        if (WRITE)
        {
            if (l.state < M)
                *out = T(h);
            out += M;
        }
    }
    return h;
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
    const uint state = min(l.state, M - 1);
    A h = A(0.0f);
    if (c > 0)
        h = starts[(b * (chunks - 1) + c - 1) * M + state];
    else if (from_zi)
        h = A(zi[b * M + state]);
    if (l.state >= M)
        h = A(0.0f);
    const long offset = long(b) * steps * M;
    run_steps<true, true>(a + b * stride_b, stride_t, x + offset, out + offset,
                          c * length, min(c * length + length, uint(steps)),
                          h, l);
}

// The maps of chunks 0 to chunks - 2, each of chunk steps. The first
// n_batch * ceil((chunks - 1) / run) * run vectors write o of chunk c of
// sequence b (see sequence_chunk) to offsets[b, c]: its end state from
// zero, or for c = 0 from zi[b] if from_zi. The vectors after write P, one
// per chunk of time-varying A and one for time-invariant A, all of whose
// chunks share it: vector (p * maps_per_a + j) * M + k after them writes
// column k of map j of A[p], a_batch of them, to maps[p, j], as the product
// of unit vector k by the chunk's matrices. Chunk 0's o holds zi, so its P
// is zero for time-varying A, and the scan of the maps starts from zero,
// which time-invariant A's P, shared by chunk 0, then leaves at zero.
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
    const Lane l = lane_of(index, lane);
    const uint n_batch = uint(batch), length = uint(chunk);
    const uint maps_per_item = (uint(steps) - 1) / length;
    const uint n_offsets =
        n_batch * ((maps_per_item - 1) / uint(run) + 1) * uint(run);
    if (l.vector < n_offsets)
    {
        uint b, c;
        if (!sequence_chunk(l.vector, n_batch, maps_per_item, uint(run), b, c))
            return;
        const A h = c == 0 && from_zi && l.state < M
            ? A(zi[b * M + min(l.state, M - 1)])
            : A(0.0f);
        const A end = run_steps<true, false>(
            a + b * stride_b, stride_t, x + long(b) * steps * M, nullptr,
            c * length, c * length + length, h, l);
        if (l.state < M)
            offsets[(b * maps_per_item + c) * M + l.state] = end;
        return;
    }
    const uint maps_per_a = stride_t == 0 ? 1 : maps_per_item;
    const uint column = l.vector - n_offsets;
    if (column >= uint(a_batch) * maps_per_a * M)
        return;
    const uint k = column % M, item = column / M;
    const uint p = item / maps_per_a, j = item % maps_per_a;
    // Time-invariant A's map is that of any chunk: chunk 0's steps.
    const bool zero = stride_t != 0 && j == 0;
    A h = l.state == k && !zero ? Acc<T>::one() : A(0.0f);
    if (!zero)
        h = run_steps<false, false>(a + p * stride_b, stride_t, x, nullptr,
                                    j * length, j * length + length, h, l);
    if (l.state < M)
        maps[(long(p * maps_per_a + j) * M + l.state) * M + k] = h;
}
