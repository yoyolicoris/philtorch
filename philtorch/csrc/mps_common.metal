// What the Metal kernels share, compiled at the head of each of their
// sources: scalar_recur_mps.mm and recur_mps.mm prepend it.
#include <metal_stdlib>
using namespace metal;

// The width of a SIMD group on every Apple GPU, which the kernels assume, as
// torch's Metal shim can't query it.
constant constexpr uint SIMD_WIDTH = 32;

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
    // float2(1.0f) would be 1 + i.
    static float2 one() { return float2(1.0f, 0.0f); }
};

inline float mul(float a, float b) { return a * b; }
inline float2 mul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
