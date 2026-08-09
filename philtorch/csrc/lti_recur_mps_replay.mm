#include <torch/extension.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <limits>

namespace
{

constexpr const char *METAL_SOURCE = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct AffinePair
{
    float multiplier;
    float value;
};

inline AffinePair compose_affine(AffinePair left, AffinePair right)
{
    AffinePair result;
    result.multiplier = left.multiplier * right.multiplier;
    result.value = left.value * right.multiplier + right.value;
    return result;
}

inline AffinePair affine_identity()
{
    AffinePair identity;
    identity.multiplier = 1.0f;
    identity.value = 0.0f;
    return identity;
}

inline AffinePair load_recurrence_pair(
    const device float *decays,
    const device float *initial,
    const device float *input,
    uint batch,
    uint step,
    uint sequence_length,
    uint batched_decay)
{
    AffinePair pair;
    if (step == 0)
    {
        pair.multiplier = 0.0f;
        pair.value = initial[batch];
    }
    else if (step <= sequence_length)
    {
        pair.multiplier = decays[batched_decay ? batch : 0];
        pair.value = input[batch * sequence_length + step - 1];
    }
    else
    {
        pair = affine_identity();
    }
    return pair;
}

inline AffinePair simd_inclusive_affine(AffinePair value, uint lane)
{
    for (uint offset = 1; offset < 32; offset <<= 1)
    {
        AffinePair left;
        left.multiplier = simd_shuffle_up(value.multiplier, offset);
        left.value = simd_shuffle_up(value.value, offset);
        if (lane >= offset)
        {
            value = compose_affine(left, value);
        }
    }
    return value;
}

kernel void replay_reduce_tiles(
    const device float *decays [[buffer(0)]],
    const device float *initial [[buffer(1)]],
    const device float *input [[buffer(2)]],
    device AffinePair *block_totals [[buffer(3)]],
    constant uint &sequence_length [[buffer(4)]],
    constant uint &n_tiles [[buffer(5)]],
    constant uint &batched_decay [[buffer(6)]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    constexpr uint tile_size = 512;
    constexpr uint simd_groups = 8;
    threadgroup AffinePair group_totals[simd_groups];

    const uint tile = group.x;
    const uint batch = group.y;
    const uint first_step = tile * tile_size + 2 * lid;
    const uint second_step = first_step + 1;
    const AffinePair first = load_recurrence_pair(
        decays, initial, input, batch, first_step,
        sequence_length, batched_decay);
    const AffinePair second = load_recurrence_pair(
        decays, initial, input, batch, second_step,
        sequence_length, batched_decay);
    const AffinePair thread_total = compose_affine(first, second);
    const AffinePair group_inclusive =
        simd_inclusive_affine(thread_total, lane);

    if (lane == 31)
    {
        group_totals[simd_group] = group_inclusive;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0)
    {
        const AffinePair group_value =
            lane < simd_groups ? group_totals[lane] : affine_identity();
        const AffinePair groups_inclusive =
            simd_inclusive_affine(group_value, lane);
        if (lane == simd_groups - 1)
        {
            block_totals[batch * n_tiles + tile] = groups_inclusive;
        }
    }
}

kernel void replay_scan_block_totals(
    const device AffinePair *block_totals [[buffer(0)]],
    device AffinePair *block_prefixes [[buffer(1)]],
    constant uint &n_tiles [[buffer(2)]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    constexpr uint tile_size = 512;
    constexpr uint simd_groups = 8;
    threadgroup AffinePair group_totals[simd_groups];
    threadgroup AffinePair group_prefixes[simd_groups];
    threadgroup AffinePair carry;

    const uint batch = group.x;
    const uint n_chunks = (n_tiles - 1) / tile_size + 1;
    if (lid == 0)
    {
        carry = affine_identity();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint chunk = 0; chunk < n_chunks; ++chunk)
    {
        const uint first_tile = chunk * tile_size + 2 * lid;
        const uint second_tile = first_tile + 1;
        const AffinePair first = first_tile < n_tiles
            ? block_totals[batch * n_tiles + first_tile]
            : affine_identity();
        const AffinePair second = second_tile < n_tiles
            ? block_totals[batch * n_tiles + second_tile]
            : affine_identity();
        const AffinePair thread_total = compose_affine(first, second);
        const AffinePair group_inclusive =
            simd_inclusive_affine(thread_total, lane);

        if (lane == 31)
        {
            group_totals[simd_group] = group_inclusive;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0)
        {
            const AffinePair group_value =
                lane < simd_groups ? group_totals[lane] : affine_identity();
            const AffinePair groups_inclusive =
                simd_inclusive_affine(group_value, lane);
            AffinePair prior_group;
            prior_group.multiplier = simd_shuffle_up(
                groups_inclusive.multiplier, 1);
            prior_group.value = simd_shuffle_up(groups_inclusive.value, 1);
            if (lane < simd_groups)
            {
                group_prefixes[lane] =
                    lane == 0 ? affine_identity() : prior_group;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        AffinePair prior_thread;
        prior_thread.multiplier = simd_shuffle_up(
            group_inclusive.multiplier, 1);
        prior_thread.value = simd_shuffle_up(group_inclusive.value, 1);
        const AffinePair local_prefix = compose_affine(
            group_prefixes[simd_group],
            lane == 0 ? affine_identity() : prior_thread);
        const AffinePair chunk_prefix = carry;
        const AffinePair first_prefix =
            compose_affine(chunk_prefix, local_prefix);
        const AffinePair first_inclusive = compose_affine(first_prefix, first);
        const AffinePair second_inclusive =
            compose_affine(first_inclusive, second);

        if (first_tile < n_tiles)
        {
            block_prefixes[batch * n_tiles + first_tile] = first_prefix;
        }
        if (second_tile < n_tiles)
        {
            block_prefixes[batch * n_tiles + second_tile] = first_inclusive;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 255)
        {
            carry = second_inclusive;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void replay_scan_tiles(
    const device float *decays [[buffer(0)]],
    const device float *initial [[buffer(1)]],
    const device float *input [[buffer(2)]],
    const device AffinePair *block_prefixes [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &sequence_length [[buffer(5)]],
    constant uint &n_tiles [[buffer(6)]],
    constant uint &batched_decay [[buffer(7)]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    constexpr uint tile_size = 512;
    constexpr uint simd_groups = 8;
    threadgroup AffinePair group_totals[simd_groups];
    threadgroup AffinePair group_prefixes[simd_groups];
    threadgroup AffinePair tile_prefix;

    const uint tile = group.x;
    const uint batch = group.y;
    const uint first_step = tile * tile_size + 2 * lid;
    const uint second_step = first_step + 1;
    const AffinePair first = load_recurrence_pair(
        decays, initial, input, batch, first_step,
        sequence_length, batched_decay);
    const AffinePair second = load_recurrence_pair(
        decays, initial, input, batch, second_step,
        sequence_length, batched_decay);
    const AffinePair thread_total = compose_affine(first, second);
    const AffinePair group_inclusive =
        simd_inclusive_affine(thread_total, lane);

    if (lane == 31)
    {
        group_totals[simd_group] = group_inclusive;
    }
    if (lid == 0)
    {
        tile_prefix = block_prefixes[batch * n_tiles + tile];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0)
    {
        const AffinePair group_value =
            lane < simd_groups ? group_totals[lane] : affine_identity();
        const AffinePair groups_inclusive =
            simd_inclusive_affine(group_value, lane);
        AffinePair prior_group;
        prior_group.multiplier = simd_shuffle_up(
            groups_inclusive.multiplier, 1);
        prior_group.value = simd_shuffle_up(groups_inclusive.value, 1);
        if (lane < simd_groups)
        {
            group_prefixes[lane] =
                lane == 0 ? affine_identity() : prior_group;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    AffinePair prior_thread;
    prior_thread.multiplier = simd_shuffle_up(
        group_inclusive.multiplier, 1);
    prior_thread.value = simd_shuffle_up(group_inclusive.value, 1);
    const AffinePair thread_prefix = compose_affine(
        group_prefixes[simd_group],
        lane == 0 ? affine_identity() : prior_thread);
    const AffinePair first_inclusive = compose_affine(thread_prefix, first);
    const AffinePair second_inclusive = compose_affine(first_inclusive, second);

    if (first_step > 0 && first_step <= sequence_length)
    {
        output[batch * sequence_length + first_step - 1] =
            compose_affine(tile_prefix, first_inclusive).value;
    }
    if (second_step <= sequence_length)
    {
        output[batch * sequence_length + second_step - 1] =
            compose_affine(tile_prefix, second_inclusive).value;
    }
}
)METAL";

struct MetalPipelines
{
    id<MTLComputePipelineState> reduce_tiles;
    id<MTLComputePipelineState> scan_block_totals;
    id<MTLComputePipelineState> scan_tiles;
};

id<MTLBuffer> get_mtl_buffer(const at::Tensor &tensor)
{
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

NSUInteger get_buffer_offset(const at::Tensor &tensor)
{
    return static_cast<NSUInteger>(tensor.storage_offset() * tensor.element_size());
}

MetalPipelines &get_pipelines()
{
    static MetalPipelines pipelines;
    static dispatch_once_t once_token;
    dispatch_once(&once_token, ^{
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      TORCH_CHECK(device, "Failed to find a Metal device");

      NSError *error = nil;
      NSString *source = [NSString stringWithUTF8String:METAL_SOURCE];
      id<MTLLibrary> library = [device newLibraryWithSource:source
                                                    options:nil
                                                      error:&error];
      TORCH_CHECK(library, "Failed to compile scalar recurrence Metal kernels: ",
                  error.localizedDescription.UTF8String);

      id<MTLFunction> reduce_tiles_function =
          [library newFunctionWithName:@"replay_reduce_tiles"];
      id<MTLFunction> scan_block_totals_function =
          [library newFunctionWithName:@"replay_scan_block_totals"];
      id<MTLFunction> scan_tiles_function =
          [library newFunctionWithName:@"replay_scan_tiles"];
      TORCH_CHECK(reduce_tiles_function && scan_block_totals_function &&
                      scan_tiles_function,
                  "Failed to load scalar recurrence Metal functions");

      pipelines.reduce_tiles =
          [device newComputePipelineStateWithFunction:reduce_tiles_function
                                                error:&error];
      TORCH_CHECK(pipelines.reduce_tiles,
                  "Failed to create tile reduction pipeline: ",
                  error.localizedDescription.UTF8String);
      pipelines.scan_block_totals =
          [device newComputePipelineStateWithFunction:scan_block_totals_function
                                                error:&error];
      TORCH_CHECK(pipelines.scan_block_totals,
                  "Failed to create block-total scan pipeline: ",
                  error.localizedDescription.UTF8String);
      pipelines.scan_tiles =
          [device newComputePipelineStateWithFunction:scan_tiles_function
                                                error:&error];
      TORCH_CHECK(pipelines.scan_tiles,
                  "Failed to create tile replay pipeline: ",
                  error.localizedDescription.UTF8String);
    });
    return pipelines;
}

at::Tensor lti_recur_mps_impl(const at::Tensor &a, const at::Tensor &zi,
                              const at::Tensor &x)
{
    TORCH_CHECK(a.device().is_mps() && zi.device().is_mps() && x.device().is_mps(),
                "a, zi, and x must be MPS tensors");
    TORCH_CHECK(a.scalar_type() == x.scalar_type() &&
                    zi.scalar_type() == x.scalar_type(),
                "a, zi, and x must have the same scalar type");
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "MPS scalar recurrence currently supports float32 only");
    TORCH_CHECK(a.dim() <= 1, "a must be a vector or a scalar");
    TORCH_CHECK(zi.dim() == 1, "zi must be a vector");
    TORCH_CHECK(x.dim() == 2, "x must be a matrix");
    TORCH_CHECK(zi.size(0) == x.size(0),
                "zi and x must have the same batch size");
    TORCH_CHECK(a.numel() == 1 || a.numel() == x.size(0),
                "a must contain one value or one value per batch");
    TORCH_CHECK(x.size(1) > 0, "x must contain at least one time step");

    const int64_t n_batches_64 = x.size(0);
    const int64_t sequence_length_64 = x.size(1);
    if (n_batches_64 == 0)
    {
        return at::empty_like(x);
    }

    constexpr uint64_t tile_size = 512;
    const uint64_t n_steps_wide = static_cast<uint64_t>(sequence_length_64) + 1;
    const uint64_t n_tiles_wide = (n_steps_wide - 1) / tile_size + 1;
    TORCH_CHECK(n_steps_wide <= std::numeric_limits<uint32_t>::max() &&
                    n_tiles_wide <= std::numeric_limits<uint32_t>::max() &&
                    static_cast<uint64_t>(n_batches_64) * n_tiles_wide <=
                        std::numeric_limits<uint32_t>::max() &&
                    static_cast<uint64_t>(n_batches_64) * sequence_length_64 <=
                        std::numeric_limits<uint32_t>::max(),
                "MPS scalar recurrence input is too large");

    const uint32_t n_batches = static_cast<uint32_t>(n_batches_64);
    const uint32_t n_tiles = static_cast<uint32_t>(n_tiles_wide);
    const uint32_t sequence_length = static_cast<uint32_t>(sequence_length_64);
    const uint32_t batched_decay = a.numel() == n_batches_64 ? 1 : 0;
    const auto a_contiguous = a.contiguous();
    const auto zi_contiguous = zi.contiguous();
    const auto x_contiguous = x.contiguous();
    auto block_totals = at::empty(
        {n_batches_64, static_cast<int64_t>(n_tiles), 2}, x.options());
    auto block_prefixes = at::empty_like(block_totals);
    auto output = at::empty_like(x_contiguous);
    auto &pipelines = get_pipelines();
    constexpr NSUInteger threads_per_group = 256;
    TORCH_CHECK(pipelines.reduce_tiles.threadExecutionWidth == 32 &&
                    pipelines.scan_block_totals.threadExecutionWidth == 32 &&
                    pipelines.scan_tiles.threadExecutionWidth == 32,
                "MPS scalar recurrence requires 32-lane SIMD groups");
    TORCH_CHECK(
        pipelines.reduce_tiles.maxTotalThreadsPerThreadgroup >= threads_per_group &&
            pipelines.scan_block_totals.maxTotalThreadsPerThreadgroup >=
                threads_per_group &&
            pipelines.scan_tiles.maxTotalThreadsPerThreadgroup >= threads_per_group,
        "Metal device does not support the scalar recurrence threadgroup size");

    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer, "Failed to retrieve the MPS command buffer");
    dispatch_queue_t queue = torch::mps::get_dispatch_queue();

    dispatch_sync(queue, ^{
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create the tile reduction encoder");
      [encoder setComputePipelineState:pipelines.reduce_tiles];
      [encoder setBuffer:get_mtl_buffer(a_contiguous)
                   offset:get_buffer_offset(a_contiguous)
                  atIndex:0];
      [encoder setBuffer:get_mtl_buffer(zi_contiguous)
                   offset:get_buffer_offset(zi_contiguous)
                  atIndex:1];
      [encoder setBuffer:get_mtl_buffer(x_contiguous)
                   offset:get_buffer_offset(x_contiguous)
                  atIndex:2];
      [encoder setBuffer:get_mtl_buffer(block_totals)
                   offset:get_buffer_offset(block_totals)
                  atIndex:3];
      [encoder setBytes:&sequence_length
                  length:sizeof(sequence_length)
                 atIndex:4];
      [encoder setBytes:&n_tiles length:sizeof(n_tiles) atIndex:5];
      [encoder setBytes:&batched_decay
                  length:sizeof(batched_decay)
                 atIndex:6];
      [encoder dispatchThreadgroups:MTLSizeMake(n_tiles, n_batches, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      [encoder endEncoding];

      encoder = [command_buffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create the block-total scan encoder");
      [encoder setComputePipelineState:pipelines.scan_block_totals];
      [encoder setBuffer:get_mtl_buffer(block_totals)
                   offset:get_buffer_offset(block_totals)
                  atIndex:0];
      [encoder setBuffer:get_mtl_buffer(block_prefixes)
                   offset:get_buffer_offset(block_prefixes)
                  atIndex:1];
      [encoder setBytes:&n_tiles length:sizeof(n_tiles) atIndex:2];
      [encoder dispatchThreadgroups:MTLSizeMake(n_batches, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      [encoder endEncoding];

      encoder = [command_buffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create the tile replay encoder");
      [encoder setComputePipelineState:pipelines.scan_tiles];
      [encoder setBuffer:get_mtl_buffer(a_contiguous)
                   offset:get_buffer_offset(a_contiguous)
                  atIndex:0];
      [encoder setBuffer:get_mtl_buffer(zi_contiguous)
                   offset:get_buffer_offset(zi_contiguous)
                  atIndex:1];
      [encoder setBuffer:get_mtl_buffer(x_contiguous)
                   offset:get_buffer_offset(x_contiguous)
                  atIndex:2];
      [encoder setBuffer:get_mtl_buffer(block_prefixes)
                   offset:get_buffer_offset(block_prefixes)
                  atIndex:3];
      [encoder setBuffer:get_mtl_buffer(output)
                   offset:get_buffer_offset(output)
                  atIndex:4];
      [encoder setBytes:&sequence_length
                  length:sizeof(sequence_length)
                 atIndex:5];
      [encoder setBytes:&n_tiles length:sizeof(n_tiles) atIndex:6];
      [encoder setBytes:&batched_decay
                  length:sizeof(batched_decay)
                 atIndex:7];
      [encoder dispatchThreadgroups:MTLSizeMake(n_tiles, n_batches, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
      [encoder endEncoding];
      torch::mps::commit();
    });

    return output;
}

} // namespace

TORCH_LIBRARY_IMPL(philtorch, MPS, m)
{
    m.impl("lti_recur", &lti_recur_mps_impl);
}