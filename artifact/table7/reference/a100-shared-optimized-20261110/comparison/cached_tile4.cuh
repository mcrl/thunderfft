#pragma once

#include "historical_uniform_5e37821.cuh"

// Fresh reconstruction of the shared-memory radix-4 ablation, not a recovered
// original paper implementation. Reuses the uniform register kernel's W64 LUT
// and TF32 MMA helpers, but materializes every butterfly result in shared
// memory. No register permutation is performed between radix-4 stages.
//
// One warp handles 16 independent N=64 transforms. Natural radix-4 DIT indices
// select shared operands, with base-4 digit reversal only on the initial input
// load. The uniform helper's MMA component order is r0,r1,r2,r3,i0,i1,i2,i3:
// lanes q=0,1 load real components, and q=2,3 load imaginary components; each
// lane owns butterfly indices 2*(q&1) and 2*(q&1)+1. The same mapping stores D.
//
// A single shared buffer holds the stage data. Each butterfly waits after its
// operand loads/MMA and before its stores to prevent cross-lane write-after-read
// hazards. Distinct butterflies within one stage touch disjoint elements, so
// they need no extra store-to-next-butterfly fence. The end-of-stage barrier
// publishes all results for the next stage and for the final global stores.
// A fixed bijective XOR bank layout stores exactly 16*64 complex values (8 KiB).
// Adjacent words pair transforms t and t+8, enabling float2 butterfly ld/st.
// Each 16-lane LDS.64/STS.64 transaction uses all 32 banks once at every stage.
// Only physical shared addresses are swizzled: logical FFT indices and the MMA
// operand/result register order remain unchanged; no inter-stage register shuffle.
// Issued arithmetic remains 3*16 MMAs per warp, or 6,144 Tensor Core FLOPs/FFT.
namespace thff_table7_shared_stage {

__device__ __forceinline__ int shared_address(int transform, int index,
                                            int component) {
    // Transform bit 3 chooses an adjacent word within a float2. For each
    // 16-lane vector instruction transaction, the bank's upper four bits are
    // a bijection of the lane bits: transform bits 0,1, component, and parity.
    // Transform bit 2 and the remaining index bits select the shared row.
    // The omitted index bit 1 is recovered from parity and bits 3,5.
    const int parity = ((index >> 1) ^ (index >> 3) ^ (index >> 5)) & 1;
    const int packed_index = (index & 1) | ((index >> 2) << 1);
    return (packed_index * 2 + ((transform >> 2) & 1)) * 32 +
           (transform & 3) * 8 + component * 4 + parity * 2 + (transform >> 3);
}

// Launch grid(batch/16), block(32), dynamic shared bytes 0.
// Batch must be a positive multiple of 16; W_64 must contain 64 complex roots
// of unity using the same forward sign convention as the uniform candidates.
// Forward, unnormalized, in-place, FP32 storage; historical TF32 input handling.
// Correctness is defined for repeat=1. As in the other ablation candidates,
// repeats>1 measure sustained stage arithmetic and may produce nonfinite data.
__global__ void fft_kernel_radix4_shared_stage_batch16(
    cuFloatComplex* d_data, const cuFloatComplex* __restrict__ W_64,
    unsigned int repeat) {
    constexpr int length = 64;
    constexpr int transforms = 16;
    constexpr int warp_size = 32;
    __shared__ __align__(16) float shared_data[transforms * length * 2];

    const int lane = threadIdx.x & 31;
    const int block_offset = blockIdx.x * transforms * length;

#pragma unroll
    for (int transform = 0; transform < transforms; ++transform) {
#pragma unroll
        for (int tile = 0; tile < length / warp_size; ++tile) {
            const int index = lane + tile * warp_size;
            const cuFloatComplex value = d_data[
                block_offset + transform * length +
                thff_table7_5e37821::reverse_2bit_groups<6>(index)];
            shared_data[shared_address(transform, index, 0)] = value.x;
            shared_data[shared_address(transform, index, 1)] = value.y;
        }
    }
    __syncwarp();

    const int transform0 = lane / 4;
    const int component = (lane % 4) / 2;
    const int radix_index0 = 2 * (lane & 1);
    const int radix_index1 = radix_index0 + 1;
    const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Experimental coefficient cache: 1 + 4 + 16 unique pairs. Unlike the
    // ordinary ablation, LUT reads occur once per launch rather than once per
    // repeated stage. Shared data traffic and all 48 MMAs remain inside R.
    float coeff[21][2];
#pragma unroll
    for (int stage = 0; stage < 3; ++stage) {
        const int stride = 1 << (2 * stage);
        const int base = (stride - 1) / 3;
#pragma unroll
        for (int offset = 0; offset < stride; ++offset) {
            thff_table7_5e37821::fill_reg_b<length>(
                coeff[base + offset], 2 * stage, stride, 0, 0, offset, W_64);
            // Keep each pair as an explicit value at the repetition boundary.
            asm volatile("" : "+f"(coeff[base + offset][0]),
                              "+f"(coeff[base + offset][1]));
        }
    }

#pragma unroll 1
    for (unsigned int repetition = 0; repetition < repeat; ++repetition) {
#pragma unroll
        for (int stage = 0; stage < 3; ++stage) {
            const int stride = 1 << (2 * stage);
#ifndef THFF_SHARED_TILE
#define THFF_SHARED_TILE 4
#endif
            static_assert(16 % THFF_SHARED_TILE == 0);
#pragma unroll
            for (int first = 0; first < length / 4; first += THFF_SHARED_TILE) {
                float result[THFF_SHARED_TILE][4];
#pragma unroll
                for (int tile = 0; tile < THFF_SHARED_TILE; ++tile) {
                    const int butterfly = first + tile;
                    const int group = (butterfly / stride) * (4 * stride);
                    const int offset = butterfly % stride;
                    const int index0 = group + offset + stride * radix_index0;
                    const int index1 = group + offset + stride * radix_index1;
                    const int address0 = shared_address(transform0, index0, component);
                    const int address1 = shared_address(transform0, index1, component);
                    const float2 pair0 = *reinterpret_cast<const float2*>(&shared_data[address0]);
                    const float2 pair1 = *reinterpret_cast<const float2*>(&shared_data[address1]);
                    const float a[4] = {pair0.x, pair0.y, pair1.x, pair1.y};
                    const int coefficient_base = (stride - 1) / 3;
                    const float b[2] = {coeff[coefficient_base + offset][0],
                                        coeff[coefficient_base + offset][1]};
                    thff_table7_5e37821::mma_m16n8k8_tf32_f32_rowcol(
                        result[tile], a, b, zero);
                }
                // A tile consists of disjoint butterflies. All its cross-lane
                // reads complete before any in-place writes, then the stage
                // barrier publishes every tile to the next stage.
                __syncwarp();
#pragma unroll
                for (int tile = 0; tile < THFF_SHARED_TILE; ++tile) {
                    const int butterfly = first + tile;
                    const int group = (butterfly / stride) * (4 * stride);
                    const int offset = butterfly % stride;
                    const int index0 = group + offset + stride * radix_index0;
                    const int index1 = group + offset + stride * radix_index1;
                    const int address0 = shared_address(transform0, index0, component);
                    const int address1 = shared_address(transform0, index1, component);
                    *reinterpret_cast<float2*>(&shared_data[address0]) =
                        make_float2(result[tile][0], result[tile][2]);
                    *reinterpret_cast<float2*>(&shared_data[address1]) =
                        make_float2(result[tile][1], result[tile][3]);
                }
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (int transform = 0; transform < transforms; ++transform) {
#pragma unroll
        for (int tile = 0; tile < length / warp_size; ++tile) {
            const int index = lane + tile * warp_size;
            d_data[block_offset + transform * length + index] =
                make_cuFloatComplex(
                    shared_data[shared_address(transform, index, 0)],
                    shared_data[shared_address(transform, index, 1)]);
        }
    }
}

}  // namespace thff_table7_shared_stage
