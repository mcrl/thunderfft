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
// A nine-complex-value padding matches the older shared reconstruction's row
// spacing. Static shared storage is 16*(64+9)*sizeof(cuFloatComplex) = 9,344 B.
// Issued arithmetic remains 3*16 MMAs per warp, or 6,144 Tensor Core FLOPs/FFT.
namespace thff_table7_shared_stage {

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
    constexpr int shared_stride = 2 * (length + 9);
    __shared__ float shared_data[transforms * shared_stride];

    const int lane = threadIdx.x;
    const int block_offset = blockIdx.x * transforms * length;

#pragma unroll
    for (int transform = 0; transform < transforms; ++transform) {
#pragma unroll
        for (int index = lane; index < length; index += warp_size) {
            const cuFloatComplex value = d_data[
                block_offset + transform * length +
                thff_table7_5e37821::reverse_2bit_groups<6>(index)];
            shared_data[transform * shared_stride + 2 * index] = value.x;
            shared_data[transform * shared_stride + 2 * index + 1] = value.y;
        }
    }
    __syncwarp();

    const int transform0 = (lane / 4) * shared_stride;
    const int transform1 = transform0 + 8 * shared_stride;
    const int component = (lane % 4) / 2;
    const int radix_index0 = 2 * (lane & 1);
    const int radix_index1 = radix_index0 + 1;
    const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
    for (unsigned int repetition = 0; repetition < repeat; ++repetition) {
#pragma unroll
        for (int stage = 0; stage < 3; ++stage) {
            const int stride = 1 << (2 * stage);
#pragma unroll
            for (int butterfly = 0; butterfly < length / 4; ++butterfly) {
                const int group = (butterfly / stride) * (4 * stride);
                const int offset = butterfly % stride;
                const int index0 =
                    2 * (group + offset + stride * radix_index0) + component;
                const int index1 =
                    2 * (group + offset + stride * radix_index1) + component;
                const float a[4] = {
                    shared_data[transform0 + index0],
                    shared_data[transform1 + index0],
                    shared_data[transform0 + index1],
                    shared_data[transform1 + index1],
                };
                float b[2];
                float d[4];
                thff_table7_5e37821::fill_reg_b<length>(
                    b, 2 * stage, stride, 0, 0, offset, W_64);
                thff_table7_5e37821::mma_m16n8k8_tf32_f32_rowcol(d, a, b, zero);
                __syncwarp();
                shared_data[transform0 + index0] = d[0];
                shared_data[transform0 + index1] = d[1];
                shared_data[transform1 + index0] = d[2];
                shared_data[transform1 + index1] = d[3];
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (int transform = 0; transform < transforms; ++transform) {
#pragma unroll
        for (int index = lane; index < length; index += warp_size) {
            d_data[block_offset + transform * length + index] =
                make_cuFloatComplex(
                    shared_data[transform * shared_stride + 2 * index],
                    shared_data[transform * shared_stride + 2 * index + 1]);
        }
    }
}

}  // namespace thff_table7_shared_stage
