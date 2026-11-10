#pragma once

#include "recovered.cuh"

// Reconstructed radix-4 ablation, not an identified original paper kernel.
// Historical sources in rhrnald/fft commit
// 403c6ae8b152684d5b60bf04ebf067f02e97e026:
//   fft/fft_tc_sm.h: shared-stage radix-8 TF32 layout, per-transform padding 9,
//                   bit-group-reversed input and stage synchronization.
//   fft/my_fft.cu: fill_reg_b_branch and m16n8k8 TF32 MMA helpers, retained in
//                  recovered.cuh.
// This adapts the shared-stage layout to radix 4 and a single m16n8k8 MMA per
// butterfly. Unlike the register candidates, every stage reads and writes
// shared memory. A warp barrier before each butterfly's stores explicitly
// separates all operand reads from in-place writes; a second barrier publishes
// the complete stage before the next stage can read it. The historical radix-8
// source contained only the latter barrier.
//
// One warp transforms 16 independent N=64 complex inputs. Per FFT, issued
// Tensor Core work is 3 stages * 16 MMAs * (2*16*8*8) / 16 = 6144 FLOPs.
// This counts issued MMA arithmetic, not the conventional useful FFT count.
namespace thff_table7_legacy {

// Launch: grid(batch/16), block(32), dynamic shared bytes 0.
// Preconditions: batch is a positive multiple of 16; only threadIdx.x is used.
// In-place, forward, unnormalized, FP32 storage with historical TF32 MMA input
// handling. W_64 is unused by fill_reg_b_branch; nullptr is valid.
__global__ void fft_kernel_radix4_shared_batch16(
    cuFloatComplex* d_data, const cuFloatComplex* W_64, unsigned int repeat) {
    constexpr int length = 64;
    constexpr int transforms = 16;
    constexpr int warp_size = 32;
    constexpr int shared_stride = length + 9;
    __shared__ cuFloatComplex shared_data[transforms * shared_stride];

    const int lane = threadIdx.x;
    const int block_offset = blockIdx.x * transforms * length;

#pragma unroll
    for (int transform = 0; transform < transforms; ++transform) {
#pragma unroll
        for (int index = lane; index < length; index += warp_size) {
            shared_data[transform * shared_stride + index] =
                d_data[block_offset + transform * length +
                       reverse_2bit_groups<6>(index)];
        }
    }
    __syncwarp();

    // A's two fragment rows correspond to transforms lane/4 and lane/4+8.
    cuFloatComplex* batch0 = shared_data + (lane / 4) * shared_stride;
    cuFloatComplex* batch1 = batch0 + 8 * shared_stride;
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
                const int index = group + offset + stride * (lane % 4);
                const cuFloatComplex input0 = batch0[index];
                const cuFloatComplex input1 = batch1[index];
                const float a[4] = {input0.x, input1.x, input0.y, input1.y};
                float b[2];
                float d[4];

                // Natural shared stage layout needs no register permutation.
                // The helper folds both the DFT-4 coefficient and the stage
                // twiddle into B, as in the historical register arithmetic.
                fill_reg_b_branch<length>(b, 2 * stage, stride, 0, 0, offset,
                                          W_64);
                mma_m16n8k8_tf32_f32_rowcol(d, a, b, zero);
                __syncwarp();
                batch0[index] = make_cuFloatComplex(d[0], d[1]);
                batch1[index] = make_cuFloatComplex(d[2], d[3]);
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (int transform = 0; transform < transforms; ++transform) {
#pragma unroll
        for (int index = lane; index < length; index += warp_size) {
            d_data[block_offset + transform * length + index] =
                shared_data[transform * shared_stride + index];
        }
    }
}

}  // namespace thff_table7_legacy
