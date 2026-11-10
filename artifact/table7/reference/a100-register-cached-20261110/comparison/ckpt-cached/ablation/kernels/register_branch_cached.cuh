#pragma once

#include "historical_branch_51e346c.cuh"

// Derived from the complete 51e346c register FFT, not an original paper source.
// Preserve its data mapping, lane-dependent stage permutations and all 48 MMAs.
// Prepare real B coefficients once before the repetition loop, matching the
// timing boundary of shared_stage_radix4. The (t_2R-t_R)/R protocol excludes
// that preparation and the initial/final shared-memory IO. Every repeated
// transform keeps its data in registers; no stage communication is omitted.
// R=1 is a forward, unnormalized N=64 FFT. R>1 is a throughput workload and
// may overflow, just like the other Table 7 candidates.
namespace thff_table7_register_cached {

template <typename T>
__device__ void permute_radix4_branch(T &a, T &b, T &c, T &d, int pattern) {
    if (pattern == 1 || pattern == 3) {
        T tmp = a;
        a = b;
        b = tmp;

        tmp = c;
        c = d;
        d = tmp;
    }
    if (pattern == 0 || pattern == 3) {
        T tmp = b;
        b = d;
        d = tmp;
    }
    if (pattern == 2 || pattern == 3) {
        T tmp = a;
        a = c;
        c = tmp;
    }
}


__global__ void fft_kernel_radix64_batch16(
    cuFloatComplex* d_data, const cuFloatComplex* __restrict__ W_64,
    unsigned int repeat) {
    constexpr int length = 64;
    constexpr int transforms = 16;
    constexpr int warp_size = 32;
    constexpr int ept = length * transforms / warp_size;
    const int lane = threadIdx.x;
    const int block_offset = blockIdx.x * length * transforms;
    cuFloatComplex reg[ept];
    __shared__ cuFloatComplex shared_data[ept * (warp_size + 1)];

#pragma unroll
    for (int i = 0; i < ept; ++i) {
        shared_data[i * (warp_size + 1) + lane] =
            d_data[block_offset + i * warp_size + lane];
    }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        const int column = thff_table7_51e346c::reverse_2bit_groups<4>(i);
        reg[i] = shared_data[(lane / 2) * (warp_size + 1) +
                             column + (ept / 2) * (lane % 2)];
        reg[i + ept / 2] = shared_data[
            (ept / 2) * (warp_size + 1) + (lane / 2) * (warp_size + 1) +
            column + (ept / 2) * (lane % 2)];
    }

    // The branch layout has 28 distinct coefficient pairs. The four stage-0
    // pairs alias stage-1 butterflies 0,4,8,12, and stage-2 butterflies 0,4,8,12
    // alias stage-1 butterflies 0,1,2,3. These are exact LUT-index identities,
    // including the lane-dependent real/imaginary component convention.
    float coeff[28][2];
#pragma unroll
    for (int stage = 1; stage < 3; ++stage) {
        const int stride = 1 << (2 * stage);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (stage == 1 || (j & 3) != 0) {
                const int slot = stage == 1 ? j : 16 + 3 * (j / 4) + (j & 3) - 1;
                const int i_perm = (j / stride) & 3;
                const int j_perm = (j / (stride / 4)) & 3;
                thff_table7_51e346c::fill_reg_b<length>(
                    coeff[slot], 2 * stage, stride, i_perm, j_perm,
                    j % stride, W_64);
                asm volatile("" : "+f"(coeff[slot][0]), "+f"(coeff[slot][1]));
            }
        }
    }
    const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
    for (unsigned int repetition = 0; repetition < repeat; ++repetition) {
#pragma unroll
        for (int stage = 0; stage < 3; ++stage) {
            const int stride = 1 << (2 * stage);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float a[4] = {reg[j].x, reg[j + 16].x,
                                    reg[j].y, reg[j + 16].y};
                const int slot = stage == 0 ? 4 * (j & 3) :
                    stage == 1 ? j :
                    (j & 3) == 0 ? j / 4 : 16 + 3 * (j / 4) + (j & 3) - 1;
                float result[4];
                thff_table7_51e346c::mma_m16n8k8_tf32_f32_rowcol(
                    result, a, coeff[slot], zero);
                reg[j].x = result[0];
                reg[j].y = result[1];
                reg[j + 16].x = result[2];
                reg[j + 16].y = result[3];
            }
            if (stage < 2) {
#pragma unroll
                for (int j = 0; j < ept; j += 4 * stride) {
#pragma unroll
                    for (int k = 0; k < stride; ++k) {
                        permute_radix4_branch(
                            reg[j + k], reg[j + k + stride],
                            reg[j + k + 2 * stride], reg[j + k + 3 * stride],
                            lane & 3);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        shared_data[(warp_size + 1) * (lane / 2) +
                    16 * (lane % 2) + i] = reg[i];
        shared_data[(ept / 2) * (warp_size + 1) +
                    (warp_size + 1) * (lane / 2) +
                    16 * (lane % 2) + i] = reg[i + ept / 2];
    }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < ept; ++i) {
        d_data[block_offset + lane + i * warp_size] =
            shared_data[i * (warp_size + 1) + lane];
    }
}

}  // namespace thff_table7_register_cached
