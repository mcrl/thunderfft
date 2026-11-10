#pragma once

#include "recovered.cuh"

// Reconstructed N=64 mixed-radix (16 x 4) shared-memory baseline.
// This is newly written artifact code, not a recovered tcFFT implementation or
// a certified identity of the paper's radix-16 ablation row. The original
// project's W() and TF32 MMA helper are reused from recovered.cuh; the dense
// complex embedding, mixed-radix mapping and stage driver below are new.
//
// One warp handles 16 independent forward, unnormalized transforms. Launch
// batch / 16 blocks with exactly 32 threads; batch must be divisible by 16.
// Static shared memory is 16 * (64 + 9) complex floats = 9344 bytes per block.
// The W_64 argument is retained for compatibility and is not dereferenced.
namespace thff_table7_legacy {

static constexpr int kRadix16SharedStride = 64 + 9;
static constexpr int kRadix16SharedElements = 16 * kRadix16SharedStride;
static constexpr int kRadix16MmasPerWarp = 80;
static constexpr int kRadix16MmaFlopsPerFft =
    kRadix16MmasPerWarp * (2 * 16 * 8 * 8) / 16; // 10240, not 32*N*log2(N).

namespace radix16_reconstruction {

// For input complex index q and output p, the stage coefficient is
// exp(-2*pi*i*q*(j + Stride*p)/(Radix*Stride)). Expand it into a real
// matrix with interleaved real/imaginary rows and columns:
//                  output real   output imaginary
//   input real         cos              sin
//   input imaginary   -sin              cos
// The quarter-turn term implements all four entries with the historical W().
template <int Radix, int Stride>
__device__ __forceinline__ float coefficient(int real_row, int real_col,
                                             int j) {
    const int q = real_row / 2;
    const int p = real_col / 2;
    const int quarter_turn = (real_col % 2 - real_row % 2) *
                             ((Radix * Stride) / 4);
    return W(q * (j + Stride * p) + quarter_turn, Radix * Stride).x;
}

// Batched complex DFT, represented by A[16, 2*Radix] * B[2*Radix, 2*Radix].
// MMA m16n8k8 fragment coordinates for lane = 4*g + t:
//   A = {(g,k+t), (g+8,k+t), (g,k+t+4), (g+8,k+t+4)}
//   B = {(k+t,n+g), (k+t+4,n+g)}
//   D = {(g,n+2*t), (g,n+2*t+1), (g+8,n+2*t), (g+8,n+2*t+1)}.
// Each paired D entry is one output complex value. All source A fragments are
// loaded before any output writes; the barrier prevents an early lane from
// overwriting shared input still needed by another lane. Subsequent output
// tiles read those saved fragments, never the overwritten shared memory.
template <int Radix, int Stride>
__device__ __forceinline__ void stage(cuFloatComplex* shared, int base, int j) {
    static_assert(Radix == 16 || Radix == 4, "supported mixed-radix stages");
    constexpr int tiles = (2 * Radix) / 8;
    const int row = threadIdx.x / 4;
    const int lane_col = threadIdx.x % 4;
    float input[tiles][4];

#pragma unroll
    for (int kt = 0; kt < tiles; ++kt) {
        const int k0 = kt * 8 + lane_col;
        const int k1 = k0 + 4;
        const cuFloatComplex a0 =
            shared[row * kRadix16SharedStride + base + (k0 / 2) * Stride];
        const cuFloatComplex a1 =
            shared[(row + 8) * kRadix16SharedStride + base + (k0 / 2) * Stride];
        const cuFloatComplex a2 =
            shared[row * kRadix16SharedStride + base + (k1 / 2) * Stride];
        const cuFloatComplex a3 =
            shared[(row + 8) * kRadix16SharedStride + base + (k1 / 2) * Stride];
        input[kt][0] = (k0 & 1) ? a0.y : a0.x;
        input[kt][1] = (k0 & 1) ? a1.y : a1.x;
        input[kt][2] = (k1 & 1) ? a2.y : a2.x;
        input[kt][3] = (k1 & 1) ? a3.y : a3.x;
    }
    __syncwarp();

#pragma unroll
    for (int nt = 0; nt < tiles; ++nt) {
        float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int kt = 0; kt < tiles; ++kt) {
            const int k = kt * 8 + lane_col;
            const int n = nt * 8 + row;
            const float weights[2] = {
                coefficient<Radix, Stride>(k, n, j),
                coefficient<Radix, Stride>(k + 4, n, j)};
            mma_m16n8k8_tf32_f32_rowcol(accumulator, input[kt], weights,
                                       accumulator);
        }
        const int destination = base + (nt * 4 + lane_col) * Stride;
        shared[row * kRadix16SharedStride + destination] =
            make_cuFloatComplex(accumulator[0], accumulator[1]);
        shared[(row + 8) * kRadix16SharedStride + destination] =
            make_cuFloatComplex(accumulator[2], accumulator[3]);
    }
}

} // namespace radix16_reconstruction

__global__ void fft_kernel_radix16_batch16(
    cuFloatComplex* data, const cuFloatComplex* __restrict__ W_64,
    unsigned int inside_repeats) {
    (void)W_64;
    __shared__ cuFloatComplex shared[kRadix16SharedElements];
    const int lane = threadIdx.x;
    const unsigned int offset = blockIdx.x * (16 * 64);

    // mixed-radix digit reversal: i = 16*g + q -> source 4*q + g.
    // Stage 1 computes Y[g,j] = sum_q x[4*q+g] * exp(-2*pi*i*q*j/16).
#pragma unroll
    for (int batch = 0; batch < 16; ++batch) {
#pragma unroll
        for (int i = lane; i < 64; i += 32) {
            const int source = (i % 16) * 4 + i / 16;
            shared[batch * kRadix16SharedStride + i] =
                data[offset + batch * 64 + source];
        }
    }
    __syncwarp();

    // Like the historical throughput candidates, subsequent repetitions reuse
    // the resident layout without another digit reversal. R=1 is the public
    // FFT correctness contract; R>1 is the repeated arithmetic timing workload.
    for (unsigned int repeat = 0; repeat < inside_repeats; ++repeat) {
#pragma unroll
        for (int group = 0; group < 4; ++group)
            radix16_reconstruction::stage<16, 1>(shared, group * 16, 0);
        __syncwarp();

        // X[j + 16*p] = sum_g Y[g,j] * exp(-2*pi*i*g*(j/64 + p/4)).
        // The first stage issues 4 butterflies * 4 N tiles * 4 K tiles = 64
        // MMAs; this tail issues 16 butterflies * 1 * 1 = 16 MMAs. Thus one
        // iteration has exactly 80 m16n8k8 instructions per warp and 10240
        // issued MMA FLOPs per FFT. This counts dense zero/cancelling work too.
#pragma unroll
        for (int j = 0; j < 16; ++j)
            radix16_reconstruction::stage<4, 16>(shared, j, j);
        __syncwarp();
    }

#pragma unroll
    for (int batch = 0; batch < 16; ++batch) {
#pragma unroll
        for (int i = lane; i < 64; i += 32)
            data[offset + batch * 64 + i] =
                shared[batch * kRadix16SharedStride + i];
    }
}

} // namespace thff_table7_legacy
