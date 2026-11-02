#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufft.h>

#include <vector>
#include <cstdlib>
#include <iostream>

#include <cute/tensor.hpp>
#include <thunderfft/thunderfft.cuh>

#include "../utils.h"

using namespace cute;

// template <typename T> __device__ void swap_vals(T &a, T &b) {
//     T tmp = a;
//     a = b;
//     b = tmp;
// }

// template <typename T> __device__ void swap_thread_data(T *thread_data) {
//     swap_vals(thread_data[1], thread_data[4]);
//     swap_vals(thread_data[17], thread_data[20]);
//     swap_vals(thread_data[2], thread_data[8]);
//     swap_vals(thread_data[18], thread_data[24]);
//     swap_vals(thread_data[3], thread_data[12]);
//     swap_vals(thread_data[19], thread_data[28]);
//     swap_vals(thread_data[6], thread_data[9]);
//     swap_vals(thread_data[22], thread_data[25]);
//     swap_vals(thread_data[7], thread_data[13]);
//     swap_vals(thread_data[23], thread_data[29]);
//     swap_vals(thread_data[11], thread_data[14]);
//     swap_vals(thread_data[27], thread_data[30]);
// }

// ----------------------------
// Kernel: half2 input/output, float2(or cufftComplex) filter
// ----------------------------

__device__ __forceinline__
int smem_index_128_pad32(int row, int col) {
    // TILE=128, pad: +1 per 32 elements
    constexpr int TILE = 128;
    const int lin = row * TILE + col;
    return lin + (lin >> 5); // lin/32
}

__device__ static void print(half2* smem) {
    __syncthreads();
    if(threadIdx.x==0 && blockIdx.x==0) {
        for(int i=0; i<128; i++) {
            for(int j=0; j<128; j++) {
                int sidx = smem_index_128_pad32(i, j);
                printf("(%f,%f)", __half2float(smem[sidx].x), __half2float(smem[sidx].y));
            }
            printf("\n");
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// 1024x1024 FFT-conv helpers (half2, 1-batch 1024-pt kernel)
// ---------------------------------------------------------------------------
static constexpr int kFFTSize1024 = 1024;
static constexpr int kTileRows1024 = 32;
static constexpr int kWorkspaceStride1024 = (64 + 64 / 32) * 16; // 1056

__global__ void conv1024_x_forward(const half2* __restrict__ d_input,
                                   half2* __restrict__ d_workspace,
                                   int tile_idx) {
    extern __shared__ unsigned char smem_raw[];
    half2* smem_data = reinterpret_cast<half2*>(smem_raw);
    half2* smem_ws = smem_data + kFFTSize1024 * kTileRows1024;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int warp_count = blockDim.x / 32;
    const int rows_per_warp = kTileRows1024 / warp_count;
    const int row_base = tile_idx * kTileRows1024 + warp_id * rows_per_warp;

    using L1024 = thunderfft::layout_t<kFFTSize1024, 1, 1, kFFTSize1024, 32, 0, false>;
    half2 W_reg[36];
    thunderfft::unit_fp16::make_reg_b_precompute<kFFTSize1024, true>(W_reg);
    half2 reg[32];
    half2* warp_ws = smem_ws + warp_id * kWorkspaceStride1024;

    #pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) {
        const int row = row_base + r;
        half2* row_smem = smem_data + (warp_id * rows_per_warp + r) * kFFTSize1024;

        for (int col = lane; col < kFFTSize1024; col += 32) {
            row_smem[col] = d_input[row * kFFTSize1024 + col];
        }
        __syncwarp();

        thunderfft::ThunderFFT_smem2reg<half, L1024>(reg, row_smem);
        thunderfft::ThunderFFT_kernel_reg<half, kFFTSize1024, 1, true>(reg, W_reg, warp_ws);
        thunderfft::ThunderFFT_reg2smem<half, L1024>(row_smem, reg);
        __syncwarp();

        for (int col = lane; col < kFFTSize1024; col += 32) {
            d_workspace[row * kFFTSize1024 + col] = row_smem[col];
        }
        __syncwarp();
    }
}

__global__ void conv1024_y_fwd_mul_inv(half2* __restrict__ d_workspace,
                                      const half2* __restrict__ d_filter,
                                      int tile_idx) {
    extern __shared__ unsigned char smem_raw[];
    half2* smem_data = reinterpret_cast<half2*>(smem_raw);
    half2* smem_ws = smem_data + kFFTSize1024 * kTileRows1024;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int warp_count = blockDim.x / 32;
    const int cols_per_warp = kTileRows1024 / warp_count;
    const int col_base = tile_idx * kTileRows1024 + warp_id * cols_per_warp;

    using L1024 = thunderfft::layout_t<kFFTSize1024, 1, 1, kFFTSize1024, 32, 0, false>;
    half2 W_fwd[36];
    half2 W_inv[36];
    thunderfft::unit_fp16::make_reg_b_precompute<kFFTSize1024, true>(W_fwd);
    thunderfft::unit_fp16::make_reg_b_precompute<kFFTSize1024, false>(W_inv);
    half2 reg[32];
    half2* warp_ws = smem_ws + warp_id * kWorkspaceStride1024;

    #pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
        const int col = col_base + c;
        half2* col_smem = smem_data + (warp_id * cols_per_warp + c) * kFFTSize1024;

        for (int row = lane; row < kFFTSize1024; row += 32) {
            col_smem[row] = d_workspace[row * kFFTSize1024 + col];
        }
        __syncwarp();

        thunderfft::ThunderFFT_smem2reg<half, L1024>(reg, col_smem);
        thunderfft::ThunderFFT_kernel_reg<half, kFFTSize1024, 1, true>(reg, W_fwd, warp_ws);
        thunderfft::ThunderFFT_reg2smem<half, L1024>(col_smem, reg);
        __syncwarp();

        for (int row = lane; row < kFFTSize1024; row += 32) {
            const int fidx = row * kFFTSize1024 + col;
            col_smem[row] = cmul(col_smem[row], d_filter[fidx]);
        }
        __syncwarp();

        thunderfft::ThunderFFT_smem2reg<half, L1024>(reg, col_smem);
        thunderfft::ThunderFFT_kernel_reg<half, kFFTSize1024, 1, false>(reg, W_inv, warp_ws);
        thunderfft::ThunderFFT_reg2smem<half, L1024>(col_smem, reg);
        __syncwarp();

        for (int row = lane; row < kFFTSize1024; row += 32) {
            d_workspace[row * kFFTSize1024 + col] = col_smem[row];
        }
        __syncwarp();
    }
}

__global__ void conv1024_x_inverse(const half2* __restrict__ d_workspace,
                                   half2* __restrict__ d_output,
                                   int out_size,
                                   int tile_idx) {
    extern __shared__ unsigned char smem_raw[];
    half2* smem_data = reinterpret_cast<half2*>(smem_raw);
    half2* smem_ws = smem_data + kFFTSize1024 * kTileRows1024;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int warp_count = blockDim.x / 32;
    const int rows_per_warp = kTileRows1024 / warp_count;
    const int row_base = tile_idx * kTileRows1024 + warp_id * rows_per_warp;

    using L1024 = thunderfft::layout_t<kFFTSize1024, 1, 1, kFFTSize1024, 32, 0, false>;
    half2 W_inv[36];
    thunderfft::unit_fp16::make_reg_b_precompute<kFFTSize1024, false>(W_inv);
    half2 reg[32];
    half2* warp_ws = smem_ws + warp_id * kWorkspaceStride1024;

    #pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) {
        const int row = row_base + r;
        half2* row_smem = smem_data + (warp_id * rows_per_warp + r) * kFFTSize1024;

        for (int col = lane; col < kFFTSize1024; col += 32) {
            row_smem[col] = d_workspace[row * kFFTSize1024 + col];
        }
        __syncwarp();

        thunderfft::ThunderFFT_smem2reg<half, L1024>(reg, row_smem);
        thunderfft::ThunderFFT_kernel_reg<half, kFFTSize1024, 1, false>(reg, W_inv, warp_ws);
        thunderfft::ThunderFFT_reg2smem<half, L1024>(row_smem, reg);
        __syncwarp();

        if (row < out_size) {
            for (int col = lane; col < out_size; col += 32) {
                d_output[row * out_size + col] = row_smem[col];
            }
        }
        __syncwarp();
    }
}
template <int f>
__launch_bounds__(128)
__global__ void convolution_kernel(
    const half2* __restrict__ d_input,   // [N*N] half2 complex
    const half2* __restrict__ d_filter,  // currently unused
    half2* __restrict__ d_output,        // [(N-f+1)^2] half2 complex
    int N)
{
    extern __shared__ unsigned char smem_raw[];
    half2* smem = reinterpret_cast<half2*>(smem_raw);

    constexpr int TILE = 128;
    const int valid_tile = TILE - f + 1;
    const int out_size   = N - f + 1;

    const int tile_x0 = int(blockIdx.x) * valid_tile;
    const int tile_y0 = int(blockIdx.y) * valid_tile;

    const int col = int(threadIdx.x);

    // ----------------------------
    // 1) Load TILE x TILE into shared memory (OOB -> 0)
    // ----------------------------
    if (col < TILE) {
        const half2 z = __floats2half2_rn(0.0f, 0.0f);

        #pragma unroll
        for (int row = 0; row < TILE; ++row) {
            const int gx = tile_x0 + row;
            const int gy = tile_y0 + col;

            const int sidx = smem_index_128_pad32(row, col);
            smem[sidx] = (gx < N && gy < N) ? d_input[gx * N + gy] : z;
        }
    }
    __syncthreads();
    {
    using vec2 = vec2_t<half>;
    constexpr int N = 128;
    constexpr int BPB = 8;
    constexpr int ept = N * BPB / thunderfft::threads_per_warp;
    constexpr int groups = N / BPB;
    constexpr int padded_stride = N + N / 32;
    using Lx = thunderfft::layout_t<N, BPB, 1, N, 32, 1, false>;
    using Ly = thunderfft::layout_t<N, BPB, N, 1, 32, 1, false>;

    vec2 reg[ept];
    vec2 W_fwd[28];
    vec2 W_inv[28];
    thunderfft::unit_fp16::make_reg_b_precompute<N, true>(W_fwd);
    thunderfft::unit_fp16::make_reg_b_precompute<N, false>(W_inv);

    const int warp_id = int(threadIdx.x) / thunderfft::threads_per_warp;
    const int warp_count = int(blockDim.x) / thunderfft::threads_per_warp;

    // X-axis FFT (rows).
    for (int g = warp_id; g < groups; g += warp_count) {
        vec2* smem_group = reinterpret_cast<vec2*>(smem) + g * BPB * padded_stride;

        thunderfft::ThunderFFT_smem2reg<half, Lx>(reg, smem_group);
        __syncwarp();

        thunderfft::ThunderFFT_kernel_reg<half, N, BPB, true>(reg, W_fwd, smem_group);
        __syncwarp();


        // for(int i=0; i<ept; i++) reg[i].x/=128, reg[i].y/=128;
        thunderfft::ThunderFFT_reg2smem<half, Lx>(smem_group, reg);
        __syncwarp();
    }
    __syncthreads();

    // Y-axis FFT (columns).
    for (int g = warp_id; g < groups; g += warp_count) {
        const int col_group = g * BPB;
        const int col_offset = col_group + col_group / 32;
        vec2* smem_group = reinterpret_cast<vec2*>(smem) + col_offset;

        thunderfft::ThunderFFT_smem2reg<half, Ly>(reg, smem_group);
        __syncwarp();

        thunderfft::ThunderFFT_kernel_reg<half, N, BPB, true>(reg, W_fwd, smem_group);
        __syncwarp();

        
        // for(int i=0; i<ept; i++) reg[i].x/=128, reg[i].y/=128;


        int lane = threadIdx.x % 32;
        int lane_mod = lane % 4;
        int col = col_group + lane/4;
        for (int idx = 0; idx < ept; ++idx) {
            const int i = idx & (ept / 2 - 1);
            const int row = (i + lane_mod * 16) + 64 * (idx >= ept / 2);
            const int fidx = row * N + col;
            reg[idx] = cmul(reg[idx], d_filter[fidx]);
        }
        thunderfft::ThunderFFT_reg2smem<half, Ly>(smem_group, reg);
        __syncwarp();
    // }
        __syncthreads();

        // Previous in-register multiply (disabled).
        // const int lane = int(threadIdx.x) % thunderfft::threads_per_warp;
        // const int lane_mod = lane & 3;
        // const int col = col_group + lane / 4;
        // #pragma unroll
        // for (int idx = 0; idx < ept; ++idx) {
        //     const int i = idx & (ept / 2 - 1);
        //     const int row = (i + lane_mod * 16) + 64 * (idx >= ept / 2);
        //     const int fidx = row * N + col;
        //     reg[idx] = cmul(reg[idx], d_filter[fidx]);
        // }
        __syncthreads();

    // Element-wise multiplication in frequency domain (smem -> reg -> smem).
        // if (col < N) {
        // // if (col >=col_group && col<col_group + 8) {
        //     #pragma unroll
        //     for (int row = 0; row < N; ++row) {
        //         const int sidx = smem_index_128_pad32(row, col);
        //         const int fidx = row * N + col;
        //         smem[sidx] = cmul(smem[sidx], d_filter[fidx]);
        //     }
        // }
        // __syncthreads();

        // swap_thread_data(reg);
    // for (int g = warp_id; g < groups; g += warp_count) {
    //     const int col_group = g * BPB;
    //     const int col_offset = col_group + col_group / 32;
        // vec2* smem_group = reinterpret_cast<vec2*>(smem) + col_offset;

        thunderfft::ThunderFFT_smem2reg<half, Ly>(reg, smem_group);
        __syncwarp();

        thunderfft::ThunderFFT_kernel_reg<half, N, BPB, false>(reg, W_inv, smem_group);
        __syncwarp();

        
        // for(int i=0; i<ept; i++) reg[i].x*=128, reg[i].y*=128;
        thunderfft::ThunderFFT_reg2smem<half, Ly>(smem_group, reg);
        __syncwarp();
    }
    __syncthreads();

    // // Inverse X-axis FFT (rows).
    for (int g = warp_id; g < groups; g += warp_count) {
        vec2* smem_group = reinterpret_cast<vec2*>(smem) + g * BPB * padded_stride;

        thunderfft::ThunderFFT_smem2reg<half, Lx>(reg, smem_group);
        __syncwarp();

        thunderfft::ThunderFFT_kernel_reg<half, N, BPB, false>(reg, W_inv, smem_group);
        __syncwarp();

        // for(int i=0; i<ept; i++) reg[i].x*=128, reg[i].y*=128;
        thunderfft::ThunderFFT_reg2smem<half, Lx>(smem_group, reg);
        __syncwarp();
    }
    __syncthreads();
    }

    if (col < valid_tile) {
        #pragma unroll
        for (int row = 0; row < valid_tile; ++row) {
            const int gx = tile_x0 + row;
            const int gy = tile_y0 + col;

            if (gx < out_size && gy < out_size) {
                const int sidx = smem_index_128_pad32(row, col);
                d_output[gx * out_size + gy] = smem[sidx];
            }
        }
    }
}

// ----------------------------
// CPU conversion helpers
// ----------------------------
static inline half2 float2_to_half2_complex(const float2& v) {
    return __floats2half2_rn(v.x, v.y);
}

static inline float2 half2_to_float2_complex(const half2& h) {
    float2 out;
    out.x = __half2float(__low2half(h));
    out.y = __half2float(__high2half(h));
    return out;
}

__global__ void float2_to_half2_kernel(const float2* __restrict__ in,
                                       half2* __restrict__ out,
                                       int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float2 v = in[i];
        out[i] = __floats2half2_rn(v.x, v.y);
    }
}

template <int f>
void my_convolution(const float2* h_input,
                    const float2* h_filter,
                    float2* h_output,
                    int N,
                    int warp_count) {
    static constexpr int warmup_runs = 5;
    static constexpr int runs = 100;

    if (N != kFFTSize1024) {
        std::cerr << "[ThunderFFT] This path expects N=1024.\n";
        return;
    }
    if (warp_count <= 0 || warp_count > kTileRows1024 || (kTileRows1024 % warp_count) != 0) {
        std::cerr << "[ThunderFFT] warp_count must divide 32.\n";
        return;
    }

    constexpr int tile_size = kFFTSize1024;
    const int output_size = N - f + 1;

    // for(int i=0; i<128; i++) {
    //     for(int j=0; j<128; j++) {
    //         std::cout << h_input[i*128+j].x << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // for(int i=0; i<3; i++) {
    //     for(int j=0; j<3; j++) {
    //         std::cout << h_filter[i*3+j].x << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // ----------------------------
    // CPU: float2 -> half2 (input)
    // ----------------------------
    std::vector<half2> h_input_half((size_t)N * (size_t)N);
    for (int i = 0; i < N * N; ++i) {
        h_input_half[(size_t)i] = float2_to_half2_complex(h_input[i]);
    }

    // Output half2 buffer on host
    std::vector<half2> h_output_half((size_t)output_size * (size_t)output_size);

    // ----------------------------
    // Pad filter on CPU as float2
    // ----------------------------
    std::vector<float2> h_filter_padded((size_t)tile_size * (size_t)tile_size);
    for (int i = 0; i < tile_size * tile_size; ++i) {
        h_filter_padded[(size_t)i] = make_float2(0.0f, 0.0f);
    }

    for (int i = 0; i < f; ++i) {
        for (int j = 0; j < f; ++j) {
            int idx_i = (tile_size - i) % tile_size;
            int idx_j = (tile_size - j) % tile_size;

            const float scale = 1.0f / float(tile_size) / float(tile_size);
            const float2 v = h_filter[i * f + j];

            h_filter_padded[(size_t)idx_i * tile_size + (size_t)idx_j] =
                make_float2(v.x * scale, v.y * scale);
        }
    }

    // ----------------------------
    // Device allocations
    //   - input/output: half2
    //   - filter FFT workspace: float2
    //   - filter after FFT: half2
    // ----------------------------
    half2* d_input  = nullptr;
    half2* d_output = nullptr;
    half2* d_workspace = nullptr;
    float2* d_filter_float = nullptr; // for cuFFT
    half2* d_filter_half = nullptr;   // final filter for kernel

    const size_t bytes_in  = sizeof(half2) * (size_t)N * (size_t)N;
    const size_t bytes_out = sizeof(half2) * (size_t)output_size * (size_t)output_size;
    const size_t bytes_f_f = sizeof(float2) * (size_t)tile_size * (size_t)tile_size;
    const size_t bytes_f_h = sizeof(half2)  * (size_t)tile_size * (size_t)tile_size;
    const size_t bytes_ws  = sizeof(half2)  * (size_t)tile_size * (size_t)tile_size;

    CHECK_CUDA(cudaMalloc(&d_input, bytes_in));
    CHECK_CUDA(cudaMalloc(&d_output, bytes_out));
    CHECK_CUDA(cudaMalloc(&d_filter_float, bytes_f_f));
    CHECK_CUDA(cudaMalloc(&d_filter_half,  bytes_f_h));
    CHECK_CUDA(cudaMalloc(&d_workspace,    bytes_ws));

    // H2D input (half2)
    CHECK_CUDA(cudaMemcpy(d_input, h_input_half.data(), bytes_in, cudaMemcpyHostToDevice));

    // H2D filter padded (float2)
    CHECK_CUDA(cudaMemcpy(d_filter_float,
                          h_filter_padded.data(),
                          bytes_f_f,
                          cudaMemcpyHostToDevice));

    // ----------------------------
    // FFT filter in-place (float)
    // cuFFT expects cufftComplex*, but float2 has same layout (x,y).
    // We avoid cufftComplex in our storage types; only cast at API boundary.
    // ----------------------------
    cufftHandle plan_forward;
    CHECK_CUFFT(cufftPlan2d(&plan_forward, tile_size, tile_size, CUFFT_C2C));
    CHECK_CUFFT(cufftExecC2C(plan_forward,
                             reinterpret_cast<cufftComplex*>(d_filter_float),
                             reinterpret_cast<cufftComplex*>(d_filter_float),
                             CUFFT_FORWARD));

    // ----------------------------
    // Convert FFT'd filter float2 -> half2 (on GPU)
    // ----------------------------
    {
        const int n = tile_size * tile_size;
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        float2_to_half2_kernel<<<blocks, threads>>>(d_filter_float, d_filter_half, n);
        CHECK_CUDA(cudaGetLastError());
    }

    // ThunderFFT init
    thunderfft::ThunderFFTInitialize<half>(kFFTSize1024);

    const size_t shared_memory_size =
        sizeof(half2) * (kFFTSize1024 * kTileRows1024 + kWorkspaceStride1024 * warp_count);

    CHECK_CUDA(cudaFuncSetAttribute(conv1024_x_forward,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared_memory_size)));
    CHECK_CUDA(cudaFuncSetAttribute(conv1024_y_fwd_mul_inv,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared_memory_size)));
    CHECK_CUDA(cudaFuncSetAttribute(conv1024_x_inverse,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared_memory_size)));

    const dim3 blockSize(warp_count * 32);
    const int tiles = (kFFTSize1024 + kTileRows1024 - 1) / kTileRows1024;

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < tiles; ++i) {
        for (int j = 0; j < tiles; ++j) {
            conv1024_x_forward<<<1, blockSize, shared_memory_size>>>(d_input, d_workspace, i);
            CHECK_CUDA(cudaGetLastError());
            conv1024_y_fwd_mul_inv<<<1, blockSize, shared_memory_size>>>(d_workspace, d_filter_half, j);
            CHECK_CUDA(cudaGetLastError());
            conv1024_x_inverse<<<1, blockSize, shared_memory_size>>>(d_workspace, d_output, output_size, i);
            CHECK_CUDA(cudaGetLastError());
        }
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::cout << "[ThunderFFT] Complex 2D convolution (half2 I/O, filter half2): "
              << ms << " ms" << std::endl;

    // ----------------------------
    // D2H output (half2) then CPU convert to float2
    // ----------------------------
    CHECK_CUDA(cudaMemcpy(h_output_half.data(), d_output, bytes_out, cudaMemcpyDeviceToHost));

    for (int i = 0; i < output_size * output_size; ++i) {
        h_output[i] = half2_to_float2_complex(h_output_half[(size_t)i]);
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_filter_float));
    CHECK_CUDA(cudaFree(d_filter_half));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUFFT(cufftDestroy(plan_forward));
}

template void my_convolution<3>(const float2*, const float2*, float2*, int, int);
template void my_convolution<33>(const float2*, const float2*, float2*, int, int);
