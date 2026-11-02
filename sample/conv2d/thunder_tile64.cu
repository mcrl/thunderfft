#include <cufftdx.hpp>
#include <cute/tensor.hpp>
#include <thunderfft/thunderfft.cuh>

#include "../utils.h"

using namespace cute;

template <typename T> __device__ void swap_vals(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename T> __device__ void swap_thread_data(T *thread_data) {
    swap_vals(thread_data[1], thread_data[4]);
    swap_vals(thread_data[17], thread_data[20]);
    swap_vals(thread_data[2], thread_data[8]);
    swap_vals(thread_data[18], thread_data[24]);
    swap_vals(thread_data[3], thread_data[12]);
    swap_vals(thread_data[19], thread_data[28]);
    swap_vals(thread_data[6], thread_data[9]);
    swap_vals(thread_data[22], thread_data[25]);
    swap_vals(thread_data[7], thread_data[13]);
    swap_vals(thread_data[23], thread_data[29]);
    swap_vals(thread_data[11], thread_data[14]);
    swap_vals(thread_data[27], thread_data[30]);
}

// thread block size: 128
template <int f, class IGLayout, class FGLayout, class OGLayout, 
class SMLayoutX, class SMLayoutY, class ICTATiler, class OCTATiler>
__launch_bounds__(128)
__global__ void convolution_kernel(cufftdx::complex<float> *d_input, 
    cufftdx::complex<float> *d_filter, cufftdx::complex<float> *d_output, int N, float* dW,
    IGLayout input_gmem_layout, FGLayout filter_gmem_layout, OGLayout output_gmem_layout, 
    SMLayoutX smem_layout_x, SMLayoutY smem_layout_y, ICTATiler input_cta_tiler, OCTATiler output_cta_tiler)
{
    using complex_type = cufftdx::complex<float>;

    static_assert(f == 33, "f must be 33");

    static constexpr unsigned int tile_size = 64;
    static constexpr unsigned int batch_per_warp = 16;
    static constexpr unsigned int warp_size = 32;
    static constexpr unsigned int valid_tile_size = tile_size - f + 1;
    static constexpr unsigned int ept = tile_size * batch_per_warp / warp_size;

    // Allocate register
    extern __shared__ __align__(alignof(float4)) complex_type tile[];

    // thread idx
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    const int output_size = N - f + 1;

    auto input_gmem_tensor = make_tensor(make_gmem_ptr(d_input), input_gmem_layout);
    auto filter_gmem_tensor = make_tensor(make_gmem_ptr(d_filter), filter_gmem_layout);

    auto output_gmem_tensor = make_tensor(make_gmem_ptr(d_output), output_gmem_layout);

    auto smem_tensor_x = make_tensor(make_smem_ptr(tile), smem_layout_x);
    auto smem_tensor_y = make_tensor(make_smem_ptr(tile), smem_layout_y);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y);

    auto input_tile_gmem_tensor = local_tile(input_gmem_tensor, input_cta_tiler, cta_coord);
    auto output_tile_gmem_tensor = local_tile(output_gmem_tensor, output_cta_tiler, cta_coord);

    auto reg = make_tensor<cufftdx::complex<float>>(make_shape(Int<ept>{}), LayoutRight());

    // gmem -> smem    
    int row_index, col_index, reversed_row_index, reversed_col_index;

    // auto thread_layout = make_layout(make_shape(Int<4>{}, Int<32>{}), LayoutRight());
    // auto tG = local_partition(input_tile_gmem_tensor, thread_layout, threadIdx.x);
    // auto tS = local_partition(smem_tensor_x, thread_layout, threadIdx.x);
    // copy(tG, tS);
    // cp_async_fence();
    // cp_async_wait<0>();
    for (int j = 0; j < tile_size/warp_size; j++) {
        col_index = 32*j + lane_id;
        for (int i = 0; i < batch_per_warp; i++) {
            row_index = batch_per_warp*warp_id + i;
            smem_tensor_x(row_index, col_index) = input_tile_gmem_tensor(row_index, col_index);
        }
    }
    
    __syncthreads();

    // x-dim FFT: smem -> reg
    row_index = lane_id/4 + warp_id * batch_per_warp;
    for (int i = 0; i < ept/2; i++) {
        reversed_col_index = reverse_bit_groups<2, 6>(i*4 + (lane_id % 4));
        reg(i) = smem_tensor_x(row_index, reversed_col_index);
        reg(i + ept/2) = smem_tensor_x(row_index + batch_per_warp/2, reversed_col_index);
    }
    __syncthreads();

    // dW - d_input is trash value
    thunderfft::detail::unit::fft_kernel_r64_b16<true>(reinterpret_cast<float*>(reg.data()), dW);
    
    // x-dim FFT: reg -> smem
    for (int i = 0; i < ept/2; i++) {
        int col_index = i + (ept/2)*(lane_id % 4);
        smem_tensor_y(row_index, col_index) = reg(i);
        smem_tensor_y(row_index + batch_per_warp/2, col_index) = reg(i + ept/2);
    }

    __syncthreads();

    // y-dim FFT: smem -> reg
    col_index = lane_id/4 + warp_id * batch_per_warp;
    for (int i = 0; i < ept/2; i++) {
        reversed_row_index = reverse_bit_groups<2, 6>(i*4 + (lane_id % 4));
        reg(i) = smem_tensor_y(reversed_row_index, col_index);
        reg(i + ept/2) = smem_tensor_y(reversed_row_index, col_index + batch_per_warp/2);
    }

    __syncthreads();

    thunderfft::detail::unit::fft_kernel_r64_b16<true>(reinterpret_cast<float*>(reg.data()), dW);

    for (int i = 0; i < ept/2; i++) {
        row_index = i + (ept/2)*(lane_id % 4);
        reg(i) *= filter_gmem_tensor(row_index, col_index);
        reg(i + ept/2) *= filter_gmem_tensor(row_index, col_index + batch_per_warp/2);
        // reg(i) *= complex_type(1.3f, 2.4f);
        // reg(i + ept/2) *= complex_type(1.3f, 2.4f);
    }

    swap_thread_data(reg.data());

    // // y-dim FFT: reg -> smem
    // for (int i = 0; i < ept/2; i++) {
    //     int row_index = i + (ept/2)*(lane_id % 4);
    //     warp_smem_tensor_y(row_index, lane_id/4) = reg(i);
    //     warp_smem_tensor_y(row_index, lane_id/4 + batch_per_warp/2) = reg(i + ept/2);
    // }

    // __syncthreads();
    
    // // complex element-wise multiplication
    // for (int j = 0; j < tile_size/warp_size; j++) {
    //     int col_index = 32*j + lane_id;
    //     for (int i = 0; i < batch_per_warp; i++) {
    //         int row_index = batch_per_warp*warp_id + i;
    //         smem_tensor(row_index, col_index) *= filter_gmem_tensor(row_index, col_index);
    //     }
    // }

    // __syncthreads();

    // // y-dim IFFT: smem -> reg
    // for (int i = 0; i < ept/2; i++) {
    //     reversed_row_index = reverse_bit_groups<2, 6>(i*4 + (lane_id % 4));
    //     reg(i) = warp_smem_tensor_y(reversed_row_index, lane_id/4);
    //     reg(i + ept/2) = warp_smem_tensor_y(reversed_row_index, lane_id/4 + batch_per_warp/2);
    // }

    // __syncthreads();

    thunderfft::detail::unit::fft_kernel_r64_b16<false>(reinterpret_cast<float*>(reg.data()), dW);

    // y-dim IFFT: reg -> smem
    for (int i = 0; i < ept/2; i++) {
        row_index = i + (ept/2)*(lane_id % 4);
        smem_tensor_y(row_index, col_index) = reg(i);
        smem_tensor_y(row_index, col_index + batch_per_warp/2) = reg(i + ept/2);
    }

    __syncthreads();

    row_index = lane_id/4 + warp_id * batch_per_warp;
    if (row_index < valid_tile_size) {
        // x-dim IFFT: smem -> reg
        for (int i = 0; i < ept/2; i++) {
            reversed_col_index = reverse_bit_groups<2, 6>(i*4 + (lane_id % 4));
            reg(i) = smem_tensor_y(row_index, reversed_col_index);
            reg(i + ept/2) = smem_tensor_y(row_index + batch_per_warp/2, reversed_col_index);
        }

        __syncthreads();

        thunderfft::detail::unit::fft_kernel_r64_b16<false>(reinterpret_cast<float*>(reg.data()), dW);
        
        // x-dim IFFT: reg -> smem
        for (int i = 0; i < ept/2; i++) {
            int col_index = i + (ept/2)*(lane_id % 4);
            smem_tensor_x(row_index, col_index) = reg(i);
            smem_tensor_x(row_index + batch_per_warp/2, col_index) = reg(i + ept/2);
        }
    }

    __syncthreads();

    // smem -> gmem
    if (row_index < valid_tile_size) {
        int col_index = lane_id;
        if (col_index < valid_tile_size) {
            for (int i = 0; i < batch_per_warp; i++) {
                row_index = i + warp_id * batch_per_warp;
                output_tile_gmem_tensor(row_index, col_index) = smem_tensor_x(row_index, col_index);
            }
        }
    }
}

template <int f>
void my_convolution(const float2* h_input, const float2* h_filter, 
                   float2* h_output, int N) {
    static constexpr int warmup_runs = 5;
    static constexpr int runs = 100;

    cufftdx::complex<float>* d_input = nullptr;
    cufftdx::complex<float>* d_filter = nullptr;
    cufftdx::complex<float>* d_output = nullptr;
    static constexpr unsigned int tile_size = 64;
    static constexpr int valid_tile_size = tile_size - f + 1;
    int output_size = N - f + 1;
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(cufftdx::complex<float>) * N * N));
    // Filter needs to be padded to tile_size x tile_size for FFT-based convolution
    CHECK_CUDA(cudaMalloc(&d_filter, sizeof(cufftdx::complex<float>) * tile_size * tile_size));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(cufftdx::complex<float>) * output_size * output_size));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(float2) * N * N, cudaMemcpyHostToDevice));
    
    // Pad filter to tile_size x tile_size (zero-padded)
    cufftdx::complex<float>* h_filter_padded = (cufftdx::complex<float>*)std::calloc(tile_size * tile_size, sizeof(cufftdx::complex<float>));
    for (int i = 0; i < f; ++i) {
        for (int j = 0; j < f; ++j) {
            // Flip filter and place at wrap-around position for circular convolution
            int idx_i = (tile_size - i) % tile_size;
            int idx_j = (tile_size - j) % tile_size;
            h_filter_padded[idx_i * tile_size + idx_j].x = h_filter[i * f + j].x / tile_size / tile_size;
            h_filter_padded[idx_i * tile_size + idx_j].y = h_filter[i * f + j].y / tile_size / tile_size;
        }
    }
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter_padded, sizeof(cufftdx::complex<float>) * tile_size * tile_size, cudaMemcpyHostToDevice));
    std::free(h_filter_padded);

    // FFT filter
    cufftHandle plan_forward;
    CHECK_CUFFT(cufftPlan2d(&plan_forward, tile_size, tile_size, CUFFT_C2C));
    CHECK_CUFFT(cufftExecC2C(plan_forward, reinterpret_cast<cufftComplex*>(d_filter), reinterpret_cast<cufftComplex*>(d_filter), CUFFT_FORWARD));

    dim3 blockSize(128);
    dim3 gridSize((output_size + valid_tile_size - 1) / valid_tile_size,
                  (output_size + valid_tile_size - 1) / valid_tile_size);

    auto input_gmem_layout = make_layout(
        make_shape(
            make_shape(Int<tile_size>{}, gridSize.x), 
            make_shape(Int<tile_size>{}, gridSize.y)
        ), 
        make_stride(
            make_stride(N, valid_tile_size*N), 
            make_stride(Int<1>{}, Int<valid_tile_size>{})
        )
    );

    auto output_gmem_layout = make_layout(
        make_shape(
            make_shape(Int<valid_tile_size>{}, gridSize.x), 
            make_shape(Int<valid_tile_size>{}, gridSize.y)
        ),
        make_stride(
            make_stride(output_size, valid_tile_size*output_size),
            make_stride(Int<1>{}, Int<valid_tile_size>{})
        )
    );

    auto filter_gmem_layout = make_layout(
        make_shape(
            Int<tile_size>{}, Int<tile_size>{}
        ),
        LayoutRight()
    );

    smem_layout_x = make_layout(
        make_shape(Int<tile_size>{}, make_shape(Int<tile_size/4>{}, Int<4>{})), 
        make_stride(Int<tile_size+4>{}, make_stride(Int<1>{}, Int<tile_size/4+1>{}))
    );

    class typename SMLayoutY = decltype(smem_layout_x);

    auto smem_layout_x = make_layout(
        make_shape(Int<tile_size>{}, make_shape(Int<tile_size/4>{}, Int<4>{})), 
        make_stride(Int<tile_size+4>{}, make_stride(Int<1>{}, Int<tile_size/4+1>{}))
    );

    auto smem_layout_y = make_layout(
        make_shape(
            make_shape(Int<tile_size/4>{}, Int<4>{}), 
            make_shape(Int<tile_size/4>{}, Int<4>{})
        ), 
        make_stride(
            make_stride(Int<tile_size+4>{}, Int<(tile_size+4)*tile_size/4+4>{}), 
            make_stride(Int<1>{}, Int<tile_size/4+1>{})
        )
    );

    auto input_cta_tiler = make_shape(Int<tile_size>{}, Int<tile_size>{});
    auto output_cta_tiler = make_shape(Int<valid_tile_size>{}, Int<valid_tile_size>{});

    float* dW = thunderfft::preprocess_W<float>(64);
    size_t shared_memory_size = sizeof(cufftdx::complex<float>) * cosize_v<decltype(smem_layout_y)>;
    
    // auto kernel = [&]() { 
    //     convolution_kernel<f><<<gridSize, blockSize, shared_memory_size>>>
    //     (d_input, d_filter, d_output, N, dW, input_gmem_layout, filter_gmem_layout, 
    //         output_gmem_layout, smem_layout_x, smem_layout_y, input_cta_tiler, output_cta_tiler); 
    // };

    float ms = measure_execution_ms(kernel, warmup_runs, runs);
    std::cout << "[ThunderFFT] Complex 2D convolution: " << ms << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_output, d_output, sizeof(cufftdx::complex<float>) * output_size * output_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUFFT(cufftDestroy(plan_forward));
}

// template void my_convolution<33>(const float2*, const float2*, float2*, int);
