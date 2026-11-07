#include <cufftdx.hpp>
#include <cute/tensor.hpp>
#include "../utils.h"
#include "cute/tensor_impl.hpp"
#include <functional>
#include <vector>

#ifndef THFF_CONV_SM
#define THFF_CONV_SM 800
#endif

using namespace cufftdx;
using namespace cute;

static constexpr int loop_x = 1;
static constexpr int loop_y = 1;

template <typename real_type, int tile_size, int f, class fft_c2c_forward, class fft_c2c_inverse, 
class IGLayout, class FGLayout, class OGLayout, class SMLayout, class ICTATiler, class OCTATiler>
__launch_bounds__(tile_size)
__global__ void conv_kernel_thread(cufftdx::complex<real_type> *d_input,
                                   cufftdx::complex<real_type> *d_filter,
                                   cufftdx::complex<real_type> *d_output, int N, 
                                   IGLayout input_gmem_layout, FGLayout filter_gmem_layout, OGLayout output_gmem_layout, 
                                   SMLayout smem_layout, ICTATiler input_cta_tiler, OCTATiler output_cta_tiler)
{
    using complex_type = cufftdx::complex<real_type>;

    extern __shared__ __align__(alignof(float4)) complex_type tile[];

    const int local_thread_id = threadIdx.x;
    static constexpr int valid_tile_size = tile_size - f + 1;
    const int output_size = N - f + 1;

    auto input_gmem_tensor = make_tensor(make_gmem_ptr(d_input), input_gmem_layout);
    auto filter_gmem_tensor = make_tensor(make_gmem_ptr(d_filter), filter_gmem_layout);

    auto output_gmem_tensor = make_tensor(make_gmem_ptr(d_output), output_gmem_layout);

    auto smem_tensor = make_tensor(make_smem_ptr(tile), smem_layout);
    auto filter_smem_tensor = make_tensor(make_smem_ptr(tile+cosize_v<decltype(smem_layout)>), smem_layout);

    auto cta_coord = make_coord(blockIdx.y, blockIdx.x);

    auto IG_tile = local_tile(input_gmem_tensor, input_cta_tiler, cta_coord);
    auto OG_tile = local_tile(output_gmem_tensor, output_cta_tiler, cta_coord);

    auto input_loop_tiler = make_shape(Int<tile_size>{}, Int<tile_size>{});
    auto output_loop_tiler = make_shape(Int<valid_tile_size>{}, Int<valid_tile_size>{});

    auto reg = make_tensor<complex_type>(make_shape(Int<tile_size>{}));

    int row_index, col_index;

    // load filter to smem
    // for (int row_index = 0; row_index < tile_size; row_index++) {
    //     col_index = local_thread_id;
    //     filter_smem_tensor(row_index, col_index) = filter_gmem_tensor(row_index, col_index);
    // }

    __syncthreads();

    for (int loop_y_idx = 0; loop_y_idx < loop_y; ++loop_y_idx) {
        for (int loop_x_idx = 0; loop_x_idx < loop_x; ++loop_x_idx) {
            auto loop_coord = make_coord(loop_y_idx, loop_x_idx);
            auto input_tile_gmem_tensor = local_tile(IG_tile, input_loop_tiler, loop_coord);
            auto output_tile_gmem_tensor = local_tile(OG_tile, output_loop_tiler, loop_coord);
            
            // 1) Column-wise C2C (forward)
            col_index = local_thread_id;
            for (row_index = 0; row_index < tile_size; ++row_index) {
                bool is_valid = blockIdx.y * loop_y * valid_tile_size + loop_y_idx * valid_tile_size + row_index < N &&
                                blockIdx.x * loop_x * valid_tile_size + loop_x_idx * valid_tile_size + col_index < N;
                reg(row_index) = is_valid ? input_tile_gmem_tensor(row_index, col_index) : complex_type(0.0f, 0.0f);
            }
            fft_c2c_forward().execute(reg.data());
            
            col_index = local_thread_id;
            for (row_index = 0; row_index < tile_size; ++row_index) {
                smem_tensor(row_index, col_index) = reg(row_index);
            }

            __syncthreads();

            // 2) Row-wise C2C (forward)
            row_index = local_thread_id;
            for (col_index = 0; col_index < tile_size; ++col_index) {
                reg(col_index) = smem_tensor(row_index, col_index);
            }
            fft_c2c_forward().execute(reg.data());

            // 3) Pointwise multiply
            for (col_index = 0; col_index < tile_size; ++col_index) {
                reg(col_index) *= filter_gmem_tensor(col_index, row_index); // filter transposed
            }

            // 4) Row-wise C2C (inverse)
            fft_c2c_inverse().execute(reg.data());

            for (col_index = 0; col_index < tile_size; ++col_index) {
                smem_tensor(row_index, col_index) = reg(col_index);
            }
            __syncthreads();

            // 5) Column-wise C2C
            col_index = local_thread_id;
            if (col_index < valid_tile_size) {
                for (row_index = 0; row_index < tile_size; ++row_index) {
                    reg(row_index) = smem_tensor(row_index, col_index);
                }

                fft_c2c_inverse().execute(reg.data());

                for (row_index = 0; row_index < valid_tile_size; ++row_index) {
                    bool is_valid = blockIdx.y * loop_y * valid_tile_size + loop_y_idx * valid_tile_size + row_index < output_size &&
                                    blockIdx.x * loop_x * valid_tile_size + loop_x_idx * valid_tile_size + col_index < output_size;
                    if (is_valid) {
                        output_tile_gmem_tensor(row_index, col_index) = reg(row_index);
                    }
                }
            }
        }
    }
}

// Returns one mean time per trial (filter FFT + tiled kernel), or nothing if gate
// rejects the output of an untimed first run.
template <int f>
std::vector<float> cufftdx_convolution(const float2* h_input, const float2* h_filter, int N,
                                       const std::function<bool(const float2*)>& gate, int warmup_runs,
                                       int runs, int trials) {
    using complex_type = cufftdx::complex<float>;
    using fft_c2c = decltype(Thread() + Precision<float>() + SM<THFF_CONV_SM>() + Size<64>() + Type<fft_type::c2c>());
    using fft_c2c_forward = decltype(fft_c2c() + Direction<fft_direction::forward>());
    using fft_c2c_inverse = decltype(fft_c2c() + Direction<fft_direction::inverse>());
    cufftdx::complex<float>* d_input = nullptr;
    cufftdx::complex<float>* d_filter_spatial = nullptr;
    cufftdx::complex<float>* d_filter = nullptr;
    cufftdx::complex<float>* d_output = nullptr;
    static constexpr unsigned int tile_size = 64;
    const unsigned int valid_tile_size = tile_size - f + 1;
    int output_size = N - f + 1;
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(cufftdx::complex<float>) * N * N));
    // Filter needs to be padded to tile_size x tile_size for FFT-based convolution
    CHECK_CUDA(cudaMalloc(&d_filter, sizeof(cufftdx::complex<float>) * tile_size * tile_size));
    CHECK_CUDA(cudaMalloc(&d_filter_spatial, sizeof(cufftdx::complex<float>) * tile_size * tile_size));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(cufftdx::complex<float>) * output_size * output_size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(float2) * N * N, cudaMemcpyHostToDevice));

    // Pad filter to tile_size x tile_size (zero-padded)
    cufftdx::complex<float>* h_filter_padded = (cufftdx::complex<float>*)std::calloc(tile_size * tile_size, sizeof(cufftdx::complex<float>));
    for (int i = 0; i < f; ++i) {
        for (int j = 0; j < f; ++j) {
            // Flip filter and place at wrap-around position for circular convolution
            int idx_i = (tile_size - i) % tile_size;
            int idx_j = (tile_size - j) % tile_size;
            h_filter_padded[idx_i * tile_size + idx_j].x = h_filter[j * f + i].x / tile_size / tile_size;
            h_filter_padded[idx_i * tile_size + idx_j].y = h_filter[j * f + i].y / tile_size / tile_size;
        }
    }
    CHECK_CUDA(cudaMemcpy(d_filter_spatial, h_filter_padded, sizeof(cufftdx::complex<float>) * tile_size * tile_size, cudaMemcpyHostToDevice));
    std::free(h_filter_padded);

    // The filter FFT runs inside every timed iteration (Figure 5 timing scope).
    cufftHandle plan_forward;
    CHECK_CUFFT(cufftPlan2d(&plan_forward, tile_size, tile_size, CUFFT_C2C));

    dim3 blockSize(tile_size);
    dim3 gridSize((output_size + valid_tile_size*loop_y - 1) / valid_tile_size / loop_y,
    (output_size + valid_tile_size*loop_x - 1) / valid_tile_size / loop_x);

    auto input_gmem_layout = make_layout(
        make_shape(
            make_shape(make_shape(Int<tile_size>{}, Int<loop_y>{}), gridSize.y), 
            make_shape(make_shape(Int<tile_size>{}, Int<loop_x>{}), gridSize.x)
        ), 
        make_stride(
            make_stride(make_stride(N, N*valid_tile_size), valid_tile_size*N*loop_y), 
            make_stride(make_stride(Int<1>{}, Int<valid_tile_size>{}), Int<valid_tile_size*loop_x>{})
        )
    );

    auto output_gmem_layout = make_layout(
        make_shape(
            make_shape(make_shape(Int<valid_tile_size>{}, Int<loop_y>{}), gridSize.y), 
            make_shape(make_shape(Int<valid_tile_size>{}, Int<loop_x>{}), gridSize.x)
        ),
        make_stride(
            make_stride(make_stride(output_size, valid_tile_size*output_size), valid_tile_size*output_size*loop_y),
            make_stride(make_stride(Int<1>{}, Int<valid_tile_size>{}), Int<valid_tile_size*loop_x>{})
        )
    );

    auto filter_gmem_layout = make_layout(
        make_shape(
            Int<tile_size>{}, Int<tile_size>{}
        ),
        LayoutRight()
    );

    auto layout = make_layout(
        make_shape(Int<tile_size>{}, Int<tile_size>{}),
        make_stride(Int<tile_size>{}, Int<1>{})
    );
    
    auto smem_layout = composition(Swizzle<4, 0, 6>{}, layout);

    auto input_cta_tiler = make_shape(make_shape(make_shape(Int<tile_size>{}, Int<loop_y>{}), Int<1>{}), make_shape(make_shape(Int<tile_size>{}, Int<loop_x>{}), Int<1>{}));
    auto output_cta_tiler = make_shape(make_shape(make_shape(Int<valid_tile_size>{}, Int<loop_y>{}), Int<1>{}), make_shape(make_shape(Int<valid_tile_size>{}, Int<loop_x>{}), Int<1>{}));

    // Get maximum shared memory size
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    size_t shared_memory_size = sizeof(cufftdx::complex<float>) * cosize_v<decltype(smem_layout)>;
    size_t shared_memory_size_config = std::min(4*shared_memory_size, prop.sharedMemPerBlockOptin);
    CHECK_CUDA(cudaFuncSetAttribute(conv_kernel_thread<float, tile_size, f, fft_c2c_forward, fft_c2c_inverse, decltype(input_gmem_layout), decltype(filter_gmem_layout), decltype(output_gmem_layout), decltype(smem_layout), decltype(input_cta_tiler), decltype(output_cta_tiler)>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_memory_size_config)));
    CHECK_CUDA(cudaGetLastError());
    
    auto kernel = [&]() {
        CHECK_CUFFT(cufftExecC2C(plan_forward, reinterpret_cast<cufftComplex*>(d_filter_spatial),
                                 reinterpret_cast<cufftComplex*>(d_filter), CUFFT_FORWARD));
        conv_kernel_thread<float, tile_size, f, fft_c2c_forward, fft_c2c_inverse><<<gridSize, blockSize, shared_memory_size>>>
        (d_input, d_filter, d_output, N, input_gmem_layout, filter_gmem_layout, output_gmem_layout, smem_layout, input_cta_tiler, output_cta_tiler); 
    };
    kernel();
    CHECK_CUDA(cudaGetLastError());
    std::vector<float2> h_output(size_t(output_size) * output_size);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, sizeof(float2) * h_output.size(), cudaMemcpyDeviceToHost));
    std::vector<float> times;
    if (gate(h_output.data())) {
        const unsigned timed = adaptive_runs(kernel, runs);
        // Warm up once before the first trial; later trials follow back to back.
        for (int t = 0; t < trials; ++t)
            times.push_back(measure_execution_ms(kernel, t == 0 ? std::min<unsigned>(warmup_runs, (timed + 2) / 3) : 0, timed));
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter_spatial));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUFFT(cufftDestroy(plan_forward));
    return times;
}

// Template instantiations for odd filter sizes 3 to 59
template std::vector<float> cufftdx_convolution<3>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<5>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<7>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<9>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<11>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<13>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<15>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<17>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<19>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<21>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<23>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<25>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<27>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<29>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<31>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<33>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<35>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<37>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<39>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<41>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<43>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<45>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<47>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<49>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<51>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<53>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<55>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<57>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);
template std::vector<float> cufftdx_convolution<59>(const float2*, const float2*, int, const std::function<bool(const float2*)>&, int, int, int);