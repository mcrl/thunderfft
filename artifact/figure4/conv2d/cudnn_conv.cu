// cuDNN FP32 NCHW 1x1xNxN CUDNN_CROSS_CORRELATION; the fastest of the top three heuristic algorithms.
#include "common.cuh"

#include <cudnn.h>

#include <sstream>

namespace figure4 {
namespace {

void check_cudnn(cudnnStatus_t status, const char* expression, int line) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(expression) + ": " + cudnnGetErrorString(status) + " (cudnn_conv.cu:" +
                                 std::to_string(line) + ")");
}
#define FIG4_CUDNN(expression) check_cudnn((expression), #expression, __LINE__)

class CudnnMethod final : public Method {
public:
    explicit CudnnMethod(const Problem& p) : problem_(p), output_(size_t(p.M) * p.M) {
        if (cudnnGetVersion() != CUDNN_VERSION)
            throw std::runtime_error("cuDNN runtime " + std::to_string(cudnnGetVersion()) + " differs from headers " +
                                     std::to_string(CUDNN_VERSION));
        FIG4_CUDNN(cudnnCreate(&handle_));
        FIG4_CUDNN(cudnnSetStream(handle_, p.stream));
        FIG4_CUDNN(cudnnCreateTensorDescriptor(&input_desc_));
        FIG4_CUDNN(cudnnCreateTensorDescriptor(&output_desc_));
        FIG4_CUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
        FIG4_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
        FIG4_CUDNN(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, p.N, p.N));
        FIG4_CUDNN(cudnnSetFilter4dDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, p.f, p.f));
        FIG4_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc_, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                                   CUDNN_DATA_FLOAT));
        FIG4_CUDNN(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, p.M, p.M));

        cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        int found = 0;
        FIG4_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle_, input_desc_, filter_desc_, conv_desc_, output_desc_,
                                                          CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &found, perf));
        cudaEvent_t start, stop;
        FIG4_CUDA(cudaEventCreate(&start));
        FIG4_CUDA(cudaEventCreate(&stop));
        float best_ms = -1.0f;
        std::ostringstream tried;
        for (int k = 0, count = 0; k < found && count < 3; ++k) {
            if (perf[k].status != CUDNN_STATUS_SUCCESS) continue;
            ++count;
            size_t bytes = 0;
            FIG4_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle_, input_desc_, filter_desc_, conv_desc_,
                                                               output_desc_, perf[k].algo, &bytes));
            void* buffer = nullptr;
            if (bytes) FIG4_CUDA(cudaMalloc(&buffer, bytes));
            forward(perf[k].algo, buffer, bytes);
            FIG4_CUDA(cudaEventRecord(start, p.stream));
            forward(perf[k].algo, buffer, bytes);
            FIG4_CUDA(cudaEventRecord(stop, p.stream));
            FIG4_CUDA(cudaEventSynchronize(stop));
            float ms = 0.0f;
            FIG4_CUDA(cudaEventElapsedTime(&ms, start, stop));
            tried << (count > 1 ? " " : "") << int(perf[k].algo) << ":" << ms << "ms";
            if (best_ms < 0.0f || ms < best_ms) {
                if (workspace_) FIG4_CUDA(cudaFree(workspace_));
                best_ms = ms, algo_ = perf[k].algo, workspace_ = buffer, workspace_bytes_ = bytes;
            } else if (buffer) {
                FIG4_CUDA(cudaFree(buffer));
            }
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (best_ms < 0.0f) throw std::runtime_error("cuDNN heuristics returned no supported algorithm");
        tried_ = tried.str();
    }
    ~CudnnMethod() override {
        if (workspace_) cudaFree(workspace_);
        cudnnDestroyConvolutionDescriptor(conv_desc_);
        cudnnDestroyFilterDescriptor(filter_desc_);
        cudnnDestroyTensorDescriptor(output_desc_);
        cudnnDestroyTensorDescriptor(input_desc_);
        cudnnDestroy(handle_);
    }
    void run() override { forward(algo_, workspace_, workspace_bytes_); }
    const float* output() const override { return output_.data; }
    size_t output_pitch() const override { return size_t(problem_.M); }
    bool filter_fft_included() const override { return false; }
    std::string details() const override {
        std::ostringstream out;
        out << "cuDNN " << cudnnGetVersion() << " algorithm: " << int(algo_) << " workspace_bytes=" << workspace_bytes_
            << " candidates: " << tried_;
        return out.str();
    }

private:
    void forward(cudnnConvolutionFwdAlgo_t algo, void* workspace, size_t bytes) {
        const float alpha = 1.0f, beta = 0.0f;
        FIG4_CUDNN(cudnnConvolutionForward(handle_, &alpha, input_desc_, problem_.d_input, filter_desc_,
                                           problem_.d_filter, conv_desc_, algo, workspace, bytes, &beta, output_desc_,
                                           output_.data));
    }

    Problem problem_;
    DeviceBuffer<float> output_;
    cudnnHandle_t handle_ = nullptr;
    cudnnTensorDescriptor_t input_desc_ = nullptr, output_desc_ = nullptr;
    cudnnFilterDescriptor_t filter_desc_ = nullptr;
    cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
    cudnnConvolutionFwdAlgo_t algo_{};
    std::string tried_;
    void* workspace_ = nullptr;
    size_t workspace_bytes_ = 0;
};

}  // namespace

std::unique_ptr<Method> make_cudnn(const Problem& p) { return std::make_unique<CudnnMethod>(p); }

}  // namespace figure4
