// Table 4 instruction-throughput reference. This uses the same TF32 MMA
// primitive as the N=64 unit FFT, with independent accumulator chains.
#include <thunderfft/thunderfft.cuh>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kUnroll = 32;
constexpr int kWarpSize = 32;
constexpr std::uint64_t kFlopsPerMma = 2 * 16 * 8 * 8;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

struct Options {
    int iterations = 128;
    int trials = 5;
    int warmups = 2;
    int waves = 1;
    int chains = 8;
    int threads = 128;
};

int parse_integer(const std::string& value, const std::string& option) {
    int number = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error("Invalid integer for " + option + ": " + value);
    }
    return number;
}

void bounded(int value, int low, int high, const char* name) {
    if (value < low || value > high) {
        throw std::runtime_error(std::string(name) + " must be in [" +
                                 std::to_string(low) + ", " +
                                 std::to_string(high) + "]");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + argument);
        const int value = parse_integer(argv[++i], argument);
        if (argument == "--iterations") options.iterations = value;
        else if (argument == "--trials") options.trials = value;
        else if (argument == "--warmups") options.warmups = value;
        else if (argument == "--waves") options.waves = value;
        else if (argument == "--chains") options.chains = value;
        else if (argument == "--threads") options.threads = value;
        else throw std::runtime_error("Unknown option: " + argument);
    }
    bounded(options.iterations, 1, 8192, "iterations");
    bounded(options.trials, 1, 50, "trials");
    bounded(options.warmups, 0, 100, "warmups");
    bounded(options.waves, 1, 16, "waves");
    if (options.chains != 4 && options.chains != 8)
        throw std::runtime_error("chains must be 4 or 8");
    if (options.threads != 128 && options.threads != 256)
        throw std::runtime_error("threads must be 128 or 256");
    // At most 262144 additions of 1/128 per chain, all exactly representable.
    static_assert(8192 * kUnroll < (1 << 24));
    return options;
}

template <int Chains>
__global__ void ideal_mma(float* output, int iterations, float input_a, float input_b) {
    const float a[4] = {input_a, input_a, input_a, input_a};
    const float b[2] = {input_b, input_b};
    float accumulators[Chains][4] = {};

    // A and B arrive through kernel arguments, not compile-time operands.
    // Uniform fragments make every matrix entry equal. With A=B=1/32 each
    // m16n8k8 instruction adds 8*(1/32)*(1/32)=1/128 to each accumulator.
    // The existing primitive uses volatile inline PTX, and every accumulator
    // is stored and checked on the host after the timed region.
    #pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration) {
        #pragma unroll
        for (int repeat = 0; repeat < kUnroll; ++repeat) {
            #pragma unroll
            for (int chain = 0; chain < Chains; ++chain) {
                thunderfft::unit::mma_m16n8k8_tf32_f32_rowcol(
                    accumulators[chain], a, b, accumulators[chain]);
            }
        }
    }

    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t thread_count = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    #pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            // Adjacent lanes write adjacent values for every fragment entry.
            output[(chain * 4 + element) * thread_count + thread] = accumulators[chain][element];
        }
    }
}

struct DeviceOutput {
    float* data = nullptr;
    explicit DeviceOutput(std::size_t count) {
        check_cuda(cudaMalloc(&data, count * sizeof(float)), "cudaMalloc output");
    }
    ~DeviceOutput() { if (data) cudaFree(data); }
    DeviceOutput(const DeviceOutput&) = delete;
    DeviceOutput& operator=(const DeviceOutput&) = delete;
};

struct Event {
    cudaEvent_t value = nullptr;
    Event() { check_cuda(cudaEventCreate(&value), "cudaEventCreate"); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

double validate(const DeviceOutput& device, std::vector<float>& host, int iterations) {
    check_cuda(cudaMemcpy(host.data(), device.data, host.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "cudaMemcpy output");
    const float expected = static_cast<float>(iterations * kUnroll) / 128.0f;
    double maximum = 0.0;
    for (std::size_t index = 0; index < host.size(); ++index) {
        const float actual = host[index];
        if (!std::isfinite(actual) || actual != expected) {
            std::ostringstream message;
            message << "MMA validation failed at output " << index << ": actual="
                    << std::setprecision(10) << actual << ", expected=" << expected;
            throw std::runtime_error(message.str());
        }
        maximum = std::max(maximum, std::abs(static_cast<double>(actual) - expected));
    }
    return maximum;
}

std::string gpu_uuid(const cudaUUID_t& uuid) {
    std::ostringstream formatted;
    formatted << "GPU-" << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) formatted << '-';
        formatted << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[index]));
    }
    return formatted.str();
}

struct Trial {
    double time_us;
    double max_abs_error;
};

template <int Chains>
void run(const Options& options, const cudaDeviceProp& properties) {
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes, ideal_mma<Chains>), "cudaFuncGetAttributes");
    int active_blocks = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks, ideal_mma<Chains>, options.threads, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    if (active_blocks < 1) throw std::runtime_error("Kernel has zero active blocks per SM");
    const int blocks = properties.multiProcessorCount * active_blocks * options.waves;
    const std::size_t output_count = static_cast<std::size_t>(blocks) * options.threads * Chains * 4;
    DeviceOutput output(output_count);
    std::vector<float> host(output_count);
    Event start, stop;

    auto launch = [&](int iterations) {
        ideal_mma<Chains><<<blocks, options.threads>>>(output.data, iterations, 1.0f / 32.0f, 1.0f / 32.0f);
        check_cuda(cudaGetLastError(), "ideal_mma launch");
    };

    // A short numerical check precedes warmups and all timings.
    launch(1);
    validate(output, host, 1);
    for (int warmup = 0; warmup < options.warmups; ++warmup) launch(options.iterations);
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
    if (options.warmups > 0) validate(output, host, options.iterations);

    std::vector<Trial> trials;
    for (int trial = 0; trial < options.trials; ++trial) {
        check_cuda(cudaEventRecord(start.value), "cudaEventRecord start");
        launch(options.iterations);
        check_cuda(cudaEventRecord(stop.value), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop.value), "cudaEventSynchronize stop");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start.value, stop.value), "cudaEventElapsedTime");
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f)
            throw std::runtime_error("CUDA event duration is non-positive or non-finite");
        const double error = validate(output, host, options.iterations);
        trials.push_back({static_cast<double>(elapsed_ms) * 1000.0, error});
    }

    const std::uint64_t mma_instructions = static_cast<std::uint64_t>(blocks) *
        (options.threads / kWarpSize) * Chains * options.iterations * kUnroll;
    const std::uint64_t flop_count = mma_instructions * kFlopsPerMma;
    // Emit no partial success rows if any trial failed numerical validation.
    std::cout << "trial,blocks,threads,chains,iterations,unroll,mma_instructions,flop_count,"
                 "time_us,tflops,max_abs_error,passed,registers,shared_bytes,active_blocks_per_sm\n";
    std::cout << std::setprecision(12);
    for (std::size_t trial = 0; trial < trials.size(); ++trial) {
        const Trial& result = trials[trial];
        const double tflops = static_cast<double>(flop_count) / (result.time_us * 1000000.0);
        std::cout << trial << ',' << blocks << ',' << options.threads << ',' << Chains << ','
                  << options.iterations << ',' << kUnroll << ',' << mma_instructions << ','
                  << flop_count << ',' << result.time_us << ',' << tflops << ','
                  << result.max_abs_error << ",true," << attributes.numRegs << ','
                  << attributes.sharedSizeBytes << ',' << active_blocks << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cerr << "Usage: thff_ideal_mma [--iterations 1..8192] [--trials 1..50] "
                         "[--warmups 0..100] [--waves 1..16] [--chains 4|8] [--threads 128|256]\n";
            return 0;
        }
        const Options options = parse_options(argc, argv);
        check_cuda(cudaSetDevice(0), "cudaSetDevice 0");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        if (properties.major < 8)
            throw std::runtime_error("TF32 m16n8k8 MMA requires compute capability 8.0 or newer");
        int runtime_version = 0, driver_version = 0;
        check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
        check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
        std::cerr << "GPU=" << properties.name << " compute=" << properties.major << '.'
                  << properties.minor << " sm_count=" << properties.multiProcessorCount << '\n';
        std::cerr << "GPU_UUID=" << gpu_uuid(properties.uuid) << '\n';
        std::cerr << "compiled_CUDART=" << CUDART_VERSION << " runtime=" << runtime_version
                  << " driver=" << driver_version << '\n';
        if (options.chains == 4) run<4>(options, properties);
        else run<8>(options, properties);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
