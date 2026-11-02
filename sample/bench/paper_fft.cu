// Timing adapter for the sample/bench FFT kernels.
// Paper Table 3 uses the differential compute time;
// the out-of-place, single-FFT end-to-end time is reported separately.
#include "thunderfft_bench.h"
#include <cufft.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_OK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) \
  throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(e)); } while (0)
#define FFT_OK(expr) do { cufftResult e = (expr); if (e != CUFFT_SUCCESS) \
  throw std::runtime_error(std::string(#expr) + ": cuFFT " + std::to_string(int(e))); } while (0)

template<class T> __device__ vec2_t<T> make_value(float x, float y) {
  if constexpr (std::is_same_v<T,half>) return __floats2half2_rn(x,y);
  else return make_float2(x,y);
}
template<class T> __device__ double2 as_double(vec2_t<T> v) {
  if constexpr (std::is_same_v<T,half>) return make_double2(__half2float(v.x), __half2float(v.y));
  else return make_double2(v.x,v.y);
}
template<class T> __global__ void initialize(vec2_t<T>* input,
                                           double2* reference, size_t count, int n) {
  for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<count; i+=gridDim.x*blockDim.x) {
    unsigned z = unsigned(i)*1664525u+1013904223u;
    float x = (float(int(z & 65535u))-32768.0f)/32768.0f/std::sqrt(float(n));
    z = z*1664525u+1013904223u;
    float y = (float(int(z & 65535u))-32768.0f)/32768.0f/std::sqrt(float(n));
    // Keep the separate accuracy input complex; timed inputs follow the
    // original benchmark's real sinusoid divided by sqrt(N).
    if (!reference) {
      x = sinf(float(6.2831853071795864769 * double(i % n) / n)) / sqrtf(float(n));
      y = 0.0f;
    }
    auto value=make_value<T>(x,y); input[i]=value;
    if (reference) reference[i]=as_double<T>(value);
  }
}

template<class L> constexpr size_t layout_elements() {
  constexpr size_t last = size_t(L::N-1)*L::elem_stride + size_t(L::batch-1)*L::batch_stride;
  return last + (last/L::pad_period)*L::pad + 1;
}

template<class T,int N> constexpr size_t shared_bytes() {
  constexpr int b=thunderfft::batch_per_block<N>;
  using L=thunderfft::bench_layout<T,N,b>;
  constexpr size_t original=size_t(N+pad_h(N))*b;
  constexpr size_t in=layout_elements<typename L::L_in>();
  constexpr size_t out=layout_elements<typename L::L_out>();
  // Use the same safe caller workspace for both parent and corrected headers;
  // older pad_h(4096) underallocated the FP16 internal skew by seven elements.
  constexpr size_t transpose = N == 4096 ? size_t(N + N/16 + 32)*b : 0;
  return sizeof(vec2_t<T>)*std::max(std::max(original,transpose),std::max(in,out));
}

template<class Launch> double elapsed_us(Launch launch, int runs) {
  cudaEvent_t begin,end;
  CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end));
  CUDA_OK(cudaEventRecord(begin));
  for (int i=0;i<runs;++i) launch();
  CUDA_OK(cudaEventRecord(end)); CUDA_OK(cudaEventSynchronize(end));
  float ms=0; CUDA_OK(cudaEventElapsedTime(&ms,begin,end));
  CUDA_OK(cudaEventDestroy(begin)); CUDA_OK(cudaEventDestroy(end));
  return double(ms)*1000.0/runs;
}

template<class T,int N,bool Shared> void run(int batch,int repeats,int trials) {
  using V=vec2_t<T>;
  constexpr int bpb=thunderfft::batch_per_block<N>;
  constexpr int wpb=thunderfft::warp_per_block<N>;
  constexpr size_t smem=shared_bytes<T,N>();
  if (batch<=0 || batch%bpb) throw std::runtime_error("batch must be a positive BPB multiple");
  auto attributes=[&] {
    if constexpr (Shared) {
      CUDA_OK(cudaFuncSetAttribute(thunderfft::ThunderFFT_benchmark_smem<T,N,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
      CUDA_OK(cudaFuncSetAttribute(thunderfft::ThunderFFT_benchmark_smem_e2e<T,N,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    } else {
      CUDA_OK(cudaFuncSetAttribute(thunderfft::ThunderFFT_benchmark_reg<T,N,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
      CUDA_OK(cudaFuncSetAttribute(thunderfft::ThunderFFT_benchmark_reg_e2e<T,N,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    }
  };
  thunderfft::ThunderFFTInitialize<T>(N); attributes();
  auto launch=[&](V* input,V* output,int count,int r,bool e2e) {
    dim3 grid(count/bpb),block(32,wpb);
    if constexpr (Shared) {
      if(e2e) thunderfft::ThunderFFT_benchmark_smem_e2e<T,N,true><<<grid,block,smem>>>(input,output,nullptr);
      else thunderfft::ThunderFFT_benchmark_smem<T,N,true><<<grid,block,smem>>>(input,output,nullptr,r);
    } else {
      if(e2e) thunderfft::ThunderFFT_benchmark_reg_e2e<T,N,true><<<grid,block,smem>>>(input,output,nullptr);
      else thunderfft::ThunderFFT_benchmark_reg<T,N,true><<<grid,block,smem>>>(input,output,nullptr,r);
    }
    CUDA_OK(cudaGetLastError());
  };
  // A separately allocated, unmutated validation input. Both precisions are
  // compared against cuFFT FP64 on the exact rounded input they receive.
  constexpr int validation_batch=64;
  constexpr size_t validation_count=size_t(N)*validation_batch;
  V *vi=nullptr,*vo=nullptr; double2 *vr=nullptr;
  CUDA_OK(cudaMalloc(&vi,validation_count*sizeof(V)));
  CUDA_OK(cudaMalloc(&vo,validation_count*sizeof(V)));
  CUDA_OK(cudaMalloc(&vr,validation_count*sizeof(double2)));
  initialize<T><<<128,256>>>(vi,vr,validation_count,N); CUDA_OK(cudaGetLastError());
  cufftHandle plan; FFT_OK(cufftPlan1d(&plan,N,CUFFT_Z2Z,validation_batch));
  FFT_OK(cufftExecZ2Z(plan,reinterpret_cast<cufftDoubleComplex*>(vr),reinterpret_cast<cufftDoubleComplex*>(vr),CUFFT_FORWARD));
  launch(vi,vo,validation_batch,1,false); CUDA_OK(cudaDeviceSynchronize());
  std::vector<V> actual(validation_count); std::vector<double2> reference(validation_count);
  CUDA_OK(cudaMemcpy(actual.data(),vo,validation_count*sizeof(V),cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(reference.data(),vr,validation_count*sizeof(double2),cudaMemcpyDeviceToHost));
  double se=0,sr=0; bool finite=true;
  for(size_t i=0;i<validation_count;++i) {
    double x,y;
    if constexpr(std::is_same_v<T,half>) {x=__half2float(actual[i].x);y=__half2float(actual[i].y);}
    else {x=actual[i].x;y=actual[i].y;}
    finite=finite && std::isfinite(x) && std::isfinite(y);
    se+=(x-reference[i].x)*(x-reference[i].x)+(y-reference[i].y)*(y-reference[i].y);
    sr+=reference[i].x*reference[i].x+reference[i].y*reference[i].y;
  }
  double rel=std::sqrt(se/sr);
  FFT_OK(cufftDestroy(plan)); CUDA_OK(cudaFree(vi)); CUDA_OK(cudaFree(vo)); CUDA_OK(cudaFree(vr));
  if(!finite || !std::isfinite(rel) || rel>5e-3) throw std::runtime_error("single-FFT correctness gate failed relL2="+std::to_string(rel));
  V *input=nullptr,*output=nullptr; size_t bytes=size_t(N)*batch*sizeof(V);
  CUDA_OK(cudaMalloc(&input,bytes)); CUDA_OK(cudaMalloc(&output,bytes));
  initialize<T><<<1024,256>>>(input,nullptr,size_t(N)*batch,N); CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  // Stabilize clocks before collecting the R/2R samples. This warmup is
  // outside every reported interval; record it separately from the paper data.
  auto warmup=[&]{launch(input,output,batch,repeats,false);};
  for(int w=0;w<3;++w) {warmup(); CUDA_OK(cudaDeviceSynchronize());}
  for(int t=0;t<trials;++t) {
    auto repeated=[&](int r) {return [&,r]{launch(input,output,batch,r,false);};};
    repeated(repeats)(); CUDA_OK(cudaDeviceSynchronize());
    double t_r=elapsed_us(repeated(repeats),1);
    repeated(2*repeats)(); CUDA_OK(cudaDeviceSynchronize());
    double t_2r=elapsed_us(repeated(2*repeats),1);
    if (!(std::isfinite(t_r) && std::isfinite(t_2r) && t_2r > t_r))
      throw std::runtime_error("nonpositive/nonfinite differential timing");
    auto once=[&]{launch(input,output,batch,1,true);};
    once(); CUDA_OK(cudaDeviceSynchronize()); double e2e=elapsed_us(once,10);
    std::printf("%d,%s,%s,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.12g,%zu\n",N,
      (std::is_same_v<T, half> ? "fp16" : "tf32"),Shared?"shared":"register",batch,repeats,t,
      t_r,t_2r,(t_2r-t_r)/repeats,e2e,rel,smem);std::fflush(stdout);
  }
  CUDA_OK(cudaFree(input)); CUDA_OK(cudaFree(output)); thunderfft::ThunderFFTFinalize<T>();
}

template<int N> void choose(const std::string& precision,const std::string& placement,int b,int r,int t) {
  if(precision=="fp16") {
    if(placement=="shared") run<half,N,true>(b,r,t); else run<half,N,false>(b,r,t);
  } else {
    if(placement=="shared") run<float,N,true>(b,r,t); else run<float,N,false>(b,r,t);
  }
}
int main(int argc,char**argv) {
  try {
    if(argc<4 || argc>7) {std::fprintf(stderr,"usage: %s N fp16|tf32 register|shared [repeats=1000] [trials=3] [batch=65536]\n",argv[0]);return 2;}
    int n=std::stoi(argv[1]),r=argc>4?std::stoi(argv[4]):1000,t=argc>5?std::stoi(argv[5]):3,b=argc>6?std::stoi(argv[6]):65536;
    std::string precision=argv[2],placement=argv[3];
    if((precision!="fp16"&&precision!="tf32")||(placement!="register"&&placement!="shared")||r<=0||t<=0)throw std::runtime_error("invalid arguments");
    cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop,0));
    int driver=0,runtime=0,cufft_version=0;
    CUDA_OK(cudaDriverGetVersion(&driver)); CUDA_OK(cudaRuntimeGetVersion(&runtime));
    FFT_OK(cufftGetVersion(&cufft_version));
    std::fprintf(stderr,"compiled_CUDART=%d runtime=%d driver=%d cuFFT=%d extra_R_warmups=3\n",
                 CUDART_VERSION,runtime,driver,cufft_version);
    std::fprintf(stderr,"GPU=%s compute=%d.%d batch=%d R=%d trials=%d single_fft_relL2_limit=0.005\n",prop.name,prop.major,prop.minor,b,r,t);
    std::puts("N,precision,placement,batch,inside_repeats,trial,t_R_us,t_2R_us,compute_us,e2e_out_of_place_us,single_fft_relL2,shared_bytes");
    switch(n) {
      case 64:choose<64>(precision,placement,b,r,t);break;
      case 128:choose<128>(precision,placement,b,r,t);break;
      case 256:choose<256>(precision,placement,b,r,t);break;
      case 512:choose<512>(precision,placement,b,r,t);break;
      case 1024:choose<1024>(precision,placement,b,r,t);break;
      case 2048:choose<2048>(precision,placement,b,r,t);break;
      case 4096:choose<4096>(precision,placement,b,r,t);break;
      default:throw std::runtime_error("unsupported N");
    }
    return 0;
  } catch(const std::exception& e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
}
