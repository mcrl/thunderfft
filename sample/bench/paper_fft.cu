// Timing adapter for the sample/bench FFT kernels.
// Paper Table 3 uses the differential compute time;
// the out-of-place, single-FFT end-to-end time is reported separately.
// With --library cufftdx (built as thff_paper_fft_cufftdx), cuFFTDx block FFTs
// run through the same input, FP64 correctness gate and R/2R timing.
#include "thunderfft_bench.h"
#ifdef THFF_WITH_CUFFTDX
#include "cufftdx_bench.h"
#endif
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

// Shared correctness gate and timing for one kernel configuration.
// launch(input, output, count, inside_repeats, e2e) enqueues one forward FFT kernel.
template<class T,int N,class Launch>
void measure(int batch,int repeats,int trials,int granularity,size_t smem,const char* precision,
             const char* placement,const std::string& suffix,Launch launch) {
  using V=vec2_t<T>;
  if (batch<=0 || batch%granularity) throw std::runtime_error("batch must be a positive FFTs-per-block multiple");
  // A separately allocated, unmutated validation input. Both precisions are
  // compared against cuFFT FP64 on the exact rounded input they receive.
  // 64 FFTs, or one full launch block when that is larger (cuFFTDx thread FFTs).
  const int validation_batch=std::max(64,granularity);
  if (validation_batch%granularity) throw std::runtime_error("validation batch is not an FFTs-per-block multiple");
  const size_t validation_count=size_t(N)*validation_batch;
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
    std::printf("%d,%s,%s,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.12g,%zu%s\n",N,precision,placement,batch,repeats,t,
      t_r,t_2r,(t_2r-t_r)/repeats,e2e,rel,smem,suffix.c_str());std::fflush(stdout);
  }
  CUDA_OK(cudaFree(input)); CUDA_OK(cudaFree(output));
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
  const char* precision=std::is_same_v<T,half> ? "fp16" : "tf32";
  measure<T,N>(batch,repeats,trials,bpb,smem,precision,Shared?"shared":"register","",launch);
  thunderfft::ThunderFFTFinalize<T>();
}

struct Options { std::string library="thunderfft",candidate; };

#ifdef THFF_WITH_CUFFTDX
// Every compiled cuFFTDx candidate (or the --candidate match) is gated and
// timed separately; e2e launches the same kernel with one inside repeat.
template<class T,int N,bool Shared> void run_cufftdx(int batch,int repeats,int trials,const Options& o) {
  using V=vec2_t<T>;
  int matched=0;
  thff_cufftdx::for_each_candidate<V,N,Shared>([&](const thff_cufftdx::Candidate<V>& c) {
    if(!o.candidate.empty() && o.candidate!=c.label) return;
    ++matched;
    CUDA_OK(cudaFuncSetAttribute(c.kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(c.shared)));
    auto launch=[&](V* input,V* output,int count,int r,bool e2e) {
      c.kernel<<<count/c.ffts_per_launch_block,c.block,c.shared>>>(input,output,count,e2e?1:r);
      CUDA_OK(cudaGetLastError());
    };
    std::fprintf(stderr,"cuFFTDx N=%d candidate=%s FPB=%d EPT=%d block=%ux%ux%u shared=%zu\n",
                 N,c.label.c_str(),c.ffts_per_block,c.elements_per_thread,c.block.x,c.block.y,c.block.z,c.shared);
    const char* precision=std::is_same_v<T,half> ? "fp16" : "fp32";
    const std::string suffix=",cufftdx,"+c.label+","+std::to_string(c.ffts_per_block)+","+
      std::to_string(c.elements_per_thread)+","+std::to_string(c.block.x*c.block.y*c.block.z);
    measure<T,N>(batch,repeats,trials,c.ffts_per_launch_block,c.shared,precision,
                 Shared?"shared":"register",suffix,launch);
  });
  if(!matched) throw std::runtime_error("no compiled cuFFTDx candidate matches --candidate");
}
#endif

template<int N> void choose(const std::string& precision,const std::string& placement,int b,int r,int t,
                            const Options& o) {
  if(o.library=="cufftdx") {
#ifdef THFF_WITH_CUFFTDX
    if(precision=="fp16") {
      if(placement=="shared") run_cufftdx<half,N,true>(b,r,t,o); else run_cufftdx<half,N,false>(b,r,t,o);
    } else {
      if(placement=="shared") run_cufftdx<float,N,true>(b,r,t,o); else run_cufftdx<float,N,false>(b,r,t,o);
    }
    return;
#else
    throw std::runtime_error("built without cuFFTDx; use thff_paper_fft_cufftdx (configure with MATHDX_ROOT)");
#endif
  }
  if(precision=="fp16") {
    if(placement=="shared") run<half,N,true>(b,r,t); else run<half,N,false>(b,r,t);
  } else {
    if(placement=="shared") run<float,N,true>(b,r,t); else run<float,N,false>(b,r,t);
  }
}
int main(int argc,char**argv) {
  try {
    Options o; std::vector<std::string> args;
    for(int i=1;i<argc;++i) {
      std::string a=argv[i];
      if((a=="--library"||a=="--candidate") && i+1<argc) {
        std::string v=argv[++i];
        if(a=="--library") o.library=v; else o.candidate=v;
      } else args.push_back(a);
    }
    if(args.size()<3 || args.size()>6) {std::fprintf(stderr,"usage: %s [--library thunderfft|cufftdx] [--candidate LABEL] N fp16|tf32|fp32 register|shared [repeats=1000] [trials=3] [batch=65536]\n"
      "  thunderfft: fp16|tf32; cufftdx: fp16|fp32 (FP32 CUDA-core cuFFTDx)\n"
      "  LABEL: block_default, block_f<FPB>_e<EPT> or thread_t<threads> (default: all)\n",argv[0]);return 2;}
    int n=std::stoi(args[0]),r=args.size()>3?std::stoi(args[3]):1000,t=args.size()>4?std::stoi(args[4]):3,b=args.size()>5?std::stoi(args[5]):65536;
    std::string precision=args[1],placement=args[2];
    const bool dx=o.library=="cufftdx";
    if((o.library!="thunderfft"&&!dx)||(precision!="fp16"&&precision!=(dx?"fp32":"tf32"))||
       (placement!="register"&&placement!="shared")||r<=0||t<=0||(!dx&&!o.candidate.empty()))
      throw std::runtime_error("invalid arguments");
#ifndef THFF_WITH_CUFFTDX
    if(dx) throw std::runtime_error("built without cuFFTDx; use thff_paper_fft_cufftdx (configure with MATHDX_ROOT)");
#endif
    cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop,0));
    int driver=0,runtime=0,cufft_version=0;
    CUDA_OK(cudaDriverGetVersion(&driver)); CUDA_OK(cudaRuntimeGetVersion(&runtime));
    FFT_OK(cufftGetVersion(&cufft_version));
    std::fprintf(stderr,"compiled_CUDART=%d runtime=%d driver=%d cuFFT=%d extra_R_warmups=3\n",
                 CUDART_VERSION,runtime,driver,cufft_version);
    std::fprintf(stderr,"GPU=%s compute=%d.%d batch=%d R=%d trials=%d single_fft_relL2_limit=0.005\n",prop.name,prop.major,prop.minor,b,r,t);
    // CUDA ordinals need not match nvidia-smi indices; retain the actual device UUID.
    std::fputs("GPU_UUID=GPU-", stderr);
    for (int i=0; i<16; ++i) {
      if (i==4 || i==6 || i==8 || i==10) std::fputc('-', stderr);
      std::fprintf(stderr,"%02x", static_cast<unsigned char>(prop.uuid.bytes[i]));
    }
    std::fputc('\n', stderr);
#ifdef THFF_WITH_CUFFTDX
    if(dx) {
      std::fprintf(stderr,"library=cufftdx cuFFTDx=%d target_SM=%d\n",int(CUFFTDX_VERSION),int(THFF_CUFFTDX_SM));
      if(prop.major*100+prop.minor*10!=THFF_CUFFTDX_SM) throw std::runtime_error("device SM does not match the cuFFTDx build target");
    }
#endif
    std::printf("N,precision,placement,batch,inside_repeats,trial,t_R_us,t_2R_us,compute_us,e2e_out_of_place_us,single_fft_relL2,shared_bytes%s\n",
                dx?",library,candidate,ffts_per_block,elements_per_thread,threads_per_block":"");
    switch(n) {
      case 64:choose<64>(precision,placement,b,r,t,o);break;
      case 128:choose<128>(precision,placement,b,r,t,o);break;
      case 256:choose<256>(precision,placement,b,r,t,o);break;
      case 512:choose<512>(precision,placement,b,r,t,o);break;
      case 1024:choose<1024>(precision,placement,b,r,t,o);break;
      case 2048:choose<2048>(precision,placement,b,r,t,o);break;
      case 4096:choose<4096>(precision,placement,b,r,t,o);break;
      default:throw std::runtime_error("unsupported N");
    }
    return 0;
  } catch(const std::exception& e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
}
