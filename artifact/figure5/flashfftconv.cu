// Standalone Figure 5 driver for the unmodified pinned FlashFFTConv 4096 kernel.
// Same launch tuple as upstream's B%4==0,H%8==0 path. No libtorch runtime.
#include "monarch_cuda_16_16_16_kernel.h"
#include "../../sample/utils.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

using C = c10::complex<at::Half>;
constexpr int N = 4096;

// Upstream: reshape(H,256,16).transpose(-1,-2).reshape(H,16,16,16)
// .transpose(-1,-2).reshape(H,4096). Conversion and permutation are timed.
__global__ void filter_permute(const float2* src, C* dst, int count) {
    const int q = blockIdx.x*blockDim.x+threadIdx.x;
    if(q<count) {
        const int p=q%N, a=p/256, c=(p/16)%16, b=p%16;
        const float2 v=src[(q/N)*N+b*256+c*16+a];
        dst[q]=C(at::Half(v.x),at::Half(v.y));
    }
}

int main(int argc,char** argv) {
    if(argc<5) {
        std::cerr<<"usage: "<<argv[0]<<" K[,K...] B D validate_sequences [trials=3] [warmup=5] [runs=30]\n";
        return 2;
    }
    const int B=std::atoi(argv[2]),D=std::atoi(argv[3]),sample=std::atoi(argv[4]);
    const int trials=argc>5?std::atoi(argv[5]):3,warmup=argc>6?std::atoi(argv[6]):5,runs=argc>7?std::atoi(argv[7]):30;
    std::vector<int> filters;
    for(const char* p=argv[1];*p;) { filters.push_back(std::atoi(p));while(*p&&*p!=',')++p;if(*p)++p; }
    if(B<=0||B%4||D<=0||D%8||sample<0||trials<=0||warmup<0||runs<=0||filters.empty()||
       std::any_of(filters.begin(),filters.end(),[](int k){return k<1||k>N;})) return 2;
    // Upstream indexes batch/channel offsets with signed int. Reject oversized
    // CLI shapes before any CUDA call; all artifact profiles fit this bound.
    if(uint64_t(B)*uint64_t(D)>uint64_t(std::numeric_limits<int>::max())/uint64_t(N)) {
        std::cerr<<"B*D*4096 exceeds the upstream kernel's signed-int indexing limit\n";
        return 2;
    }
    cudaDeviceProp prop;CHECK_CUDA(cudaGetDeviceProperties(&prop,0));
    std::cerr<<"GPU="<<prop.name<<" compute="<<prop.major<<"."<<prop.minor<<" batch="<<B*D<<" seed=42\n";
    const size_t count=size_t(B)*D*N, filter_count=size_t(D)*N;
    std::vector<at::Half> x(count);
    std::minstd_rand rng(42);
    for(auto& v:x)v=at::Half(float(rng())/float(rng.max()));
    at::Half *dx=nullptr,*dy=nullptr;
    float2 *dh=nullptr,*df=nullptr;C *dk=nullptr,*dm=nullptr;
    CHECK_CUDA(cudaMalloc(&dx,count*sizeof(at::Half)));CHECK_CUDA(cudaMalloc(&dy,count*sizeof(at::Half)));
    CHECK_CUDA(cudaMalloc(&dh,filter_count*sizeof(float2)));CHECK_CUDA(cudaMalloc(&df,filter_count*sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&dk,filter_count*sizeof(C)));
    CHECK_CUDA(cudaMemcpy(dx,x.data(),count*sizeof(at::Half),cudaMemcpyHostToDevice));
    // DFT16, twiddle16x256/N, twiddle16x16, IDFT16, inverse twiddle16x256,
    // inverse twiddle16x16: normalization is upstream's forward 1/N twiddle.
    std::vector<C> matrices;
    auto append=[&](int rows,int cols,int denom,int sign,float scale) {
        for(int r=0;r<rows;++r)for(int c=0;c<cols;++c) {
            const double angle=sign*2.0*3.14159265358979323846*r*c/denom;
            matrices.emplace_back(at::Half(float(std::cos(angle))*scale),at::Half(float(std::sin(angle))*scale));
        }
    };
    append(16,16,16,-1,1);append(16,256,N,-1,1.0f/N);append(16,16,256,-1,1);
    append(16,16,16,1,1);append(16,256,N,1,1);append(16,16,256,1,1);
    CHECK_CUDA(cudaMalloc(&dm,matrices.size()*sizeof(C)));
    CHECK_CUDA(cudaMemcpy(dm,matrices.data(),matrices.size()*sizeof(C),cudaMemcpyHostToDevice));
    cufftHandle plan;CHECK_CUFFT(cufftPlan1d(&plan,N,CUFFT_C2C,D));
    std::vector<at::Half> actual(count);
    for(int k:filters) {
        std::cout<<"Filter size: "<<k<<"\n";
        std::vector<float2> h(size_t(D)*N,make_float2(0,0));
        std::minstd_rand krng(42+k);
        for(int d=0;d<D;++d)for(int j=0;j<k;++j)h[d*N+j].x=float(krng())/float(krng.max());
        CHECK_CUDA(cudaMemcpy(dh,h.data(),h.size()*sizeof(float2),cudaMemcpyHostToDevice));
        auto run=[&]() {
            CHECK_CUFFT(cufftExecC2C(plan,reinterpret_cast<cufftComplex*>(dh),reinterpret_cast<cufftComplex*>(df),CUFFT_FORWARD));
            filter_permute<<<static_cast<unsigned>((filter_count+255)/256),256>>>(df,dk,static_cast<int>(filter_count));
            monarch_conv_cuda_kernel<32,4,4096,1,16,false,4,8,4><<<dim3(B/4,D/8),dim3(32,4),(2*N+4*256)*sizeof(half)>>>(
                dx,nullptr,dk,dm,dm+256,dm+256+4096,dm+512+4096,dm+768+4096,dm+768+8192,
                dy,nullptr,B,D,N,16);
            CHECK_CUDA(cudaGetLastError());
        };
        CHECK_CUDA(cudaMemset(dy,0xff,count*sizeof(at::Half)));
        run();CHECK_CUDA(cudaMemcpy(actual.data(),dy,count*sizeof(at::Half),cudaMemcpyDeviceToHost));
        const int batch=B*D,checked=sample==0||sample>=batch?batch:sample+1;
        double num=0,den=0;
        for(int c=0;c<checked;++c) {
            const int bd=checked==batch?c:c==checked-1?batch-1:int((int64_t(c)*(batch/sample)+c)%batch);
            for(int n=0;n<N;++n) {
                double ref=0;
                for(int j=0;j<k;++j)ref+=double(float(x[size_t(bd)*N+(n-j+N)%N]))*h[(bd%D)*N+j].x;
                const double error=double(float(actual[size_t(bd)*N+n]))-ref;num+=error*error;den+=ref*ref;
            }
        }
        const double rel=std::sqrt(num/den);
        const bool passed=std::isfinite(rel)&&rel<=5e-3;
        std::cout<<std::setprecision(8)<<"Validation N=4096 vs direct circular conv (FlashFFTConv): L2_rel="<<rel<<"\n"
                 <<"Validation sequences: "<<checked<<" of "<<batch<<"\nCircular convolution validation: "<<(passed?"PASS":"FAIL")<<"\n";
        if(!passed)return 1;
        for(int t=0;t<trials;++t) {
            const float ms=measure_execution_ms(run,t==0?warmup:0,runs);
            std::cout<<"FlashFFTConv (N=4096, include_filter_fft=1, K="<<k<<", trial="<<t<<") time: "<<ms<<" ms\n";
        }
    }
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(dx));CHECK_CUDA(cudaFree(dy));CHECK_CUDA(cudaFree(dh));CHECK_CUDA(cudaFree(df));CHECK_CUDA(cudaFree(dk));CHECK_CUDA(cudaFree(dm));
    return 0;
}
