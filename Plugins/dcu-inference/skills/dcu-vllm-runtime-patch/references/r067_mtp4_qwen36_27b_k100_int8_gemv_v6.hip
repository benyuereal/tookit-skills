#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

namespace {
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
union WaveBufferResource { i32x4 content; struct { uint64_t address; uint32_t range; uint32_t config; } fields; };
__device__ __forceinline__ i32x4 make_buffer_resource(const void* p,uint32_t bytes){
 WaveBufferResource r{};uint64_t a=(uint64_t)p;
 r.content[0]=__builtin_amdgcn_readfirstlane((uint32_t)a);r.content[1]=__builtin_amdgcn_readfirstlane((uint32_t)(a>>32));
 r.content[2]=__builtin_amdgcn_readfirstlane(bytes);r.content[3]=0x00020000;return r.content;
}
__device__ __forceinline__ i32x4 buf_load_x4(i32x4 r,uint32_t off){
 i32x4 v;asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen;\n\t":"=v"(v):"v"(off),"s"(r):"memory");return v;
}
__device__ __forceinline__ void wait_v2(){asm volatile("s_waitcnt vmcnt(2)":::"memory");}
__device__ __forceinline__ void wait_v0(){asm volatile("s_waitcnt vmcnt(0)":::"memory");}

template<int K_DIM,int WAVES,int MODE>
__global__ __launch_bounds__(WAVES*64) void gemv_v6_kernel(const int8_t* __restrict__ x,const int8_t* __restrict__ w,
 const float* __restrict__ xs,const float* __restrict__ ws,hip_bfloat16* __restrict__ out,int n_out){
 const int lane=threadIdx.x&63,wave=threadIdx.x>>6,n=blockIdx.x*WAVES+wave;if(n>=n_out)return;
 constexpr int NSTEPS=K_DIM/(64*16);static_assert(K_DIM%(64*16)==0);
 const i32x4 xr=make_buffer_resource(x,K_DIM);
 const int8_t* wrp=w+(int64_t)n*K_DIM;const i32x4 wr=make_buffer_resource(wrp,K_DIM);
 int acc0=0,acc1=0;i32x4 xc{},wc{},xn{},wn{};
 uint32_t off=uint32_t(lane*16);xc=buf_load_x4(xr,off);wc=buf_load_x4(wr,off);wait_v0();
#pragma unroll
 for(int s=0;s<NSTEPS;++s){
  if(s+1<NSTEPS){
   uint32_t noff=uint32_t((lane+(s+1)*64)*16);xn=buf_load_x4(xr,noff);wn=buf_load_x4(wr,noff);wait_v2();
  } else {
   // The final current pair was prefetched by the previous iteration. There
   // is no next pair to keep outstanding, so drain VMEM before consuming it.
   wait_v0();
  }
  if constexpr(MODE==0){
   acc0=__builtin_amdgcn_sdot4((int)xc[0],(int)wc[0],acc0,false);acc1=__builtin_amdgcn_sdot4((int)xc[1],(int)wc[1],acc1,false);
   acc0=__builtin_amdgcn_sdot4((int)xc[2],(int)wc[2],acc0,false);acc1=__builtin_amdgcn_sdot4((int)xc[3],(int)wc[3],acc1,false);
  } else {
   int &a=(s&1)?acc1:acc0;a=__builtin_amdgcn_sdot4((int)xc[0],(int)wc[0],a,false);a=__builtin_amdgcn_sdot4((int)xc[1],(int)wc[1],a,false);
   a=__builtin_amdgcn_sdot4((int)xc[2],(int)wc[2],a,false);a=__builtin_amdgcn_sdot4((int)xc[3],(int)wc[3],a,false);
  }
  if(s+1<NSTEPS){xc=xn;wc=wn;}
 }
 wait_v0();int acc=acc0+acc1;
#pragma unroll
 for(int sh=32;sh>0;sh>>=1)acc+=__shfl_down(acc,sh,64);
 if(lane==0)out[n]=hip_bfloat16((float)acc*xs[0]*ws[n]);
}
template<int K,int W,int M>void launch(const int8_t*x,const int8_t*w,const float*xs,const float*ws,hip_bfloat16*out,int n,hipStream_t s){dim3 b(W*64),g((n+W-1)/W);hipLaunchKernelGGL((gemv_v6_kernel<K,W,M>),g,b,0,s,x,w,xs,ws,out,n);}
void validate(const torch::Tensor&x,const torch::Tensor&w,const torch::Tensor&xs,const torch::Tensor&ws){TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&x.scalar_type()==torch::kInt8&&w.scalar_type()==torch::kInt8,"int8 cuda");TORCH_CHECK(x.dim()==2&&x.size(0)==1&&x.is_contiguous()&&w.is_contiguous(),"contiguous");TORCH_CHECK(xs.scalar_type()==torch::kFloat32&&ws.scalar_type()==torch::kFloat32,"scale f32");}
}
torch::Tensor gemv(torch::Tensor x,torch::Tensor w,torch::Tensor xs,torch::Tensor ws,int64_t waves,int64_t mode){validate(x,w,xs,ws);auto o=torch::empty({1,w.size(0)},x.options().dtype(torch::kBFloat16));int n=w.size(0),k=w.size(1);auto*op=(hip_bfloat16*)o.data_ptr<at::BFloat16>();hipStream_t s=c10::hip::getCurrentHIPStream();
#define GM(K,W,M) launch<K,W,M>(x.data_ptr<int8_t>(),w.data_ptr<int8_t>(),xs.data_ptr<float>(),ws.data_ptr<float>(),op,n,s)
#define GW(K,W) do{if(mode==0)GM(K,W,0);else GM(K,W,1);}while(0)
#define GK(K) do{if(waves==1)GW(K,1);else if(waves==2)GW(K,2);else if(waves==4)GW(K,4);else if(waves==8)GW(K,8);else TORCH_CHECK(false,"waves");}while(0)
 switch(k){case 6144:GK(6144);break;case 5120:GK(5120);break;case 17408:GK(17408);break;default:TORCH_CHECK(false,"K unsupported");}
#undef GK
#undef GW
#undef GM
 return o;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gemv",&gemv,"K100AI buffer-pipelined sdot4 GEMV v6",py::arg("x"),py::arg("w"),py::arg("xs"),py::arg("ws"),py::arg("waves")=2,py::arg("mode")=0);}
