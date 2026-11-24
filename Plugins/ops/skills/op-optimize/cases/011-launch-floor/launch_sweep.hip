#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
__global__ void nop(){}
__global__ void grid_k(float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=1.f;}
int main(){
  float* d; hipMalloc(&d, 1<<22);
  auto bench=[&](int blocks,int threads,const char* tag){
    std::vector<float> ts;
    for(int i=0;i<5;i++) grid_k<<<blocks,threads>>>(d,1<<20);
    hipDeviceSynchronize();
    for(int i=0;i<41;i++){
      hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
      hipEventRecord(a); grid_k<<<blocks,threads>>>(d,1<<20); hipEventRecord(b);
      hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms,a,b);
      ts.push_back(ms*1000.0f); hipEventDestroy(a); hipEventDestroy(b);
    }
    std::sort(ts.begin(),ts.end());
    float med=ts[ts.size()/2];
    std::vector<float> dev; for(auto t:ts) dev.push_back(fabsf(t-med));
    std::sort(dev.begin(),dev.end());
    printf("%-22s %7.1f us  MAD=%5.1f\n",tag,med,dev[dev.size()/2]);
  };
  bench(1,1,"nop 1x1");
  bench(1,256,"grid=1 x256");
  bench(64,256,"grid=64 x256");
  bench(1024,256,"grid=1024 x256");
  bench(8192,256,"grid=8192 x256");
  bench(65536,256,"grid=65536 x256");
  return 0;
}
