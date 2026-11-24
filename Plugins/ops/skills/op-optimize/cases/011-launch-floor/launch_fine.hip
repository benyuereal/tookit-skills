#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
__global__ void g(float* o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)o[i]=1.f;}
int main(){
  float* d; hipMalloc(&d,1<<22);
  auto b=[&](int blocks){
    std::vector<float> ts;
    for(int i=0;i<5;i++) g<<<blocks,256>>>(d,1<<20);
    hipDeviceSynchronize();
    for(int i=0;i<41;i++){
      hipEvent_t a,e; hipEventCreate(&a); hipEventCreate(&e);
      hipEventRecord(a); g<<<blocks,256>>>(d,1<<20); hipEventRecord(e);
      hipEventSynchronize(e); float ms; hipEventElapsedTime(&ms,a,e);
      ts.push_back(ms*1000.f); hipEventDestroy(a); hipEventDestroy(e);
    }
    std::sort(ts.begin(),ts.end());
    printf("grid=%-7d %7.1f us\n",blocks,ts[ts.size()/2]);
  };
  for(int n: {1,512,2048,4096,8192,16384,32768}) b(n);
  return 0;
}
