#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "freshman.h"

__global__ void sumArraysGPU(float*a, float*b, float*res)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    res[i]=a[i]+b[i];
}

enum MemoryMode {
    PINNED_MEMORY,  // pinned_memory模式
    UVA_MEMORY      // UVA模式
};

int main(int argc, char **argv)
{
    utills::initDevice(0);

    MemoryMode mode = PINNED_MEMORY;
    if (argc > 1) {
        if (atoi(argv[1]) == 1) {
            mode = UVA_MEMORY;
        } else {
            mode = PINNED_MEMORY;
        }
    }

    int nElem = 1 << 24;
    size_t nBytes = nElem * sizeof(float);
    printf("Vector size %d, bytes %lu\n", nElem, nBytes);

    float *h_a, *h_b, *hostRef, *gpuRef;
    float *d_a, *d_b, *d_res;


    if (mode == PINNED_MEMORY) {
        // 固定内存是一种稀缺资源，
        // 过度使用可能会降低系统整体性能。
        // 建议只在需要频繁进行主机和设备之间数据传输的情况下使用。
        printf("Using pinned memory mode\n");

        // 方法1: 直接分配pinned内存并使用
        CHECK(cudaMallocHost((void**)&h_a, nBytes));
        CHECK(cudaMallocHost((void**)&h_b, nBytes));
        CHECK(cudaMallocHost((void**)&hostRef, nBytes)); // 设备端结果内存
        
        // 直接在pinned内存上初始化数据
        utills::initialData(h_a, nElem);
        utills::initialData(h_b, nElem);
        
        // 从pinned内存拷贝到设备
        CHECK(cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice));
    } else if (mode == UVA_MEMORY) {
        printf("Using UVA memory mode\n");
        // 使用统一虚拟地址（UVA）分配可映射的主机内存
        // 这种内存可以直接被设备访问，无需显式复制
        CHECK(cudaHostAlloc((float**)&h_a, nBytes, cudaHostAllocMapped));
        CHECK(cudaHostAlloc((float**)&h_b, nBytes, cudaHostAllocMapped));
        CHECK(cudaHostAlloc((float**)&gpuRef, nBytes, cudaHostAllocMapped));

        utills::initialData(h_a, nElem);  // 在分配后初始化数据
        utills::initialData(h_b, nElem);
        memset(gpuRef, 0, nBytes);

        // 获取主机内存映射到设备的指针，实现统一内存寻址
        CHECK(cudaHostGetDevicePointer((void**)&d_a, (void*)h_a, 0));
        CHECK(cudaHostGetDevicePointer((void**)&d_b, (void*)h_b, 0));
        CHECK(cudaHostGetDevicePointer((void**)&d_res, (void*)gpuRef, 0));
    } 
    
    // todo 


    // 清理内存
    if (mode == PINNED_MEMORY) {
        CHECK(cudaFreeHost(h_a));
        CHECK(cudaFreeHost(h_b));
        CHECK(cudaFreeHost(hostRef));
        CHECK(cudaFreeHost(gpuRef));
        CHECK(cudaFree(d_a));
        CHECK(cudaFree(d_b));
        CHECK(cudaFree(d_res));
    } else {
        CHECK(cudaFreeHost(h_a));
        CHECK(cudaFreeHost(h_b));
        CHECK(cudaFreeHost(gpuRef));
    }

    return 0;

}

