#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include "freshman.h"

__global__ void reduceGlobalmem(int *idata, int *odata, int n) {
    // 使用全局内存进行归约
    int tid = threadIdx.x;
    if (tid >= n) return; // 防止越界
    
    // 获取当前 block 的起始位置
    int *idata_st = idata + blockIdx.x * blockDim.x;
    
    // 先进行第一步归约
    // if (idx + blockDim.x * 7 < n) {
    //     idata_st[tid] += idata_st[tid + blockDim.x];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 2];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 3];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 4];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 5];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 6];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 7];
    // }
    // __syncthreads();

    // 完全展开，但有问题 如果 blockDim.x = 2048 就规约不完全了
    if (blockDim.x >= 1024 && tid < 512 && tid + 512 < n)
        idata_st[tid] += idata_st[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256 && tid + 256 < n)
        idata_st[tid] += idata_st[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128 && tid + 128 < n)
        idata_st[tid] += idata_st[tid + 128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64 && tid + 64 < n)
        idata_st[tid] += idata_st[tid + 64];
    __syncthreads();

    // stride>32 剩下的元素个数<64
    if (tid < 32) {
        volatile int *vmem = idata_st;
		vmem[tid] += vmem[tid+32];
		vmem[tid] += vmem[tid+16];
		vmem[tid] += vmem[tid+8];
		vmem[tid] += vmem[tid+4];
		vmem[tid] += vmem[tid+2];
		vmem[tid] += vmem[tid+1];
    }

    // 保存结果
    if (tid == 0) {
        odata[blockIdx.x] = idata_st[0];
    }
}


__global__ void reduceSharedmem(int *idata, int *odata, int n) {
    // 使用全局内存进行归约
    int tid = threadIdx.x;
    if (tid >= n) return; // 防止越界

    __shared__ int sdata[1024]; // 共享内存大小需要根据实际情况调整

    // 获取当前 block 的起始位置
    int *idata_st = idata + blockIdx.x * blockDim.x;
    sdata[tid] = idata_st[tid];
    __syncthreads();
    
    // 先进行第一步归约
    // if (idx + blockDim.x * 7 < n) {
    //     idata_st[tid] += idata_st[tid + blockDim.x];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 2];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 3];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 4];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 5];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 6];
    //     idata_st[tid] += idata_st[tid + blockDim.x * 7];
    // }
    // __syncthreads();

    // 完全展开，但有问题 如果 blockDim.x = 2048 就规约不完全了
	if(blockDim.x>=1024 && tid <512)
		sdata[tid] += sdata[tid+512];
	__syncthreads();

	if(blockDim.x>=512 && tid <256)
		sdata[tid] += sdata[tid+256];
	__syncthreads();

	if(blockDim.x>=256 && tid <128)
		sdata[tid] += sdata[tid+128];
	__syncthreads();

	if(blockDim.x>=128 && tid <64)
		sdata[tid] += sdata[tid+64];
	__syncthreads();

    // stride>32 剩下的元素个数<64
    if (tid < 32) {
        volatile int *vsmem = sdata;
		vsmem[tid] += vsmem[tid+32];
		vsmem[tid] += vsmem[tid+16];
		vsmem[tid] += vsmem[tid+8];
		vsmem[tid] += vsmem[tid+4];
		vsmem[tid] += vsmem[tid+2];
		vsmem[tid] += vsmem[tid+1];
    }

    // 保存结果
    if (tid == 0) {
        odata[blockIdx.x] = sdata[0];
    }
}

int main()
{
    const int DIM = 1 << 10; // 1<<10 = 1024
    std::vector<int> h_data(DIM);
    std::iota(h_data.begin(), h_data.end(), 1); // 初始化数据 1,2,3,...,DIM
    int cpu_sum = std::accumulate(h_data.begin(), h_data.end(), 0);
    printf("CPU sum: %d\n", cpu_sum);

    int *d_idata = nullptr;
    int *d_odata = nullptr;
    dim3 grid(1);
    dim3 block(1024);
    printf("grid.x: %d, block.x: %d\n", grid.x, block.x);

    // 如果定义的是 #define DIM 1 << 10
    // 编译时会文本替换 → DIM * sizeof(int) 会变成 1 << 10 * sizeof(int)
    // 在 64 位系统上 sizeof(int) = 4，所以这是：1 << 40
    // 你原本想申请 4 KB（1024 * 4），结果申请了 1 TB，当然超出显存，后续访问必然非法
    size_t size = DIM * sizeof(int);

    CHECK(cudaMalloc((void**)&d_idata, size));
    CHECK(cudaMalloc((void**)&d_odata, grid.x * sizeof(int)));
    CHECK(cudaMemcpy(d_idata, h_data.data(), size, cudaMemcpyHostToDevice));

    // 使用全局内存归约
    reduceGlobalmem<<<grid, block>>>(d_idata, d_odata, DIM);
    CHECK(cudaGetLastError()); 
    CHECK(cudaDeviceSynchronize());

    std::vector<int> h_odata(grid.x);
    CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    int gpu_sum_global = std::accumulate(h_odata.begin(), h_odata.end(), 0);
    printf("GPU sum (global mem): %d\n", gpu_sum_global);

    // 使用共享内存归约
    CHECK(cudaMemcpy(d_idata, h_data.data(), size, cudaMemcpyHostToDevice));
    reduceSharedmem<<<grid, block>>>(d_idata, d_odata, DIM);
    CHECK(cudaGetLastError()); 
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_odata.data(), d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    int gpu_sum_shared = std::accumulate(h_odata.begin(), h_odata.end(), 0);
    printf("GPU sum (shared mem): %d\n", gpu_sum_shared);

    // 清理资源
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    return 0;
}