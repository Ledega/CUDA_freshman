#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include "freshman.h"

__global__ void sumArraysGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int nElem = 1 << 24;
    printf("Vector addition of %d elements\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;
    CHECK(cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void **)&h_C, nBytes, cudaHostAllocDefault));
    utills::initialData(h_A, nElem);
    utills::initialData(h_B, nElem);
    memset(h_C, 0, nBytes);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));
    CHECK(cudaMalloc((void **)&d_C, nBytes));

    dim3 block(256);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("Grid %d Block %d\n", grid.x, block.x);

    // 1. 使用普通方式进行计算
    CudaTimer timer;
    timer.start();
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    sumArraysGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost)); // 添加这行
    timer.stop();
    printf("Time elapsed: %f ms\n", timer.elapsed());

    const int N_SEG = 4;
    int iElem = nElem / N_SEG;
    cudaStream_t stream[N_SEG];
    for (int i = 0; i < N_SEG; i++)
    {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    // 2. 简单的分段流处理
    timer.start();
    for (int i = 0; i < N_SEG; i++)
    {
        int offset = i * iElem;
        dim3 gridSeg((iElem + block.x - 1) / block.x); // 修正grid计算

        CHECK(cudaMemcpyAsync(d_A + offset, h_A + offset, iElem * sizeof(float), cudaMemcpyHostToDevice, stream[i]));
        CHECK(cudaMemcpyAsync(d_B + offset, h_B + offset, iElem * sizeof(float), cudaMemcpyHostToDevice, stream[i]));
        sumArraysGPU<<<gridSeg, block, 0, stream[i]>>>(d_A + offset, d_B + offset, d_C + offset, iElem);
        CHECK(cudaMemcpyAsync(h_C + offset, d_C + offset, iElem * sizeof(float), cudaMemcpyDeviceToHost, stream[i]));
    }
    CHECK(cudaDeviceSynchronize());
    timer.stop();
    printf("Time elapsed with stream: %f ms\n", timer.elapsed());

    // 3. 广度优先的流处理
    // memset(h_C, 0, nBytes);
    timer.start();
    for (int i = 0; i < N_SEG; i++)
    {
        int offset = i * iElem;
        CHECK(cudaMemcpyAsync(d_A + offset, h_A + offset, iElem * sizeof(float), cudaMemcpyHostToDevice, stream[i]));
        CHECK(cudaMemcpyAsync(d_B + offset, h_B + offset, iElem * sizeof(float), cudaMemcpyHostToDevice, stream[i]));
    }
    for (int i = 0; i < N_SEG; i++)
    {
        int offset = i * iElem;
        dim3 gridSeg((iElem + block.x - 1) / block.x); // 修正grid计算
        sumArraysGPU<<<gridSeg, block, 0, stream[i]>>>(d_A + offset, d_B + offset, d_C + offset, iElem);
    }
    for (int i = 0; i < N_SEG; i++)
    {
        int offset = i * iElem;
        CHECK(cudaMemcpyAsync(h_C + offset, d_C + offset, iElem * sizeof(float), cudaMemcpyDeviceToHost, stream[i]));
    }
    CHECK(cudaDeviceSynchronize());
    timer.stop();
    printf("Time elapsed with stream breadth-first: %f ms\n", timer.elapsed());

    for (int i = 0; i < N_SEG; i++)
    {
        CHECK(cudaStreamDestroy(stream[i]));
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}