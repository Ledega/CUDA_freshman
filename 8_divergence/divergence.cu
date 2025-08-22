#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "freshman.h"

__global__ void divergent_kernel1(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0f, b = 0.0f;
    if (idx % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    data[idx] = a + b;
}

__global__ void divergent_kernel2(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0f, b = 0.0f;
    int ipred = (idx % 2 == 0);
    if (ipred) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    data[idx] = a + b;
}

__global__ void warp_kernel1(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0f, b = 0.0f;
    if ((idx / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    data[idx] = a + b;
}

__global__ void warmup_kernel(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 0.0f, b = 0.0f;
    if ((idx / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    data[idx] = a + b;
}

int main() 
{
    utills::initDevice(0);

    int nElem = 1 << 16; 
    int nBytes = nElem * sizeof(float);
    float *data_h = (float *)malloc(nBytes);

    float *data_d = nullptr;
    CHECK(cudaMalloc((void **)&data_d, nBytes));
    CHECK(cudaMemset(data_d, 0, nBytes));

    dim3 block(256, 1, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1, 1);

    CudaTimer timer;

    timer.start();
    warmup_kernel<<<grid, block>>>(data_d);
    timer.stop();
    printf("Warmup kernel time: %.3f ms\n", timer.elapsed());

    timer.start();
    divergent_kernel1<<<grid, block>>>(data_d);
    timer.stop();
    printf("Divergent kernel1 time: %.3f ms\n", timer.elapsed());

    timer.start();
    divergent_kernel2<<<grid, block>>>(data_d);
    timer.stop();
    printf("Divergent kernel2 time: %.3f ms\n", timer.elapsed());

    timer.start();
    warp_kernel1<<<grid, block>>>(data_d);
    timer.stop();
    printf("Warp-level divergent kernel time: %.3f ms\n", timer.elapsed());

    cudaDeviceReset();
    return 0;
}