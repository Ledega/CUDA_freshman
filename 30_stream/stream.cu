#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include "freshman.h"

#define N 30000

__global__ void kernel_1()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum = sum + tan(0.1) * tan(0.1);
}

__global__ void kernel_2()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum = sum + tan(0.1) * tan(0.1);
}

__global__ void kernel_3()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum = sum + tan(0.1) * tan(0.1);
}

__global__ void kernel_4()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum = sum + tan(0.1) * tan(0.1);
}

int main()
{
    int nStreams = 16;
    cudaStream_t *streams = (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));
    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    dim3 block(1);
    dim3 grid(1);

    CudaTimer timer;
    timer.start();
    for (size_t i = 0; i < nStreams; i++)
    {
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }
    timer.stop();
    printf("Time elapsed: %f ms\n", timer.elapsed());

    cudaDeviceSynchronize();
    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    return 0;
}