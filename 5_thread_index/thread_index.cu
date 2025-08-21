#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include "freshman.h"

__global__ void printThreadIndex(float *arr, const int nx, const int ny) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = iy * nx + ix;
  printf("thread_id(%d, %d) block_id(%d, %d) coordinate(%d,% d) global index %2d ival %f\n",
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, arr[idx]);
}

int main() 
{
    utills::initDevice(0);
   
    int nx = 8, ny = 6;
    int nElem = nx * ny;
    int nByte = nElem * sizeof(float);

    // malloc host memory
    float* A_host=(float*)malloc(nByte);
    utills::initialData(A_host, nElem);
    utills::printMatrix(A_host, nx, ny);

    float *A_dev=nullptr;
    CHECK(cudaMalloc((float**)&A_dev, nByte));
    CHECK(cudaMemcpy(A_dev, A_host, nByte, cudaMemcpyHostToDevice));

    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    printf("Grid: (%d, %d) Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    printThreadIndex<<<grid, block>>>(A_dev, nx, ny);
    CHECK(cudaDeviceSynchronize());
    cudaFree(A_dev);
    free(A_host);

    cudaDeviceReset();

    return 0;
}