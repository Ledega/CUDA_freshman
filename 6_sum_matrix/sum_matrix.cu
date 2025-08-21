#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include "freshman.h"


__global__ void sumMatrix(float *matA, float *matB, float *matRes, const int nx, const int ny) 
{
    // 支持1D和2D调用
    if (blockDim.y == 1) {
        // 1D配置
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < nx * ny) {
            matRes[idx] = matA[idx] + matB[idx];
        }
    } else {
        // 2D配置
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;
        if (ix < nx && iy < ny) {
            int idx = iy * nx + ix;
            matRes[idx] = matA[idx] + matB[idx];
        }
    }
}

int main() 
{
    utills::initDevice(0);
   
    int nx = 1 << 12, ny = 1 << 12;
    int nElem = nx * ny;
    int nByte = nElem * sizeof(float);

    // malloc host memory
    float* A_host=(float*)malloc(nByte);
    float* B_host=(float*)malloc(nByte);
    float* C_host=(float*)malloc(nByte);

    utills::initialData(A_host, nElem);
    utills::initialData(B_host, nElem);

    // malloc device memory
    float *A_dev=nullptr, *B_dev=nullptr, *C_dev=nullptr;
    CHECK(cudaMalloc((float**)&A_dev, nByte));
    CHECK(cudaMalloc((float**)&B_dev, nByte));
    CHECK(cudaMalloc((float**)&C_dev, nByte));

    // copy data from host to device
    CHECK(cudaMemcpy(A_dev, A_host, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nByte, cudaMemcpyHostToDevice));

    int dimx = 32, dimy = 32;
    CudaTimer timer;

    // 2d grid and 2d block
    dim3 block2d(dimx, dimy);
    dim3 grid2d((nx + block2d.x - 1) / block2d.x, (ny + block2d.y - 1) / block2d.y);
    printf("Grid: (%d, %d) Block: (%d, %d)\n", grid2d.x, grid2d.y, block2d.x, block2d.y);
    timer.start();
    sumMatrix<<<grid2d, block2d>>>(A_dev, B_dev, C_dev, nx, ny);
    timer.stop();
    printf("Elapsed time: [%f] ms\n", timer.elapsed());

    // 1d grid and 1d block
    dim3 block1d(dimx);
    dim3 grid1d((nElem + block1d.x - 1) / block1d.x);
    printf("Grid: (%d, %d) Block: (%d, %d)\n", grid1d.x, grid1d.y, block1d.x, block1d.y);
    timer.start();
    sumMatrix<<<grid1d, block1d>>>(A_dev, B_dev, C_dev, nx, ny);
    timer.stop();
    printf("Elapsed time: [%f] ms\n", timer.elapsed());

    // 2d grid and 1d block
    dim3 grid2d_new((nx + block2d.x - 1) / block2d.x, ny);
    printf("Grid: (%d, %d) Block: (%d, %d)\n", grid2d_new.x, grid2d_new.y, block1d.x, block1d.y);
    timer.start();
    sumMatrix<<<grid2d_new, block1d>>>(A_dev, B_dev, C_dev, nx, ny);
    timer.stop();
    printf("Elapsed time: [%f] ms\n", timer.elapsed());

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    free(A_host);
    free(B_host);
    free(C_host);

    cudaDeviceReset();

    return 0;
}