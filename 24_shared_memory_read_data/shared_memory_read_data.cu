#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include "freshman.h"

#define BDIMX 32
#define BDIMY 32

#define BDIMX_RECT 32
#define BDIMY_RECT 16
#define IPAD 1

__global__ void setRowReadRow(float* data) {
    __shared__ float tile[BDIMX][BDIMY];
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

    data[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(float* data) {
    __shared__ float tile[BDIMX][BDIMY];
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();

    data[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDyn(float* data) {
    extern __shared__ float tile[]; // 动态分配共享内存
    int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int col_idx = threadIdx.x * blockDim.y + threadIdx.y;
    tile[row_idx] = row_idx;
    __syncthreads();

    data[row_idx] = tile[col_idx];
}

__global__ void setRowReadColIpad(float* data) {
    __shared__ float tile[BDIMX][BDIMY + IPAD];
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

    data[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynIpad(float* data) {
    extern __shared__ float tile[]; // 动态分配共享内存
    // 这里的索引看文档好理解 不深究了
    int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    int col_idx = threadIdx.x * (blockDim.x + IPAD) + threadIdx.y;
    tile[row_idx] = row_idx;
    __syncthreads();

    data[row_idx] = tile[col_idx];
}

int main(int argc, char* argv[])
{
    utills::initDevice(0);

    int mode = 0;
    if (argc == 2) {
        mode = atoi(argv[1]);
    } else {
        printf("Usage: %s <mode>\n", argv[0]);
        return -1;
    }
    
    cudaSharedMemConfig memConfig;
    CHECK(cudaDeviceGetSharedMemConfig(&memConfig));
    printf("Shared memory bank size: ");
    if (memConfig == cudaSharedMemBankSizeDefault) {
        printf("Default\n");
    } else if (memConfig == cudaSharedMemBankSizeFourByte) {
        printf("4 Bytes\n");
    } else if (memConfig == cudaSharedMemBankSizeEightByte) {
        printf("8 Bytes\n");
    } else {
        printf("Unknown\n");
    }

    const int N = BDIMX;
    // const int bytes = N * sizeof(float);  
    CudaArray<float> d_data(N);

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    dim3 blockRect(BDIMX_RECT, BDIMY_RECT);
    dim3 gridRect(1, 1);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", gridRect.x, gridRect.y, blockRect.x, blockRect.y);

    CudaTimer timer;
    timer.start();  
    switch (mode)
    {
    case 0:
        setRowReadRow<<<grid, block>>>(d_data.get());
        break;

    case 1:
        setColReadCol<<<grid, block>>>(d_data.get());
        break;

    case 2:
        setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(float)>>>(d_data.get());
        break; 

    case 3:
        setRowReadColIpad<<<gridRect, blockRect>>>(d_data.get());
        break;

    case 4:
        setRowReadColDynIpad<<<gridRect, blockRect, (BDIMX * (BDIMY + IPAD)) * sizeof(float)>>>(d_data.get());
        break;
    
    default:
        break;
    }

    timer.stop();
    float timeMs = timer.elapsed();
    printf("Time elapsed: %.3f ms\n", timeMs);

    CHECK(cudaDeviceSynchronize());

    return 0;

}