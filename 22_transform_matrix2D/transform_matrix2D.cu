#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include "freshman.h"

__global__ void copyRow(float *matA, float *matB, const int rows, const int cols) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * cols + ix;
    if (ix < cols && iy < rows) {
        matB[idx] = matA[idx];
    }
}

__global__ void copyCol(float *matA, float *matB, const int rows, const int cols) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ix * rows + iy;
    if (ix < cols && iy < rows) {
        matB[idx] = matA[idx];
    }
}

__global__ void transformNaiveRow(float *matA, float *matB, const int rows, const int cols) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < cols && iy < rows) {
        int idx_in  = iy * cols + ix;  // 原矩阵 (row-major)
        int idx_out = ix * rows + iy;  // 转置后矩阵
        matB[idx_out] = matA[idx_in];
    }
}

__global__ void transformNaiveCol(float *matA, float *matB, const int rows, const int cols) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < cols && iy < rows) {
        int idx_in  = ix * rows + iy;  // 原矩阵 (row-major)
        int idx_out = iy * cols + ix;  // 转置后矩阵
        matB[idx_out] = matA[idx_in];
    }
}

__global__ void transformNaiveRowUnroll(float * MatA, float * MatB, int rows, int cols)
{
    // 计算当前线程在x方向的全局索引，每个block处理4倍的数据
    // blockIdx.x * blockDim.x * 4 意味着每个block在x方向展开4倍
    int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    // 计算当前线程在y方向的全局索引
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 边界检查：确保y坐标在有效范围内
    if (iy < rows) {
        // 第一个元素的处理
        // 检查第一个展开的元素是否在边界内
        if (ix < cols) {
            // 源矩阵索引：按行存储 (row-major order)
            // MatA[row][col] = MatA[ix + iy * rows]
            int idx_row = iy * cols + ix;
            
            // 目标矩阵索引：转置后按列存储
            // MatB[col][row] = MatB[ix * cols + iy]
            int idx_col = ix * rows + iy;
            
            // 执行矩阵转置：将源矩阵的行变成目标矩阵的列
            MatB[idx_col] = MatA[idx_row];
        }
        
        // 第二个元素的处理（展开后的第一个额外元素）
        // 检查x坐标偏移blockDim.x后是否还在边界内
        if (ix + blockDim.x < rows) {
            // 计算偏移后的源矩阵索引
            int idx_row = (ix + blockDim.x) + iy * rows;
            
            // 计算偏移后的目标矩阵索引
            int idx_col = (ix + blockDim.x) * cols + iy;
            
            // 执行转置操作
            MatB[idx_col] = MatA[idx_row];
        }
        
        // 第三个元素的处理（展开后的第二个额外元素）
        // 检查x坐标偏移2*blockDim.x后是否还在边界内
        if (ix + 2 * blockDim.x < rows) {
            // 计算偏移后的源矩阵索引
            int idx_row = (ix + 2 * blockDim.x) + iy * rows;
            
            // 计算偏移后的目标矩阵索引
            int idx_col = (ix + 2 * blockDim.x) * cols + iy;
            
            // 执行转置操作
            MatB[idx_col] = MatA[idx_row];
        }
        
        // 第四个元素的处理（展开后的第三个额外元素）
        // 检查x坐标偏移3*blockDim.x后是否还在边界内
        if (ix + 3 * blockDim.x < rows) {
            // 计算偏移后的源矩阵索引
            int idx_row = (ix + 3 * blockDim.x) + iy * rows;
            
            // 计算偏移后的目标矩阵索引
            int idx_col = (ix + 3 * blockDim.x) * cols + iy;
            
            // 执行转置操作
            MatB[idx_col] = MatA[idx_row];
        }
    }
}

__global__ void transformNaiveColUnroll(float * MatA, float * MatB, int rows, int cols)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (iy < rows) {
        // 检查每个展开的元素是否在边界内
        if (ix < cols) {
            int idx_row = iy * cols + ix;
            int idx_col = ix * rows + iy;
            MatB[idx_row] = MatA[idx_col];
        }
        
        if (ix + blockDim.x < rows) {
            int idx_row = (ix + blockDim.x) + iy * rows;
            int idx_col = (ix + blockDim.x) * cols + iy;
            MatB[idx_row] = MatA[idx_col];
        }
        
        if (ix + 2 * blockDim.x < rows) {
            int idx_row = (ix + 2 * blockDim.x) + iy * rows;
            int idx_col = (ix + 2 * blockDim.x) * cols + iy;
            MatB[idx_row] = MatA[idx_col];
        }
        
        if (ix + 3 * blockDim.x < rows) {
            int idx_row = (ix + 3 * blockDim.x) + iy * rows;
            int idx_col = (ix + 3 * blockDim.x) * cols + iy;
            MatB[idx_row] = MatA[idx_col];
        }
    }
}

int main(int argc, char** argv) 
{
    utills::initDevice(0);

    int mode;
    if (argc == 2) {
        mode = std::atoi(argv[1]);
    } else {
        printf("GIVE ME MODE NUMBER.\n");
        return 1;
    }

    int rows = 1 << 12, cols = 1 << 12;
    int nElem = rows * cols;
    int nBytes = nElem * sizeof(float);

    std::vector<float> h_matA(nElem);
    std::vector<float> h_matB(nElem);
    std::iota(h_matA.begin(), h_matA.end(), 0.0f);
    std::iota(h_matB.begin(), h_matB.end(), 0.0f);

    CudaArray<float> d_matA(nElem);
    CudaArray<float> d_matB(nElem);

    CHECK(cudaMemcpy(d_matA.get(), h_matA.data(), nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_matB.get(), h_matB.data(), nBytes, cudaMemcpyHostToDevice));

    printf("Matrix: rows=%d, cols=%d\n", rows, cols);
    // 2D block and grid
    int dimx = 32, dimy = 32;
    dim3 block1(dimx, dimy);
    dim3 grid1((cols + block1.x -1) / block1.x, (rows + block1.y -1) / block1.y);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid1.x, grid1.y, block1.x, block1.y);

    dim3 block2(dimx, dimy);
    dim3 grid2((cols + block2.x * 4 - 1) / (block2.x * 4), 
           (rows + block2.y - 1) / block2.y);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid2.x, grid2.y, block2.x, block2.y);  
    
    CudaTimer timer;
    timer.start();
    switch (mode)
    {
    case 1:
        copyRow<<<grid1, block1>>>(d_matA.get(), d_matB.get(), rows, cols);
        break;
    
    case 2:
        copyCol<<<grid1, block1>>>(d_matA.get(), d_matB.get(), rows, cols);
        break;

    case 3:
        transformNaiveRow<<<grid1, block1>>>(d_matA.get(), d_matB.get(), rows, cols);
        break;
    
    case 4:
        transformNaiveCol<<<grid1, block1>>>(d_matA.get(), d_matB.get(), rows, cols);
        break;

    case 5:
        transformNaiveRowUnroll<<<grid2, block2>>>(d_matA.get(), d_matB.get(), rows, cols);
        break;

    case 6:
        transformNaiveColUnroll<<<grid2, block2>>>(d_matA.get(), d_matB.get(), rows, cols);
        break;

    default:
        printf("INVALID MODE. \n");
        break;
    }
    timer.stop();
    printf("Cuda time elapsed [%f] ms. \n", timer.elapsed());
    
    CHECK(cudaDeviceSynchronize());
    /*
      当调用 cudaDeviceReset() 时，会释放所有的 CUDA 资源，包括已分配的内存。
      但是在这之后，CudaArray 对象的析构函数还会尝试再次释放这些内存，导致了 double free 问题。
    */

    // CHECK(cudaDeviceReset());

    return 0;
}