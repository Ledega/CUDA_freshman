#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "freshman.h"

__global__ void sumMatrix2D(float *matA, float *matB, float *matRes, const int nx, const int ny) 
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        int idx = iy * nx + ix;
        matRes[idx] = matA[idx] + matB[idx];
    }
}

void sumMatrix2DHost(const std::vector<std::vector<float>> &matA, 
                    const std::vector<std::vector<float>> &matB, 
                    std::vector<std::vector<float>> &matC) 
{
    int rows = matA.size();
    int cols = matA[0].size();
    
    // 执行矩阵加法：C[i][j] = A[i][j] + B[i][j]
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matC[i][j] = matA[i][j] + matB[i][j];
        }
    }
}

int main(int argc,char** argv) 
{
    const int nx = 1024;
    const int ny = 1024;
    const int size = nx * ny * sizeof(float);
    
    // 分配主机内存
    std::vector<std::vector<float>> matA(ny, std::vector<float>(nx, 1.0f));
    std::vector<std::vector<float>> matB(ny, std::vector<float>(nx, 2.0f));
    std::vector<std::vector<float>> matC(ny, std::vector<float>(nx, 0.0f));

    // 分配设备内存
    float *d_matA, *d_matB, *d_matC;
    CHECK(cudaMalloc((void**)&d_matA, size));
    CHECK(cudaMalloc((void**)&d_matB, size));
    CHECK(cudaMalloc((void**)&d_matC, size));

    // 将数据从主机复制到设备
    CHECK(cudaMemcpy(d_matA, matA[0].data(), size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_matB, matB[0].data(), size, cudaMemcpyHostToDevice));

    // 设置CUDA网格和块的维度
    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    // 启动CUDA核函数
    CudaTimer timer;
    timer.start();
    sumMatrix2D<<<gridDim, blockDim>>>(d_matA, d_matB, d_matC, nx, ny);
    timer.stop();
    printf("Kernel execution time: [%.2f] ms\n", timer.elapsed());

    // 将结果从设备复制回主机
    // cudaMemcpy(matC[0].data(), d_matC, size, cudaMemcpyDeviceToHost);

    // 验证结果
    timer.start();
    sumMatrix2DHost(matA, matB, matC);
    timer.stop();
    printf("CPU execution time: [%.2f] ms\n", timer.elapsed());

    // 清理资源
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);

    return 0;
}