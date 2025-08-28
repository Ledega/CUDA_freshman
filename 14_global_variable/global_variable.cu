#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "freshman.h"

__device__ float devData;

__global__ void checkGlobalVariable() {
    printf("Device: The value of the global variable is %f. \n", devData);
    devData += 2.0f;
}

/*
    在主机端，devData只是一个标识符，不是设备全局内存的变量地址
    在核函数中，devData就是一个全局内存中的变量。
    主机代码不能直接访问设备变量，设备也不能访问主机变量，这就是CUDA编程与CPU多核最大的不同之处
*/

int main()
{
    float hostData = 1.0f;
    CHECK(cudaMemcpyToSymbol(devData, &hostData, sizeof(float)));
    printf("Host: The value of the global variable is %f. \n", devData);
    checkGlobalVariable<<<1, 1>>>();
    cudaDeviceSynchronize();
    CHECK(cudaMemcpyFromSymbol(&hostData, devData, sizeof(float)));
    printf("Host: after kernel The value of the global variable is %f. \n", hostData);
    cudaDeviceReset();
    return 0;
}