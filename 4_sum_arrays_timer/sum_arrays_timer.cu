#include <stdio.h>
#include <cuda_runtime.h>
#include "freshman.h"

void sumArrays(float * a,float * b,float * res,const int size)
{
  for(int i=0;i<size;i+=4)
  {
    res[i]=a[i]+b[i];
    res[i+1]=a[i+1]+b[i+1];
    res[i+2]=a[i+2]+b[i+2];
    res[i+3]=a[i+3]+b[i+3];
  }
}

__global__ void sum_arrays(const float *a, const float *b, float *c, float n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {

    // set up device
    utills::initDevice(0);
    
    int nElem = 1 << 24;
    int nByte = nElem * sizeof(float);
    printf("Vector size: %d\n", nElem);

    // malloc host memory
    float *h_a = (float *)malloc(nByte);
    float *h_b = (float *)malloc(nByte);
    float *h_res = (float *)malloc(nByte);
    float *h_res_from_gpu = (float *)malloc(nByte);

    memset(h_res, 0, nByte);
    memset(h_res_from_gpu, 0, nByte);

    // malloc device memory
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((float**)&d_a, nByte));
    CHECK(cudaMalloc((float**)&d_b, nByte));
    CHECK(cudaMalloc((float**)&d_c, nByte));

    utills::initialData(h_a, nElem);
    utills::initialData(h_b, nElem);

    // copy data from host to device
    CHECK(cudaMemcpy(d_a, h_a, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, nByte, cudaMemcpyHostToDevice));

    // launch kernel
    dim3 block(256, 1, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1, 1);
    sum_arrays<<<grid, block>>>(d_a, d_b, d_c, nElem);
    printf("Execution configuration <<<%d, %d>>> \n",grid.x, block.x);
    CHECK(cudaDeviceSynchronize());

    // copy data from device to host
    CHECK(cudaMemcpy(h_res_from_gpu, d_c, nByte, cudaMemcpyDeviceToHost));

    // verify result
    sumArrays(h_a, h_b, h_res, nElem);
    utills::checkResult(h_res, h_res_from_gpu, nElem);

    // free memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_res);
    free(h_res_from_gpu);

    return 0;
}