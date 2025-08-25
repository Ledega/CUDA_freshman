#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "freshman.h"

__global__ void warmup(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    int *idata_tmp = idata + blockIdx.x * blockDim.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            if ( idx + stride < n ) {
                idata_tmp[tid] += idata_tmp[tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        odata[blockIdx.x] = idata_tmp[tid];
    }
}

__global__ void reduceNeighbored(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    int *idata_tmp = idata + blockIdx.x * blockDim.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // 只有满足条件的线程参与归约操作（线程索引是2*stride的倍数）
        if (tid % (2 * stride) == 0) {
            if ( idx + stride < n ) {
                // 将当前元素与相距stride位置的元素相加
                idata_tmp[tid] += idata_tmp[tid + stride];
            }
        }
        // 同步block内所有线程，确保归约操作的正确性
        __syncthreads();
    }
    
    // 每个block的第一个线程将归约结果写入输出数组
    if (tid == 0) {
        odata[blockIdx.x] = idata_tmp[0];
    }
}

__global__ void reduceNeighboredLess(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    int *idata_tmp = idata + blockIdx.x * blockDim.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // 只有满足条件的线程参与归约操作（线程索引是2*stride的倍数）
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            // 将当前元素与相距stride位置的元素相加
            idata_tmp[tid] += idata_tmp[tid + stride];
        }
        // 同步block内所有线程，确保归约操作的正确性
        __syncthreads();
    }
    
    // 每个block的第一个线程将归约结果写入输出数组
    if (tid == 0) {
        odata[blockIdx.x] = idata_tmp[0];
    }
}

__global__ void reduceInterleaved(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    int *idata_tmp = idata + blockIdx.x * blockDim.x;
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			idata_tmp[tid] += idata_tmp[tid + stride];
		}
		__syncthreads();
	}
    
    // 每个block的第一个线程将归约结果写入输出数组
    if (tid == 0) {
        odata[blockIdx.x] = idata_tmp[0];
    }
}

int main()
{
    utills::initDevice(0);

    int nElem = 1 << 20;
    size_t nBytes = nElem * sizeof(int);

    int *h_idata = (int *)malloc(nBytes);
    int *h_odata = (int *)malloc(nBytes);
    utills::initialData_int(h_idata, nElem);
    utills::initialData_int(h_odata, nElem);

    dim3 block(512);
    dim3 grid((nElem + block.x - 1) / block.x);

    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((int**)&d_idata, nBytes));
    CHECK(cudaMalloc((int**)&d_odata, grid.x * sizeof(int)));

    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));

    // warm up
    CudaTimer timer;
    timer.start();
    warmup<<<grid, block>>>(d_idata, d_odata, nElem);
    timer.stop();
    printf("warmup <<< %d, %d >>> elapsed [%f] ms\n", grid.x, block.x, timer.elapsed());

    // reduceNeighbored
    timer.start();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, nElem);
    timer.stop();
    printf("reduceNeighbored <<< %d, %d >>> elapsed [%f] ms\n", grid.x, block.x, timer.elapsed());

    // reduceNeighboredLess
    timer.start();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, nElem);
    timer.stop();
    printf("reduceNeighboredLess <<< %d, %d >>> elapsed [%f] ms\n", grid.x, block.x, timer.elapsed());

    // reduceInterleaved
    timer.start();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, nElem);
    timer.stop();
    printf("reduceInterleaved <<< %d, %d >>> elapsed [%f] ms\n", grid.x, block.x, timer.elapsed());

    return 0;

}