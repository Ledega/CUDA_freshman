#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "freshman.h"

__global__ void reduceUnroll2(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;
    
    // 获取当前 block 的起始位置
    int *idata_st = idata + blockIdx.x * blockDim.x * 2;
    
    // 先进行第一步归约
    if (idx + blockDim.x < n) {
        idata_st[tid] += idata_st[tid + blockDim.x];
    }
    __syncthreads();

    // 后续归约步骤
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata_st[tid] += idata_st[tid + stride];
        }
        __syncthreads();
    }

    // 保存结果
    if (tid == 0) {
        odata[blockIdx.x] = idata_st[0];
    }
}

__global__ void reduceUnroll4(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 4 + tid;
    
    // 获取当前 block 的起始位置
    int *idata_st = idata + blockIdx.x * blockDim.x * 4;
    
    // 先进行第一步归约
    if (idx + blockDim.x * 3 < n) {
        idata_st[tid] += idata_st[tid + blockDim.x];
        idata_st[tid] += idata_st[tid + blockDim.x * 2];
        idata_st[tid] += idata_st[tid + blockDim.x * 3];
    }
    __syncthreads();

    // 后续归约步骤
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata_st[tid] += idata_st[tid + stride];
        }
        __syncthreads();
    }

    // 保存结果
    if (tid == 0) {
        odata[blockIdx.x] = idata_st[0];
    }
}

__global__ void reduceUnroll8(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + tid;
    
    // 获取当前 block 的起始位置
    int *idata_st = idata + blockIdx.x * blockDim.x * 8;
    
    // 先进行第一步归约
    if (idx + blockDim.x * 7 < n) {
        idata_st[tid] += idata_st[tid + blockDim.x];
        idata_st[tid] += idata_st[tid + blockDim.x * 2];
        idata_st[tid] += idata_st[tid + blockDim.x * 3];
        idata_st[tid] += idata_st[tid + blockDim.x * 4];
        idata_st[tid] += idata_st[tid + blockDim.x * 5];
        idata_st[tid] += idata_st[tid + blockDim.x * 6];
        idata_st[tid] += idata_st[tid + blockDim.x * 7];
    }
    __syncthreads();

    // 后续归约步骤
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata_st[tid] += idata_st[tid + stride];
        }
        __syncthreads();
    }

    // 保存结果
    if (tid == 0) {
        odata[blockIdx.x] = idata_st[0];
    }
}


__global__ void reduceUnroll8(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + tid;
    
    // 获取当前 block 的起始位置
    int *idata_st = idata + blockIdx.x * blockDim.x * 8;
    
    // 先进行第一步归约
    if (idx + blockDim.x * 7 < n) {
        idata_st[tid] += idata_st[tid + blockDim.x];
        idata_st[tid] += idata_st[tid + blockDim.x * 2];
        idata_st[tid] += idata_st[tid + blockDim.x * 3];
        idata_st[tid] += idata_st[tid + blockDim.x * 4];
        idata_st[tid] += idata_st[tid + blockDim.x * 5];
        idata_st[tid] += idata_st[tid + blockDim.x * 6];
        idata_st[tid] += idata_st[tid + blockDim.x * 7];
    }
    __syncthreads();

    // 后续归约步骤
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata_st[tid] += idata_st[tid + stride];
        }
        __syncthreads();
    }

    // 保存结果
    if (tid == 0) {
        odata[blockIdx.x] = idata_st[0];
    }
}


__global__ void reduceUnrollWarp8(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + tid;
    
    // 获取当前 block 的起始位置
    int *idata_st = idata + blockIdx.x * blockDim.x * 8;
    
    // 先进行第一步归约
    if (idx + blockDim.x * 7 < n) {
        idata_st[tid] += idata_st[tid + blockDim.x];
        idata_st[tid] += idata_st[tid + blockDim.x * 2];
        idata_st[tid] += idata_st[tid + blockDim.x * 3];
        idata_st[tid] += idata_st[tid + blockDim.x * 4];
        idata_st[tid] += idata_st[tid + blockDim.x * 5];
        idata_st[tid] += idata_st[tid + blockDim.x * 6];
        idata_st[tid] += idata_st[tid + blockDim.x * 7];
    }
    __syncthreads();

    // 后续归约步骤
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata_st[tid] += idata_st[tid + stride];
        }
        __syncthreads();
    }

    // stride>32 剩下的元素个数<64
    if (tid < 32) {
        volatile int *vmem = idata_st;
		vmem[tid] += vmem[tid+32];
		vmem[tid] += vmem[tid+16];
		vmem[tid] += vmem[tid+8];
		vmem[tid] += vmem[tid+4];
		vmem[tid] += vmem[tid+2];
		vmem[tid] += vmem[tid+1];
    }

    // 保存结果
    if (tid == 0) {
        odata[blockIdx.x] = idata_st[0];
    }
}


__global__ void reduceCompleteUnrollWarp8(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + tid;
    
    // 获取当前 block 的起始位置
    int *idata_st = idata + blockIdx.x * blockDim.x * 8;
    
    // 先进行第一步归约
    if (idx + blockDim.x * 7 < n) {
        idata_st[tid] += idata_st[tid + blockDim.x];
        idata_st[tid] += idata_st[tid + blockDim.x * 2];
        idata_st[tid] += idata_st[tid + blockDim.x * 3];
        idata_st[tid] += idata_st[tid + blockDim.x * 4];
        idata_st[tid] += idata_st[tid + blockDim.x * 5];
        idata_st[tid] += idata_st[tid + blockDim.x * 6];
        idata_st[tid] += idata_st[tid + blockDim.x * 7];
    }
    __syncthreads();

    // 完全展开，但有问题 如果 blockDim.x = 2048 就规约不完全了
	if(blockDim.x>=1024 && tid <512)
		idata_st[tid]+=idata_st[tid+512];
	__syncthreads();
	if(blockDim.x>=512 && tid <256)
		idata_st[tid]+=idata_st[tid+256];
	__syncthreads();
	if(blockDim.x>=256 && tid <128)
		idata_st[tid]+=idata_st[tid+128];
	__syncthreads();
	if(blockDim.x>=128 && tid <64)
		idata_st[tid]+=idata_st[tid+64];
	__syncthreads();

    // stride>32 剩下的元素个数<64
    if (tid < 32) {
        volatile int *vmem = idata_st;
		vmem[tid] += vmem[tid+32];
		vmem[tid] += vmem[tid+16];
		vmem[tid] += vmem[tid+8];
		vmem[tid] += vmem[tid+4];
		vmem[tid] += vmem[tid+2];
		vmem[tid] += vmem[tid+1];
    }

    // 保存结果
    if (tid == 0) {
        odata[blockIdx.x] = idata_st[0];
    }
}


template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *idata, int *odata, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + tid;
    
    // 获取当前 block 的起始位置
    int *idata_st = idata + blockIdx.x * blockDim.x * 8;
    
    // 先进行第一步归约
    if (idx + blockDim.x * 7 < n) {
        idata_st[tid] += idata_st[tid + blockDim.x];
        idata_st[tid] += idata_st[tid + blockDim.x * 2];
        idata_st[tid] += idata_st[tid + blockDim.x * 3];
        idata_st[tid] += idata_st[tid + blockDim.x * 4];
        idata_st[tid] += idata_st[tid + blockDim.x * 5];
        idata_st[tid] += idata_st[tid + blockDim.x * 6];
        idata_st[tid] += idata_st[tid + blockDim.x * 7];
    }
    __syncthreads();

    // 完全展开，但有问题 如果 blockDim.x = 2048 就规约不完全了
    // 这里做的只是静态确定了iBlockSize 让编译器删除多余的代码
	if(iBlockSize>=1024 && tid <512)
		idata_st[tid]+=idata_st[tid+512];
	__syncthreads();
	if(iBlockSize>=512 && tid <256)
		idata_st[tid]+=idata_st[tid+256];
	__syncthreads();
	if(iBlockSize>=256 && tid <128)
		idata_st[tid]+=idata_st[tid+128];
	__syncthreads();
	if(iBlockSize>=128 && tid <64)
		idata_st[tid]+=idata_st[tid+64];
	__syncthreads();

    // stride>32 剩下的元素个数<64
    if (tid < 32) {
        volatile int *vmem = idata_st;
		vmem[tid] += vmem[tid+32];
		vmem[tid] += vmem[tid+16];
		vmem[tid] += vmem[tid+8];
		vmem[tid] += vmem[tid+4];
		vmem[tid] += vmem[tid+2];
		vmem[tid] += vmem[tid+1];
    }

    // 保存结果
    if (tid == 0) {
        odata[blockIdx.x] = idata_st[0];
    }
}



int main() 
{
    utills::initDevice(0);

    int nElem = 1 << 20;
    int nBytes = nElem * sizeof(int);

    dim3 block(512, 1, 1);
    // 修改 grid 的计算方式
    dim3 grid((nElem + block.x * 2 - 1) / (block.x * 2), 1, 1);

    int resBytes = grid.x * sizeof(int);
    int *h_a = (int*)malloc(nBytes);
    int *h_b = (int*)malloc(resBytes);

    memset(h_a, 1, nBytes);

    int *d_in = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc((void **)&d_in, nBytes));
    CHECK(cudaMalloc((void **)&d_out, resBytes));

    CHECK(cudaMemcpy(d_in, h_a, nBytes, cudaMemcpyHostToDevice));
    
    CudaTimer timer;
    timer.start();
    reduceUnroll2<<<grid, block>>>(d_in, d_out, nElem);
    timer.stop();
    CHECK(cudaMemcpy(h_b, d_out, resBytes, cudaMemcpyDeviceToHost));
    printf("Time cost [%f] ms \n", timer.elapsed());

    // reduceCompleteUnroll
    // switch(blocksize)
	// {
	// 	case 1024:
	// 		reduceCompleteUnroll <1024><< <grid.x/8, block >> >(idata_dev, odata_dev, size);
	// 	break;
	// 	case 512:
	// 		reduceCompleteUnroll <512><< <grid.x/8, block >> >(idata_dev, odata_dev, size);
	// 	break;
	// 	case 256:
	// 		reduceCompleteUnroll <256><< <grid.x/8, block >> >(idata_dev, odata_dev, size);
	// 	break;
	// 	case 128:
	// 		reduceCompleteUnroll <128><< <grid.x/8, block >> >(idata_dev, odata_dev, size);
	// 	break;
	// }

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_a);
    free(h_b);

    CHECK(cudaDeviceReset());

    return 0;
}