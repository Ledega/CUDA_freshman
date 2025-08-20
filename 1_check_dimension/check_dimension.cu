#include <stdio.h>
#include <cuda_runtime.h>

__global__ void check_dimension(void) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("[DEVICE] grdDim: (%d, %d, %d), blockDim: (%d, %d, %d), blockIdx: (%d, %d, %d), threadIdx: (%d, %d, %d) => global idx: %d\n",
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           idx);
}

int main() {
    int nElem = 6;
    dim3 block(3, 1, 1);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("[HOST] grid: (%d, %d, %d), block: (%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    check_dimension<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
