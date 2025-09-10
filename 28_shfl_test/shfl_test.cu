#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include "freshman.h"

#define WARP_SIZE 32
#define HALF_WARP_SIZE (WARP_SIZE / 2)
#define SEGMENT_SIZE 4

__global__ void test_shfl_broadcast(int* in, int *out, const int srcLane)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int val = in[idx];
    
    /*
     * __shfl_sync 函数详解：
     * 
     * 函数原型：
     * __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize)
     * 
     * 参数说明：
     * - mask (0xFFFFFFFF)：参与操作的线程掩码
     *   · 0xFFFFFFFF = 11111111111111111111111111111111 (32位全1)
     *   · 表示warp中所有32个线程都参与shuffle操作
     *   · 可以使用部分掩码，如0x0000FFFF表示只有前16个线程参与
     * 
     * - var (val)：要传输的变量值
     *   · 每个线程都有自己的val值
     *   · 这是要进行shuffle操作的数据
     * 
     * - srcLane (srcLane)：源线程的lane ID
     *   · 指定从哪个线程获取数据
     *   · 在broadcast模式下，所有线程都从这个lane获取数据
     *   · 例如：srcLane = 0 表示所有线程都获取第0号线程的数据
     * 
     * - width (WARP_SIZE = 32)：操作的宽度
     *   · 定义shuffle操作的范围
     *   · 默认是32（整个warp）
     *   · 也可以设置为16、8、4等，将warp分成更小的段
     * 
     * 功能：
     * 这是一个broadcast操作，所有参与的线程都会获得srcLane线程的数据
     * 
     * 示例（假设srcLane = 0）：
     * 线程0的val = 0  →  shuffled_val = 0
     * 线程1的val = 1  →  shuffled_val = 0
     * 线程2的val = 2  →  shuffled_val = 0
     * ...
     * 线程31的val = 31 →  shuffled_val = 0
     * 
     * 性能优势：
     * - 在warp内部进行数据交换，无需使用共享内存
     * - 延迟极低（1个时钟周期）
     * - 不需要同步操作（在同一个warp内）
     */
    int shuffled_val = __shfl_sync(0xFFFFFFFF, val, srcLane, WARP_SIZE);
    out[idx] = shuffled_val;
}

__global__ void test_shfl_up(int* in, int *out, const int delta)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int val = in[idx];
    int shuffled_val = __shfl_up_sync(0xFFFFFFFF, val, delta, WARP_SIZE);
    out[idx] = shuffled_val;
}

__global__ void test_shfl_down(int* in, int *out, const int delta)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int val = in[idx];
    int shuffled_val = __shfl_down_sync(0xFFFFFFFF, val, delta, WARP_SIZE);
    out[idx] = shuffled_val;
}

__global__ void test_shfl_xor(int* in, int *out, const int laneMask)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int val = in[idx];
    int shuffled_val = __shfl_xor_sync(0xFFFFFFFF, val, laneMask, WARP_SIZE);
    out[idx] = shuffled_val;
}


/*
 * __shfl_sync 的其他变体：
 * 
 * 1. __shfl_up_sync(mask, var, delta, width)
 *    - 从更高lane ID的线程获取数据
 *    - 例如：delta=1时，线程i获取线程(i+1)的数据
 * 
 * 2. __shfl_down_sync(mask, var, delta, width)
 *    - 从更低lane ID的线程获取数据
 *    - 例如：delta=1时，线程i获取线程(i-1)的数据
 * 
 * 3. __shfl_xor_sync(mask, var, laneMask, width)
 *    - 基于XOR模式交换数据
 *    - 常用于归约操作
 * 
 * 使用场景：
 * - 数据广播（本例）
 * - warp级归约
 * - 数据重排
 * - 前缀求和
 * - 矩阵转置
 */

int main(int argc, char* argv[])
{
    int mode = 1;
    if (argc > 1) {
        mode = atoi(argv[1]);
    }

    const int num_elements = 64; // 2个warp，每个warp 32个线程
    const int size = num_elements * sizeof(int);

    std::vector<int> h_data(num_elements);
    CudaArray<int> d_data_in(size), d_data_out(size);

    // 初始化输入数据：h_data[i] = i
    for (int i = 0; i < num_elements; i++)
    {
        h_data[i] = i; 
    }

    CHECK(cudaMemcpy(d_data_in.get(), h_data.data(), size, cudaMemcpyHostToDevice));

    // 启动kernel
    dim3 blockDim(32);  // 每个block 32个线程（1个warp）
    dim3 gridDim((num_elements + blockDim.x - 1) / blockDim.x); // 2个block
    int srcLane = 0; // 从第0个线程广播数据
    const int delta = 1; // 上移/下移的线程数

    switch (mode)
    {
    case 0: // 全warp广播
        printf("Broadcasting from lane %d in each warp\n", srcLane);
        test_shfl_broadcast<<<gridDim, blockDim>>>(d_data_in.get(), d_data_out.get(), srcLane);
        break;
    
    case 1: // warp内上移
        printf("Shuffling up by %d lanes in each warp\n", delta);
        test_shfl_up<<<gridDim, blockDim>>>(d_data_in.get(), d_data_out.get(), delta);
        break;  

    case 2: // warp内下移
        printf("Shuffling down by %d lanes in each warp\n", delta);
        test_shfl_down<<<gridDim, blockDim>>>(d_data_in.get(), d_data_out.get(), delta);
        break;  

    case 3: // warp内XOR交换
        printf("Shuffling with XOR by mask %d in each warp\n", delta);
        test_shfl_xor<<<gridDim, blockDim>>>(d_data_in.get(), d_data_out.get(), delta);
        break;  

    default:
        break;
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // 复制结果回主机
    CHECK(cudaMemcpy(h_data.data(), d_data_out.get(), size, cudaMemcpyDeviceToHost));

    // 打印结果
    printf("\nResults after shuffle broadcast:\n");
    for (int i = 0; i < num_elements; i++)
    {
        if (i % 32 == 0) printf("\nWarp %d:\n", i / 32);
        printf("Thread %2d: %2d\n", i, h_data[i]);
    }

    return 0;
}