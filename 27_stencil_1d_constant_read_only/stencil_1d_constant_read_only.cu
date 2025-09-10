#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include "freshman.h"

#define TMPL_SIZE 9                     // 模板大小
#define TMPL_RADIO_SIZE (TMPL_SIZE/2)   // 模板半径 = 4
#define BDIM 32                         // 线程块大小

__constant__ float coef[TMPL_RADIO_SIZE];

void convolution(const std::vector<float>& in, 
                 std::vector<float>& out,
                 const std::vector<float>& template_)
{
    // 从半径位置开始，到数组末尾减去半径位置结束，避免越界
    for (int i = TMPL_RADIO_SIZE; i < in.size() - TMPL_RADIO_SIZE; i++)
    {
        // 对每个点应用模板卷积
        for (int j = 1; j <= TMPL_RADIO_SIZE; j++)
        {
            // 计算差分：in[i+j] - in[i-j]，然后乘以对应的模板系数
            out[i] += template_[j-1] * (in[i+j] - in[i-j]);
        }
    }
}

/*
    以下内容AI生成，且未经验证，仅供参考
*/
__global__ void stencil_1d(float* in, float* out)
{
    __shared__ float smem[BDIM + 2 * TMPL_RADIO_SIZE];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int sidx = threadIdx.x + TMPL_RADIO_SIZE;

    smem[sidx] = in[idx];

    if (threadIdx.x < TMPL_RADIO_SIZE)
    {
        // 左边界
        if (idx >= TMPL_RADIO_SIZE)
            smem[sidx - TMPL_RADIO_SIZE] = in[idx - TMPL_RADIO_SIZE];
        else
            smem[sidx - TMPL_RADIO_SIZE] = 0.0f;

        // 右边界
        if (idx + BDIM < gridDim.x * blockDim.x)
            smem[sidx + BDIM] = in[idx + BDIM];
        else
            smem[sidx + BDIM] = 0.0f;
    }

    __syncthreads();

    if (idx >= TMPL_RADIO_SIZE && idx < (gridDim.x * blockDim.x - TMPL_RADIO_SIZE))
    {
        float sum = 0.0f;
        for (int j = 1; j <= TMPL_RADIO_SIZE; j++)
        {
            sum += coef[j-1] * (smem[sidx + j] - smem[sidx - j]);
        }
        out[idx] = sum;
    }
}

__global__ void stencil_1d_readonly(float* in, float* out, const float* __restrict__ dcoef)
{
    __shared__ float smem[BDIM + 2 * TMPL_RADIO_SIZE];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int sidx = threadIdx.x + TMPL_RADIO_SIZE;

    smem[sidx] = in[idx];

    if (threadIdx.x < TMPL_RADIO_SIZE)
    {
        // 左边界
        if (idx >= TMPL_RADIO_SIZE)
            smem[sidx - TMPL_RADIO_SIZE] = in[idx - TMPL_RADIO_SIZE];
        else
            smem[sidx - TMPL_RADIO_SIZE] = 0.0f;

        // 右边界
        if (idx + BDIM < gridDim.x * blockDim.x)
            smem[sidx + BDIM] = in[idx + BDIM];
        else
            smem[sidx + BDIM] = 0.0f;
    }

    __syncthreads();

    if (idx >= TMPL_RADIO_SIZE && idx < (gridDim.x * blockDim.x - TMPL_RADIO_SIZE))
    {
        float sum = 0.0f;
        for (int j = 1; j <= TMPL_RADIO_SIZE; j++)
        {
            sum += dcoef[j-1] * (smem[sidx + j] - smem[sidx - j]);
        }
        out[idx] = sum;
    }
}

int main()
{
    int size = 1 << 20; // 1M elements
    size_t bytes = size * sizeof(float);

    // 分配主机内存
    std::vector<float> h_in(size);
    std::vector<float> h_out(size, 0);
    std::vector<float> h_out_ref(size, 0);
    std::vector<float> h_template(TMPL_RADIO_SIZE);

    // 初始化输入数据和模板
    std::iota(h_in.begin(), h_in.end(), 0.0f);
    for (int i = 0; i < TMPL_RADIO_SIZE; i++)
    {
        h_template[i] = static_cast<float>(i + 1);
    }

    // 分配设备内存
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // 拷贝输入数据到设备
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    // 将模板数据拷贝到常量内存
    cudaMemcpyToSymbol(coef, h_template.data(), TMPL_RADIO_SIZE * sizeof(float));

    // 计算参考结果
    convolution(h_in, h_out_ref, h_template);

    // 配置执行参数
    dim3 block(BDIM);
    dim3 grid((size + block.x - 1) / block.x);

    // 启动核函数
    stencil_1d<<<grid, block>>>(d_in, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    utills::checkResult(h_out_ref.data(), h_out.data(), size);

    // 使用只读数据缓存的核函数
    float *d_coef;
    cudaMalloc(&d_coef, TMPL_RADIO_SIZE * sizeof(float));
    cudaMemcpy(d_coef, h_template.data(), TMPL_RADIO_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // 重置输出数组
    std::fill(h_out.begin(), h_out.end(), 0);
    cudaMemset(d_out, 0, bytes);

    stencil_1d_readonly<<<grid, block>>>(d_in, d_out, d_coef);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    utills::checkResult(h_out_ref.data(), h_out.data(), size);

    // 释放设备内存 
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_coef);
    return 0;
}


