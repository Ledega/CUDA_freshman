#ifndef FRESHMAN_H
#define FRESHMAN_H

#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

namespace utills {
double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

/**
 * @brief 初始化数据数组，用随机数填充指定大小的浮点数数组
 * @param ip 指向浮点数数组的指针，用于存储生成的随机数
 * @param size 数组的大小，指定要生成的随机数个数
 * @return 无返回值
 */
void initialData(float* ip,int size)
{
  time_t t;
  // 使用当前时间作为随机数种子
  srand((unsigned )time(&t));
  // 生成随机数并填充数组
  for(int i=0;i<size;i++)
  {
    ip[i]=(float)(rand()&0xffff)/1000.0f;
  }
}

void initialData_int(int* ip, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i<size; i++)
	{
		ip[i] = int(rand()&0xff);
	}
}

void printMatrix(float * C, const int nx, const int ny)
{
  printf("Matrix<%d,%d>:\n", ny, nx);
  for(int i = 0; i < ny; i++)
  {
    for(int j = 0; j < nx; j++)
    {
      // 直接通过索引访问，避免指针运算
      printf("%6f ", C[i * nx + j]);
    }
    printf("\n");
  }
}

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: [%s]\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

}

void checkResult(float * hostRef, float * gpuRef, const int N)
{
  double epsilon=1.0E-8;
  for(int i=0;i<N;i++)
  {
    if(abs(hostRef[i]-gpuRef[i])>epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
      return;
    }
  }
  printf("Check result success!\n");
}

}

class CudaTimer {
public:
    CudaTimer() : running(false), last_time_ms(0.0f) {
        cudaError_t err;
        err = cudaEventCreate(&start_event);
        checkError(err, "cudaEventCreate(start)");
        err = cudaEventCreate(&stop_event);
        checkError(err, "cudaEventCreate(stop)");
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        if (running) return;
        cudaError_t err = cudaEventRecord(start_event);
        checkError(err, "cudaEventRecord(start)");
        running = true;
    }

void stop() {
    if (!running) return;
    
    // 记录停止事件
    cudaError_t err = cudaEventRecord(stop_event);
    checkError(err, "cudaEventRecord(stop)");
    
    // 同步停止事件，确保事件之前的所有操作都已完成
    err = cudaEventSynchronize(stop_event);
    checkError(err, "cudaEventSynchronize");
    
    running = false;
    err = cudaEventElapsedTime(&last_time_ms, start_event, stop_event);
    checkError(err, "cudaEventElapsedTime");
}

    // 返回上一次 stop() 到 start() 的时间（毫秒）
    float elapsed() const {
        return last_time_ms;
    }

    // 同步并返回时间（即使没调用 stop）
    float syncAndGet() {
        if (running) {
            stop();
        }
        return last_time_ms;
    }

    // 快捷静态方法：测量单个 kernel
    template<typename KernelFunc, typename... Args>
    static float measure(KernelFunc kernel, int blocks, int threads, Args... args) {
        CudaTimer timer;
        timer.start();
        kernel<<<blocks, threads>>>(args...);
        timer.stop();
        return timer.elapsed();
    }

  private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool running;
    float last_time_ms;

    void checkError(cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            printf("CUDA Error in CudaTimer: %s: %s\n", msg, cudaGetErrorString(err));
        }
    }

};

template<typename T>
class CudaArray
{
public:
  explicit CudaArray(size_t n) : size_(n) {
    CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
  }

  ~CudaArray() {
    CHECK(cudaFree(ptr_))
  }

  CudaArray(const CudaArray&) = delete;
  CudaArray& operator=(const CudaArray&) = delete;

  T* get()          {return ptr_;}
  const T* get()    {return ptr_;}

  private:
    T* ptr_;
    size_t size_;
};




#endif//FRESHMAN_H
