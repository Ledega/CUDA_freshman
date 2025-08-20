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

void printMatrix(float * C,const int nx,const int ny)
{
  float *ic=C;
  printf("Matrix<%d,%d>:\n",ny,nx);
  for(int i=0;i<ny;i++)
  {
    for(int j=0;j<nx;j++)
    {
      printf("%6f ",ic[j]);
    }
    ic+=nx;
    printf("\n");
  }
}

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));

}

void checkResult(float * hostRef,float * gpuRef,const int N)
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


#endif//FRESHMAN_H
