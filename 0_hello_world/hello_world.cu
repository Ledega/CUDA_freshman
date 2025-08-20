#include<stdio.h>

__global__ void hello_world(void) {
    printf("GPU: Hello world!\n");
}

int main() {
    printf("GPU: Hello world!\n");
    hello_world<<<1,10>>>();
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    cudaDeviceReset();
    printf("GPU: Hello world finished!\n");
    return 0;
}