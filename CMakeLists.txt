cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_Freshman CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=sm_86 -g -G -O3")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-9")

include_directories(./include)

add_subdirectory(0_hello_world)
add_subdirectory(1_check_dimension)