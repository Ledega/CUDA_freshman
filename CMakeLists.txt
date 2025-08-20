cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_Freshman CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=sm_86 -g -G -O3")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-9")
message(STATUS "Using GCC: ${CMAKE_CUDA_HOST_COMPILER}")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")

include_directories(./include)

add_subdirectory(0_hello_world)
add_subdirectory(1_check_dimension)
add_subdirectory(2_grid_block)
add_subdirectory(3_sum_arrays)
add_subdirectory(4_sum_arrays_timer)