cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# https://gitlab.kitware.com/cmake/cmake/-/issues/17471
# and the linked merge https://gitlab.kitware.com/cmake/cmake/-/merge_requests/1341
set(CMAKE_CUDA_HOST_COMPILER g++)
set(CMAKE_CXX_COMPILER g++)

#set(CMAKE_BUILD_TYPE Debug)

# need CXX here so that mpi finds the correct stuff
project(cuda_blob CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_executable(cuda_blob main.cu stacktrace.h prettyprint.h cutil_inline_runtime.h helper_cuda.h helper_string.h util.h)
set_target_properties(
    cuda_blob
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70-real
)
target_link_libraries(cuda_blob ${OpenCV_LIBS} CUDA::cufft)
