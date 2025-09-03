cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) 
project(cuda_app VERSION 1.0.0 LANGUAGES CXX CUDA)

#find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#set(CMAKE_CUDA_ARCHITECTURES 72;87;89)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) ## Library build
set(COMPILER_WARNINGS -Wall -Wextra -Wno-unused-parameter)
add_compile_options(${COMPILER_WARNINGS})


find_package(CUDAToolkit)
find_library(CUBLAS_LIB cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)



set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0")
add_library(cuda_debug_opts INTERFACE)
# דגלים רק ב-DEBUG ורק לפי השפה:




add_executable(gemm_simple0 gemm_simple0.cu)
target_link_libraries(gemm_simple0 PRIVATE CUDA::cublas CUDA::cudart)

add_executable(gemm_simple1 gemm_simple1.cu)

add_executable(gemm_simple2 gemm_simple2.cu)

add_executable(gemm_simple3 gemm_simple3.cu)

add_executable(gemm_simple4 gemm_simple4.cu)

add_executable(gemm_simple5 gemm_simple5.cu)

#add_executable(gemm_simple6 gemm_simple6.cu) #not works

#target_link_libraries(cuda_app PRIVATE CUDA::cudart)

#set_target_properties(cuda_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # אוטומטי לארכיטקטורה המקומית; לרוב עדיף מהתעסקות ידנית

# דיבוג נח: סמלים ו-PTX lineinfo רק בקונפיגורציית Debug
#target_compile_options(cuda_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)