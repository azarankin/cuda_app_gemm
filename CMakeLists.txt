cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) 
project(cuda_app VERSION 1.0.0 LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#set(CMAKE_CUDA_ARCHITECTURES 72;87;89)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) ## Library build
set(COMPILER_WARNINGS -Wall -Wextra -Wno-unused-parameter)
add_compile_options(${COMPILER_WARNINGS})


add_executable(cuda_app gemm.cu)

#set_target_properties(cuda_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # אוטומטי לארכיטקטורה המקומית; לרוב עדיף מהתעסקות ידנית

# דיבוג נח: סמלים ו-PTX lineinfo רק בקונפיגורציית Debug
target_compile_options(cuda_app PRIVATE
  $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>>
  $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CXX>:-g>>
)
