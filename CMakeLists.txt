cmake_minimum_required(VERSION 3.18)
project(cuda_app VERSION 1.0.0 LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit)
find_library(CUBLAS_LIB cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

enable_language(CUDA)
add_compile_options(${COMPILER_FLAGS})
add_definitions(${COMPILER_DEFENISIONS})

option(PROFILING_ENABLED "Enable profiling with nsys" OFF)





# Profiling function
function(enable_profiling_for_target target_name)

    set(PROFILE_OUTPUT "${PROFILING_OUTPUT_PATH}/${target_name}.nsys-rep")

    

    # Command to archive old profiling results and create the folders
    file(MAKE_DIRECTORY ${PROFILING_OUTPUT_ARCHIVE_PATH})
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo ">> Archiving previous profiling files for ${target_name}..."
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROFILING_OUTPUT_PATH}" "${PROFILING_OUTPUT_ARCHIVE_PATH}"
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${PROFILING_OUTPUT_PATH}"
        COMMENT "Archiving profiling results"
    )
    file(MAKE_DIRECTORY ${PROFILING_OUTPUT_PATH})

    # Target to view profiling
    add_custom_target(view_profile_${target_name}
        COMMAND ${CMAKE_COMMAND} -E echo ">> Running profiler for ${target_name}..."
        COMMAND bash -c "(nsys profile --capture-range=cudaProfilerApi --stats=true --force-overwrite=true --output=${PROFILE_OUTPUT} '$<TARGET_FILE:${target_name}>' > ${PROFILE_OUTPUT}.log); wait"
        COMMAND ${CMAKE_COMMAND} -E echo ">> Opening profiler UI..."
        COMMAND bash -c "if [ -f ${PROFILE_OUTPUT} ]; then nohup nsys-ui ${PROFILE_OUTPUT} >/dev/null 2>&1 & fi"
        COMMENT "Opening Nsight UI"
        VERBATIM
    )
endfunction()




# CUDA Interface

add_library(CudaDependencies INTERFACE)
target_link_libraries(CudaDependencies INTERFACE
    nvToolsExt
    CUDA::cublas
    CUDA::cudart
)


add_executable(gemm_simple0 gemm_simple0.cu)
target_link_libraries(gemm_simple0 PRIVATE CudaDependencies)
enable_profiling_for_target(gemm_simple0)


add_executable(gemm_simple1 gemm_simple1.cu)
target_link_libraries(gemm_simple1 PRIVATE CudaDependencies)
enable_profiling_for_target(gemm_simple1)


add_executable(gemm_simple2 gemm_simple2.cu)
target_link_libraries(gemm_simple2 PRIVATE CudaDependencies)
enable_profiling_for_target(gemm_simple2)


add_executable(gemm_simple3 gemm_simple3.cu)
target_link_libraries(gemm_simple3 PRIVATE CudaDependencies)
enable_profiling_for_target(gemm_simple3)


add_executable(gemm_simple4 gemm_simple4.cu)
target_link_libraries(gemm_simple4 PRIVATE CudaDependencies)
enable_profiling_for_target(gemm_simple4)


add_executable(gemm_simple5 gemm_simple5.cu)
target_link_libraries(gemm_simple5 PRIVATE CudaDependencies)
enable_profiling_for_target(gemm_simple5)


#add_executable(gemm_simple6 gemm_simple6.cu) #not works

#target_link_libraries(cuda_app PRIVATE CUDA::cudart)

#set_target_properties(cuda_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # אוטומטי לארכיטקטורה המקומית; לרוב עדיף מהתעסקות ידנית

# דיבוג נח: סמלים ו-PTX lineinfo רק בקונפיגורציית Debug
#target_compile_options(cuda_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)