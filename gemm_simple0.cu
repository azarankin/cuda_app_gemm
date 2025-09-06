//gemm_simple0.cu cuBlas
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
//#include <cublas_v2.h>
#include "gemm_logic/0_gemm_cublas.cuh"
#include "gemm_samples.cuh"
#include "gemm_utils.h"

int main() 
{
    const gemm::Gemm& data = gemm::complicated_sample;

    std::vector<float> h_C = gemm::gemm_cublas_run(data);

    std::cout << "sample0 cuBlas gemm,Matrix C = A x B:" << std::endl;
    utils::print_matrix_preview("C", h_C.data(), data.M, data.N);

    CudaTimer::printAll();
    return 0;
}
