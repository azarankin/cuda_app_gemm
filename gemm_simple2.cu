// gemm_simple2.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "gemm_logic/2_gemm_naive.cuh"
#include "gemm_samples.cuh"
#include "utils.h"

// sample 2
//
//   || ||     || || || ||     || || || ||
//   || ||  X               =  || || || ||
//   || ||     || || || ||     || || || ||

int main()
{
    const gemm::Gemm& data = gemm::basic_sample;

    std::vector<float> h_C = gemm::gemm_naive_run(data);

    std::cout << "sample2 naive gemm, Matrix C = A x B:" << std::endl;
    utils::print_matrix_preview("C", h_C.data(), data.M, data.N);

    return 0;
}
