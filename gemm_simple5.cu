// gemm_simple5.cu stride with cudaMemcpy2D
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "gemm_logic/5_gemm_cudamemcpy2d.cuh"
#include "gemm_samples.cuh"
#include "utils.h"

//#define TILE_WIDTH 16
// Each || is 16x16     //to refactoring
//
//   || ||     || || || ||     || || || ||
//   || ||  X               =  || || || ||
//   || ||     || || || ||     || || || ||

int main() {
    const gemm::Gemm& data = gemm::basic_sample;

    std::vector<float> h_C = gemm::gemm_cudamemcpy2d_run(data);

    std::cout << "sample5 gemm cudamemcpy2 tiled, Matrix C = A x B:" << std::endl;
    utils::print_matrix_preview("C", h_C.data(), data.M, data.N);

    CudaTimer::printAll();
    return 0;
}
