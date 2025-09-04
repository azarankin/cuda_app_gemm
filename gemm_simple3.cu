// gemm_simple3.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "gemm_logic/3_gemm_tiled.cuh"
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

    std::vector<float> h_C = gemm::gemm_tiled_run(data);

    std::cout << "sample3 gemm tiled, Matrix C = A x B:" << std::endl;
    utils::print_matrix_preview("C", h_C.data(), data.M, data.N);

    return 0;
}
