// gemm_simple3.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "gemm_logic/3_gemm_tiled.cuh"
#include "gemm_samples.cuh"
#include "gemm_utils.h"

//#define TILE_WIDTH 16
// Each || is 16x16     //to refactoring
//
//   || ||     || || || ||     || || || ||
//   || ||  X               =  || || || ||
//   || ||     || || || ||     || || || ||

int main() {
    const gemm::Gemm& data = gemm::complicated_sample;


PROFILE_REPEAT(
    gemm::gemm_tiled_run(data);
);

    std::vector<float> h_C = gemm::gemm_tiled_run(data);

    std::cout << "sample3 gemm tiled, Matrix C = A x B:" << std::endl;
    utils::print_matrix_preview("C", h_C.data(), data.M, data.N);

    CudaTimer::printAll();
    return 0;
}
