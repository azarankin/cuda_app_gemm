// gemm_simple4.cu
#include <iostream>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include "gemm_logic/4_gemm_tiled_stride.cuh"
#include "gemm_samples.cuh"
#include "utils.h"

//#define TILE_WIDTH 16
// Each || is 16x16     //to refactoring
//
//   || ||     || || || ||     || || || ||
//   || ||  X               =  || || || ||
//   || ||     || || || ||     || || || ||


namespace
{
int roundup(int val, int align) 
{
    return ((val + align - 1) / align) * align;
}
}

int main() {
    const gemm::Gemm& data = gemm::basic_sample;
    gemm::Gemm data_with_stride = gemm::A_B_stride_extend_prepare(data);
    std::vector<float> h_C_stride = gemm::gemm_tiled_stride_run(data_with_stride);

    std::cout << "sample4 gemm tiled stride, Matrix C = A x B:" << std::endl;
    const int stride_N = roundup(data.N, TILE_WIDTH);
    utils::print_matrix_preview_with_cut("C_ROI", h_C_stride.data(), data.M, data.N, stride_N);

    CudaTimer::printAll();
    return 0;
}