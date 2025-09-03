// gemm_simple5.cu // stride with  cudaMemcpy2D

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"

#define TILE_WIDTH 16

__global__ void gemm_tiled_stride(const float* A, const float* B, float* C,
                                  int M, int N, int K,
                                  int stride_K, int stride_N) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int y = threadIdx.y, x = threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + y;
    int col = blockIdx.x * TILE_WIDTH + x;

    float sum = 0.0f;
    int num_tiles = (stride_K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) {
        tile_A[y][x] = A[row * stride_K + t * TILE_WIDTH + x];
        tile_B[y][x] = B[(t * TILE_WIDTH + y) * stride_N + col];
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += tile_A[y][i] * tile_B[i][x];
        }
        __syncthreads();
    }

    C[row * stride_N + col] = sum;
}

int roundup(int val, int align) 
{
    return ((val + align - 1) / align) * align;
}

int main() {
    const int M = 3, K = 2, N = 4;
    const int stride_K = roundup(K, TILE_WIDTH);
    const int stride_N = roundup(N, TILE_WIDTH);
    const int stride_M = roundup(M, TILE_WIDTH);

    // Compact host arrays

    std::vector<float> h_A = {
        1, 2,
        3, 4,
        5, 6
    };

    std::vector<float> h_B = {
        1, 2, 3, 4,
        5, 6, 7, 8
    };

    // Device pitched allocations (manual stride) using cudaMallocPitch not used for simplicity
    float *d_A, *d_B, *d_C;
    size_t pitchA = stride_K * sizeof(float);
    size_t pitchB = stride_N * sizeof(float);
    size_t pitchC = stride_N * sizeof(float);

    cudaMalloc(&d_A, stride_K * M * sizeof(float));
    cudaMalloc(&d_B, stride_N * K * sizeof(float));
    cudaMalloc(&d_C, stride_N * M * sizeof(float));
    cudaMemset(d_C, 0, stride_N * M * sizeof(float));

    // Copy compact A to padded device memory efficiently
    cudaMemcpy2D(d_A, pitchA, h_A.data(), K * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B, pitchB, h_B.data(), N * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(stride_N / TILE_WIDTH, stride_M / TILE_WIDTH);
    gemm_tiled_stride<<<grid, block>>>(d_A, d_B, d_C, M, N, K, stride_K, stride_N);
    cudaDeviceSynchronize();

    // Copy ROI (M x N) back to compact host array
    std::vector<float> h_C_dense(M * N);
    cudaMemcpy2D(h_C_dense.data(), N * sizeof(float),
                 d_C, pitchC, N * sizeof(float), M, cudaMemcpyDeviceToHost);

    std::cout << "Matrix C (ROI " << M << "x" << N << "):" << std::endl;
    utils::print_matrix_preview("C", h_C_dense.data(), M, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
