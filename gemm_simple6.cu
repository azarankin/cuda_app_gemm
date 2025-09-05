// float4 not works
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"

#define TILE_WIDTH 16

#define TILE_WIDTH 16

__global__ void gemm_tiled_stride_float4(const float* A, const float* B, float* C,
                                         int M, int N, int K,
                                         int stride_K, int stride_N) {
    __shared__ __align__(16) float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ __align__(16) float tile_B[TILE_WIDTH][TILE_WIDTH];

    int y = threadIdx.y, x = threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + y;
    int col = blockIdx.x * TILE_WIDTH + x;

    float sum = 0.0f;
    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) {
        int a_idx = row * stride_K + t * TILE_WIDTH;
        int b_idx = (t * TILE_WIDTH + y) * stride_N + col;

        float4 a4 = reinterpret_cast<const float4*>(A + a_idx)[0];
        float4 b4 = reinterpret_cast<const float4*>(B + b_idx)[0];

        int x4 = (x / 4) * 4;
        tile_A[y][x4 + 0] = a4.x;
        tile_A[y][x4 + 1] = a4.y;
        tile_A[y][x4 + 2] = a4.z;
        tile_A[y][x4 + 3] = a4.w;

        int y4 = (y / 4) * 4;
        tile_B[y4 + 0][x] = b4.x;
        tile_B[y4 + 1][x] = b4.y;
        tile_B[y4 + 2][x] = b4.z;
        tile_B[y4 + 3][x] = b4.w;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i += 4) {
            float4 ta = reinterpret_cast<float4*>(&tile_A[y][i])[0];
            float4 tb = reinterpret_cast<float4*>(&tile_B[i][x])[0];
            sum += ta.x * tb.x + ta.y * tb.y + ta.z * tb.z + ta.w * tb.w;
        }
        __syncthreads();
    }

    C[row * stride_N + col] = sum;
}


int roundup(int val, int align) {
    return ((val + align - 1) / align) * align;
}

int main() {
    const int M = 4, K = 4, N = 4; // ודא שהם מחולקים ב־4
    const int stride_K = roundup(K, TILE_WIDTH);
    const int stride_N = roundup(N, TILE_WIDTH);
    const int stride_M = roundup(M, TILE_WIDTH);


    std::vector<float> h_A = {
        1, 2,
        3, 4,
        5, 6
    };

    std::vector<float> h_B = {
        1, 2, 3, 4,
        5, 6, 7, 8
    };

    float *d_A, *d_B, *d_C;
    size_t pitchA = stride_K * sizeof(float);
    size_t pitchB = stride_N * sizeof(float);
    size_t pitchC = stride_N * sizeof(float);

    cudaMalloc(&d_A, stride_K * M * sizeof(float));
    cudaMalloc(&d_B, stride_N * K * sizeof(float));
    cudaMalloc(&d_C, stride_N * M * sizeof(float));
    cudaMemset(d_C, 0, stride_N * M * sizeof(float));

    cudaMemcpy2D(d_A, pitchA, h_A.data(), K * sizeof(float), K * sizeof(float), M, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B, pitchB, h_B.data(), N * sizeof(float), N * sizeof(float), K, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(stride_N / TILE_WIDTH, stride_M / TILE_WIDTH);
    gemm_tiled_stride_float4<<<grid, block>>>(d_A, d_B, d_C, M, N, K, stride_K, stride_N);
    cudaDeviceSynchronize();

    std::vector<float> h_C_dense(M * N);
    cudaMemcpy2D(h_C_dense.data(), N * sizeof(float),
                 d_C, pitchC, N * sizeof(float), M, cudaMemcpyDeviceToHost);

    std::cout<<"Matrix C:\n";
    utils::print_matrix_preview("C", h_C_dense.data(), M, N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    CudaTimer::printAll();
    return 0;
}
