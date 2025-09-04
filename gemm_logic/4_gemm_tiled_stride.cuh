#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <utility>
#include "../gemm_classes.cuh"

namespace gemm
{
namespace
{
#define TILE_WIDTH 16

int roundup(int val, int align) 
{
    return ((val + align - 1) / align) * align;
}
}

__global__ void gemm_tiled_stride_kernel(const float* A, const float* B, float* C, int M, int N, int K, int stride_A, int stride_B, int stride_C) 
{
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int y = threadIdx.y;
    int x = threadIdx.x;

    int row = blockIdx.y * TILE_WIDTH + y;
    int col = blockIdx.x * TILE_WIDTH + x;

    float sum = 0.0f;

    int num_tiles = (stride_A + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) {
        tile_A[y][x] = A[row * stride_A + t * TILE_WIDTH + x];
        tile_B[y][x] = B[(t * TILE_WIDTH + y) * stride_B + col];
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += tile_A[y][i] * tile_B[i][x];
        }
        __syncthreads();
    }

    C[row * stride_C + col] = sum;
}



Gemm A_B_stride_extend_prepare(const Gemm& data)
{
    const std::vector<float>& h_A = data.h_A;
    const std::vector<float>& h_B = data.h_B;

    const int M = data.M, K = data.K, N = data.N;

    const int stride_M = roundup(M, TILE_WIDTH);
    const int stride_K = roundup(K, TILE_WIDTH);
    const int stride_N = roundup(N, TILE_WIDTH);

    Gemm data_with_stride;
    data_with_stride.h_A = std::vector<float>(stride_M * stride_K, 0.0f);
    data_with_stride.h_B = std::vector<float>(stride_K * stride_N, 0.0f);
    data_with_stride.K = K;
    data_with_stride.M = M;
    data_with_stride.N = N;



    std::vector<float>& h_A_stride = data_with_stride.h_A;
    std::vector<float>& h_B_stride = data_with_stride.h_B;


    // Copy A row by row (M rows, K elements per row)
    for (int i = 0; i < M; ++i) {
        memcpy(h_A_stride.data() + i * stride_K, h_A.data() + i * K, K * sizeof(float));
    }

    // Copy B row by row (K rows, N elements per row)
    for (int i = 0; i < K; ++i) {
        memcpy(h_B_stride.data() + i * stride_N, h_B.data() + i * N, N * sizeof(float));
    }

    return data_with_stride;
}


std::vector<float> gemm_tiled_stride_run(const Gemm& data_with_stride)
{

    const int M = data_with_stride.M, K = data_with_stride.K, N = data_with_stride.N;

    const std::vector<float>& h_A_stride = data_with_stride.h_A;
    const std::vector<float>& h_B_stride = data_with_stride.h_B;


    //std::vector<float> h_C(M * N);

    const int stride_M = roundup(M, TILE_WIDTH);
    const int stride_K = roundup(K, TILE_WIDTH);
    const int stride_N = roundup(N, TILE_WIDTH);

    std::vector<float> h_C_stride(stride_M * stride_N, 0.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, stride_M * stride_K * sizeof(float));
    cudaMalloc(&d_B, stride_K * stride_N * sizeof(float));
    cudaMalloc(&d_C, stride_M * stride_N * sizeof(float));

    cudaMemcpy(d_A, h_A_stride.data(), stride_M * stride_K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_stride.data(), stride_K * stride_N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, stride_M * stride_N * sizeof(float));

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(stride_N / TILE_WIDTH, stride_M / TILE_WIDTH);

    gemm_tiled_stride_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, stride_K, stride_N, stride_N);
    
    //cudaDeviceSynchronize();

    cudaMemcpy(h_C_stride.data(), d_C, stride_M * stride_N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C_stride;
}


} // Gemm namespace
