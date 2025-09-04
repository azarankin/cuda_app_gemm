#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "../gemm_classes.cuh"

namespace gemm
{

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(row < M && col < N))
        return;
    
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        int a_cell = row * K + i; //*K שורה
        int b_cell = i * N + col; //*N עמודה
        sum += A[a_cell] * B[b_cell];
    }

    int c_cell = row * N + col;
    
    C[c_cell] = sum;
    //printf("(row,col =(%d,%d)\tC[%d*%d+%d]=%.2f\tC[%d]=%.2f\n", row, col, row, N, col, sum, c_cell, sum);
}



std::vector<float> gemm_naive_run(const Gemm& data)
{

    const std::vector<float>& h_A = data.h_A;
    const std::vector<float>& h_B = data.h_B;
    const int M = data.M, K = data.K, N = data.N;

    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);//threads
    dim3 grid((N + 15) / 16, (M + 15) / 16);//blocks

    gemm_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    //cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}


} // Gemm namespace
