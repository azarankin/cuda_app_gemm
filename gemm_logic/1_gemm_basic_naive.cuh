#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "../gemm_classes.cuh"

namespace gemm
{


__global__ void gemm_basic_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) 
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (!(row < M && col < N))
        return;

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) 
    {
        int a_cell = row * K + i; //*K שורה
        int b_cell = i * N + col; //*N עמודה
        sum += A[a_cell] * B[b_cell];
    }
    int c_cell = row * N + col;
    C[c_cell] = sum;
    //printf("(row,col =(%d,%d)\tC[%d*%d+%d]=%.2f\tC[%d]=%.2f\n", row, col, row, N, col, sum, c_cell, sum);
}


std::vector<float> gemm_basic_naive_run(const Gemm& data)
{
    const std::vector<float>& h_A = data.h_A;
    const std::vector<float>& h_B = data.h_B;
    const int M = data.M, K = data.K, N = data.N;

    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, h_A.size() * sizeof(float));
    cudaMalloc(&d_B, h_B.size() * sizeof(float));
    cudaMalloc(&d_C, h_C.size() * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(N, M);
    gemm_basic_naive_kernel<<<1, threads>>>(d_A, d_B, d_C, M, N, K);

    //cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}


} // Gemm namespace

