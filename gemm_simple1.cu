#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"

// sample
//
//   || ||     || || || ||     || || || ||
//   || ||  X               =  || || || ||
//   || ||     || || || ||     || || || ||

__global__ void gemm_small(const float* A, const float* B, float* C, int M, int N, int K) 
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
    printf("(row,col =(%d,%d)\tC[%d*%d+%d]=%.2f\tC[%d]=%.2f\n", row, col, row, N, col, sum, c_cell, sum);
}



int main() 
{
    const int M = 3, K = 2, N = 4;

    std::vector<float> h_A {
        1, 2,
        3, 4,
        5, 6
    };  // M * K

    std::vector<float> h_B {
        1, 2, 3, 4,
        5, 6, 7, 8
    };  // K * N

    std::vector<float> h_C(M * N);  // מאופס אוטומטית

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, h_A.size() * sizeof(float));
    cudaMalloc(&d_B, h_B.size() * sizeof(float));
    cudaMalloc(&d_C, h_C.size() * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(N, M);  // (4, 3)
    gemm_small<<<1, threads>>>(d_A, d_B, d_C, M, N, K);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    

    std::cout << "Matrix C = A x B:" << std::endl;
    utils::print_matrix_preview("C", h_C.data(), M, N);
    
    return 0;
}
