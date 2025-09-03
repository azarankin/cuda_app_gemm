//gemm_simple0.cu cuBlas

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.h"

int main() 
{
    const int M = 3, K = 2, N = 4;

    std::vector<float> h_A {
        1, 2,
        3, 4,
        5, 6
    };  // M x K, Row-Major

    std::vector<float> h_B {
        1, 2, 3, 4,
        5, 6, 7, 8
    };  // K x N, Row-Major

    std::vector<float> h_C(M * N, 0.0f);  // התוצאה

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, h_A.size() * sizeof(float));
    cudaMalloc(&d_B, h_B.size() * sizeof(float));
    cudaMalloc(&d_C, h_C.size() * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    // יצירת handle של cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // הסבר: cuBLAS עובד ב־Column-Major, לכן A ו־B מוחלפים
    // כלומר: C = alpha * B × A + beta * C
    // כדי לקבל C = A × B כמו אצלך, צריך להעביר אופרטור transpose לשניהם

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,  // cols of C, rows of C, cols of A / rows of B
                &alpha,
                d_B, N,  // B: N×K (ב-CUBLAS: Column-Major)
                d_A, K,  // A: K×M
                &beta,
                d_C, N); // C: N×M

    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // ניקוי משאבים
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    std::cout << "Matrix C = A x B:" << std::endl;
    utils::print_matrix_preview("C", h_C.data(), M, N);

    return 0;
}
