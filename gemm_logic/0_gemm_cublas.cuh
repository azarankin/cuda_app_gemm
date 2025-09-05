#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cublas_v2.h>
#include "../gemm_classes.cuh"
#include "../gemm_profiling.cuh"

namespace gemm
{

std::vector<float> gemm_cublas_run(const Gemm& data)
{

    const std::vector<float>& h_A = data.h_A;
    const std::vector<float>& h_B = data.h_B;
    const int M = data.M, K = data.K, N = data.N;

    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK << cudaMalloc(&d_A, h_A.size() * sizeof(float));
    CUDA_CHECK << cudaMalloc(&d_B, h_B.size() * sizeof(float));
    CUDA_CHECK << cudaMalloc(&d_C, h_C.size() * sizeof(float));

    CUDA_CHECK << cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK << cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice);

    // יצירת handle של cuBLAS
    cublasHandle_t handle;
    CUDA_CHECK << cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // הסבר: cuBLAS עובד ב־Column-Major, לכן A ו־B מוחלפים
    // כלומר: C = alpha * B × A + beta * C
    // כדי לקבל C = A × B כמו אצלך, צריך להעביר אופרטור transpose לשניהם



CudaProfiler::BEGIN();

PROFILE_RANGE("0. GEMM cublas kernel", NvtxColor::Blue, 

    CUDA_CHECK << cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,  // cols of C, rows of C, cols of A / rows of B
                &alpha,
                d_B, N,  // B: N×K (ב-CUBLAS: Column-Major)
                d_A, K,  // A: K×M
                &beta,
                d_C, N); // C: N×M
    //cudaDeviceSynchronize();

); // PROFILE_RANGE

CudaProfiler::END();


    CUDA_CHECK << cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // ניקוי משאבים
    CUDA_CHECK << cublasDestroy(handle);
    CUDA_CHECK << cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return h_C;
}


} // Gemm namespace

