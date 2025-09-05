#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "../gemm_classes.cuh"
#include "../gemm_profiling.cuh"

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

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) 
{
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH]; // גודל טיל עבור A
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH]; // גודל טיל עבור B

    // אינדקסים מקומיים בתוך הבלוק
    int y = threadIdx.y;
    int x = threadIdx.x;

    // אינדקסים גלובליים בתוך מטריצת הפלט
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    // כמה צעדים צריך לעבור כדי לכסות את כל K
    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) 
    {
        tile_A[y][x] = 0.0f;
        tile_B[y][x] = 0.0f;
    }

    for (int t = 0; t < num_tiles; ++t) 
    {
        // אינדקסים פנימיים של האלמנטים מהטיל הנוכחי
        int a_col = t * TILE_WIDTH + x;
        int b_row = t * TILE_WIDTH + y;

        // טען ערכים לזיכרון שיתופי עם בדיקת גבולות
        if(row < M && a_col < K)
        {
            int a_cell = row * K + a_col;
            tile_A[y][x] =  A[a_cell];
        }

        if(b_row < K && col < N)
        {
            int b_cell = b_row * N + col;
            tile_B[y][x] = B[b_cell];
        }

        __syncthreads();

        // חישוב חלקי של המכפלה
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += tile_A[y][i] * tile_B[i][x];
        }

        __syncthreads();
    }

    // כתיבת התוצאה הסופית עם בדיקת גבולות
    if (!(row < M && col < N))
        return;
    
    int c_cell = row * N + col;
    C[c_cell] = sum;
    
}


std::vector<float> gemm_tiled_run(const Gemm& data)
{

    const std::vector<float>& h_A = data.h_A;
    const std::vector<float>& h_B = data.h_B;
    const int M = data.M, K = data.K, N = data.N;

    std::vector<float> h_C(M * N, 0.0f);


    float *d_A, *d_B, *d_C;
    CUDA_CHECK << cudaMalloc(&d_A, M * K * sizeof(float));
    CUDA_CHECK << cudaMalloc(&d_B, K * N * sizeof(float));
    CUDA_CHECK << cudaMalloc(&d_C, M * N * sizeof(float));

    CUDA_CHECK << cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK << cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);



CudaProfiler::BEGIN();

PROFILE_RANGE("3. GEMM tile kernel", NvtxColor::Blue, 

    gemm_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK();
    //cudaDeviceSynchronize();

); // PROFILE_RANGE

CudaProfiler::END();



    CUDA_CHECK << cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);


    CUDA_CHECK << cudaFree(d_A);
    CUDA_CHECK << cudaFree(d_B);
    CUDA_CHECK << cudaFree(d_C);
    
    return h_C;
}


} // Gemm namespace
