#pragma once
#include "gemm_classes.cuh"
#include <vector>

namespace gemm
{
// יצירת אובייקט דוגמה
Gemm basic_sample{
    .h_A = {1, 2,
            3, 4,
            5, 6},
    .h_B = {1, 2, 3, 4,
            5, 6, 7, 8},
    .M = 3, .K = 2, .N = 4
}; 





std::vector<float> h_A_basic {
    1, 2,
    3, 4,
    5, 6
};  // M * K

std::vector<float> h_B_basic {
    1, 2, 3, 4,
    5, 6, 7, 8
};  // K * N

} // Gemm namespace


