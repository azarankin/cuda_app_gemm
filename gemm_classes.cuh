#pragma once

namespace gemm
{


    
struct Gemm
{
    std::vector<float> h_A;
    std::vector<float> h_B;
    int M{}, K{}, N{};
};



} // Gemm namespace
