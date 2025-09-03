#pragma once
#include <iostream>
#include <iomanip>  // std::setw
#include <string>

namespace utils
{
    



void print_matrix_preview_with_cut(const std::string& name,const float* data, int rows, int cols, int cols_stride, int rows_display=10, int cols_display=10) 
{
    int max_rows = std::min(rows, rows_display);
    int max_cols = std::min(cols, cols_display);

    std::cout << "Matrix " << name << " (" << rows << "x" << cols << ") ROI preview (" << max_rows << "x" << max_cols << "):" << std::endl;
    
    for (int i = 0; i < max_rows; ++i) 
    {
        for (int j = 0; j < max_cols; ++j) 
        {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2)
                      << data[i * cols_stride + j] << " ";
        }
        std::cout << std::endl;
    }
}





void print_matrix_preview(const std::string& name,const float* data, int rows, int cols, int rows_display=10, int cols_display=10) 
{
    int max_rows = std::min(rows, rows_display);
    int max_cols = std::min(cols, cols_display);

    std::cout << "Matrix " << name << " (" << rows << "x" << cols << ") preview (" << max_rows << "x" << max_cols << "):" << std::endl;

    for (int i = 0; i < max_rows; ++i) 
    {
        for (int j = 0; j < max_cols; ++j) 
        {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2)
                      << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}


void print_matrix_preview(const float* data, int rows, int cols, int rows_display=10, int cols_display=10)
{
    print_matrix_preview("", data, rows, cols, rows_display, cols_display);
};


} // namespace utils