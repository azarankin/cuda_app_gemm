#pragma once
#include <iostream>
#include <iomanip>  // std::setw
#include <string>
#include <algorithm>
namespace utils
{



static inline void print_matrix_edges_impl(const std::string& name,
                                           const float* data,
                                           int rows, int cols,
                                           int row_stride,     // ld (ביחידות אלמנטים)
                                           int edge = 2,
                                           int width = 8)
{
    std::cout << "Matrix " << name << " (" << rows << "x" << cols
              << ") preview %1000 (edges=" << edge << "):\n";

    auto print_cell = [&](float v) {
        std::cout << std::setw(width) << std::fixed << std::setprecision(2) << v - 1000;
    };
    auto print_cells_range = [&](int r, int c0, int c1) {
        // [c0, c1)
        for (int c = c0; c < c1; ++c)
            print_cell(data[r * row_stride + c]);
    };

    auto print_col_ellipsis = [&]() {
        std::cout << std::setw(4) << "...";
    };

    auto print_row_edges = [&](int r) {
        const int left_cnt = std::min(edge, cols);
        print_cells_range(r, 0, left_cnt);

        if (cols > 2 * edge) {
            print_col_ellipsis();
            print_cells_range(r, cols - edge, cols);
        } else if (cols > left_cnt) {
            // אין צורך ב"..." אם אין אמצע — הדפס את היתרה
            print_cells_range(r, left_cnt, cols);
        }
        std::cout << '\n';
    };

    // למעלה
    const int top_rows = std::min(edge, rows);
    for (int r = 0; r < top_rows; ++r)
        print_row_edges(r);

    // "..." בין עליון לתחתון
    if (rows > 2 * edge)
        std::cout << "...\n";

    // למטה
    const int bottom_start = (rows > 2 * edge) ? (rows - edge) : top_rows;
    for (int r = bottom_start; r < rows; ++r)
        print_row_edges(r);
}


// ---- API 1: floent (row_stride = cols) ----
inline void print_matrix_preview(const std::string& name,
                                 const float* data,
                                 int rows, int cols,
                                 int edge = 2, int width = 8)
{
    print_matrix_edges_impl(name, data, rows, cols, /*row_stride=*/cols, edge, width);
}



// // ---- API 2: strided (pitch/LD) ----
// inline void print_matrix_preview(const std::string& name,
//                                  const float* data,
//                                  int rows, int cols,
//                                  int row_stride,   // ידני
//                                  int edge, int width = 8)
// {
//     print_matrix_edges_impl(name, data, rows, cols, row_stride, edge, width);
// }


// inline void print_matrix_preview(const float* data,
//                                  int rows, int cols,
//                                  int edge = 2, int width = 8)
// {
//     print_matrix_edges_impl("", data, rows, cols, /*row_stride=*/cols, edge, width);
// }




inline void print_matrix_preview_with_cut(const std::string& name,const float* data, int rows, int cols, int cols_stride) 
{
        print_matrix_edges_impl(name, data, rows, cols, cols_stride, 2, 8);
}


} // namespace utils