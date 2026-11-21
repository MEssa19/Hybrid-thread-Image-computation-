// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  Filters.h - Filter Module interface.
//
//  Proposal mapping: "Holds the actual math for the filters (like the
//  convolution kernel) wrapped in OpenMP parallel loops."
//
//  All pixel math uses integer arithmetic only, so results are bitwise
//  identical no matter how many threads/processes produce them.
//
//  process_window() is the single processing primitive used everywhere:
//    * sequential mode  - whole image, OpenMP disabled
//    * OpenMP-only mode - whole image, OpenMP enabled
//    * MPI / Hybrid     - each rank calls it on its window slice
//  ============================================================================
#ifndef FILTERS_H
#define FILTERS_H

#include <string>
#include "Image.h"

// Apply a filter to a window of rows.
//
//   win   : pointer to window pixel data (winH rows, W pixels, C channels)
//   dst   : output buffer for rows [y0, y1) of the window
//           (dst holds (y1 - y0) rows of W*C bytes)
//   f     : filter to apply
//   omp   : enable the OpenMP parallel loop
//   thread_map : optional (y1-y0) bytes written with omp_get_thread_num(),
//           one entry per produced row, used by --visualize to show how the
//           OpenMP work-sharing loop divided the rows among the team.
//           Every index is written by exactly one thread, so this is
//           race-free by construction and does not alter any pixel value.
//
// Border pixels clamp to the window edge ("replicate"). For MPI worker
// slices the window already contains halo rows, so interior rows use real
// neighbouring pixels and only true image borders are clamped.
void process_window(const unsigned char* win, int W, int winH, int C,
                    unsigned char* dst, int y0, int y1, FilterId f, bool omp,
                    unsigned char* thread_map = nullptr);

// Convenience: filter a whole image (used by sequential / OpenMP-only modes).
bool apply_filter(const Image& src, Image& dst, FilterId f, bool omp,
                  unsigned char* thread_map = nullptr);

// Print the filter menu (name + description), one line per filter.
void print_filter_menu();

#endif // FILTERS_H
