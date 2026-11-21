// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  Image.h - Core image container, filter identifiers and filter metadata.
//
//  Course:        Parallel and Distributed Computing
//  Demonstrates:  shared-memory (OpenMP) + distributed-memory (MPI) processing
//
//  A single Image struct is used by every module (I/O, filters, MPI engine,
//  benchmarking). Pixels are stored interleaved RGB, 8 bits per channel.
// ============================================================================
#ifndef IMAGE_H
#define IMAGE_H

#include <cstdint>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Image container (interleaved RGB, 3 channels)
// ----------------------------------------------------------------------------
struct Image {
    int width = 0;
    int height = 0;
    int channels = 3;                            // always 3 (RGB) in this project
    std::vector<unsigned char> data;             // W * H * 3 bytes, row-major

    std::size_t byte_count() const {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
               static_cast<std::size_t>(channels);
    }
    bool empty() const { return width <= 0 || height <= 0 || data.empty(); }
};

// ----------------------------------------------------------------------------
// Supported filters (Filter Module)
// ----------------------------------------------------------------------------
enum FilterId {
    FILTER_GAUSSIAN = 0,   // Gaussian Blur 5x5
    FILTER_SOBEL,          // Sobel Edge Detection 3x3
    FILTER_SHARPEN,        // Sharpen 3x3
    FILTER_BOX,            // Box (mean) Blur 3x3
    FILTER_GRAYSCALE,      // Grayscale (integer luminance)
    FILTER_MEDIAN,         // Median 3x3 (non-linear)
    FILTER_COUNT
};

FilterId     filter_from_string(const std::string& name);  // returns FILTER_COUNT if unknown
const char*  filter_name(FilterId f);
const char*  filter_description(FilterId f);
int          filter_radius(FilterId f);                    // kernel radius (halo rows for MPI)

// ----------------------------------------------------------------------------
// Synchronization demonstration: checksum computed with an OpenMP critical
// section. Because all pixel math is integer-only, the checksum is bitwise
// identical for sequential, OpenMP, MPI and Hybrid runs - a simple yet strong
// proof that the parallel path reproduces the sequential output exactly
// (no tearing, no lost/corrupted rows).
// ----------------------------------------------------------------------------
unsigned long long image_checksum(const Image& img, bool use_omp);

#endif // IMAGE_H
