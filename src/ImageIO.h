// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  ImageIO.h - I/O and File Management module.
//
//  Reads/writes uncompressed 24-bit images with no external libraries:
//    * PPM  - binary P6 (and ascii P3 for reading)
//    * BMP  - 24-bit uncompressed (BI_RGB)
//  The format is selected by the file extension (.ppm / .bmp).
// ============================================================================
#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <string>
#include "Image.h"

// Load an image. Returns false and fills 'err' on failure.
bool load_image(const std::string& path, Image& img, std::string& err);

// Save an image. Format is chosen from the file extension.
bool save_image(const std::string& path, const Image& img, std::string& err);

// Lower-cased extension (".ppm", ".bmp", ...) or "" if none.
std::string file_extension(const std::string& path);

// Build "<dir><stem>_<filter>_<mode>.<ext>" for default output naming.
std::string make_output_path(const std::string& input, const std::string& filter,
                             const std::string& mode);

// Generate a colorful synthetic test image (gradient + stripes + checkerboard
// + circles) so the engine can be demoed without any external picture file.
// Output format follows the extension of 'out_path' (.ppm or .bmp).
// Supported size range: 1x1 .. 8192x8192 (tiny images exercise the
// empty-band edge case, big ones the benchmark case).
bool make_test_image(const std::string& out_path, int W, int H, std::string& err);

#endif // IMAGE_IO_H
