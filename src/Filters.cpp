// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  Filters.cpp - Filter Module implementation (kernels + OpenMP loops).
//
//  OpenMP notes (kept inside the OpenMP 2.0 feature set supported by MSVC):
//    #pragma omp parallel for schedule(static) if(omp)
//  The 'if' clause silently degrades to serial execution when disabled,
//  which is how the same code path serves all four benchmark modes.
// ============================================================================
#include "Filters.h"

#include <cstdio>
#include <cstring>
#include <omp.h>

// ---------------------------------------------------------------------------
// filter metadata (declared in Image.h)
// ---------------------------------------------------------------------------
FilterId filter_from_string(const std::string& name) {
    if (name == "gaussian"   || name == "gaussian_blur") return FILTER_GAUSSIAN;
    if (name == "sobel"      || name == "sobel_edge")    return FILTER_SOBEL;
    if (name == "sharpen")   return FILTER_SHARPEN;
    if (name == "box"        || name == "box_blur")      return FILTER_BOX;
    if (name == "grayscale"  || name == "gray")          return FILTER_GRAYSCALE;
    if (name == "median")    return FILTER_MEDIAN;
    return FILTER_COUNT;
}

const char* filter_name(FilterId f) {
    switch (f) {
        case FILTER_GAUSSIAN:  return "gaussian";
        case FILTER_SOBEL:     return "sobel";
        case FILTER_SHARPEN:   return "sharpen";
        case FILTER_BOX:       return "box";
        case FILTER_GRAYSCALE: return "grayscale";
        case FILTER_MEDIAN:    return "median";
        default:               return "unknown";
    }
}

const char* filter_description(FilterId f) {
    switch (f) {
        case FILTER_GAUSSIAN:  return "Gaussian Blur 5x5 (kernel / 273)";
        case FILTER_SOBEL:     return "Sobel Edge Detection 3x3 (|Gx| + |Gy|)";
        case FILTER_SHARPEN:   return "Sharpen 3x3 (center weight 5)";
        case FILTER_BOX:       return "Box Blur 3x3 (mean, kernel / 9)";
        case FILTER_GRAYSCALE: return "Grayscale (integer luminance)";
        case FILTER_MEDIAN:    return "Median 3x3 (non-linear, per channel)";
        default:               return "";
    }
}

int filter_radius(FilterId f) {
    // Radius also defines the number of halo rows exchanged by the MPI engine.
    switch (f) {
        case FILTER_GAUSSIAN:  return 2;
        case FILTER_SOBEL:     return 1;
        case FILTER_SHARPEN:   return 1;
        case FILTER_BOX:       return 1;
        case FILTER_GRAYSCALE: return 0;
        case FILTER_MEDIAN:    return 1;
        default:               return 0;
    }
}

void print_filter_menu() {
    for (int i = 0; i < FILTER_COUNT; ++i) {
        const FilterId f = static_cast<FilterId>(i);
        std::printf("  %-10s - %s\n", filter_name(f), filter_description(f));
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
namespace {

inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline unsigned char clampuc(int v) {
    return static_cast<unsigned char>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// Integer luminance (weights 77/151/28 sum to 256, close to ITU-R BT.601).
inline int luma(int r, int g, int b) {
    return (77 * r + 151 * g + 28 * b) >> 8;
}

// Gaussian 5x5, sum = 273, sigma ~ 1.4
const int GAUSS_K[5][5] = {
    {  1,  4,  7,  4,  1 },
    {  4, 16, 26, 16,  4 },
    {  7, 26, 41, 26,  7 },
    {  4, 16, 26, 16,  4 },
    {  1,  4,  7,  4,  1 }
};

// 3x3 Sharpen, sum = 1 (no divisor)
const int SHARP_K[3][3] = {
    {  0, -1,  0 },
    { -1,  5, -1 },
    {  0, -1,  0 }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// generic N x N convolution (N = 2*radius+1), supports 3x3 and 5x5
// ---------------------------------------------------------------------------
static void convolve_window(const unsigned char* win, int W, int winH, int C,
                            unsigned char* dst, int y0, int y1, bool omp,
                            const int* K, int radius, int divisor, bool normalize,
                            unsigned char* thread_map) {
    const int ksize = 2 * radius + 1;
    #pragma omp parallel for schedule(static) if(omp)
    for (int y = y0; y < y1; ++y) {                       // output rows (window coords)
        if (thread_map) thread_map[y - y0] = (unsigned char)omp_get_thread_num();
        unsigned char* dstrow = dst + static_cast<std::size_t>(y - y0) * W * C;
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                int sum = 0;
                for (int dy = -radius; dy <= radius; ++dy) {
                    const int sy = clampi(y + dy, 0, winH - 1);   // replicate border
                    const unsigned char* srow = win + static_cast<std::size_t>(sy) * W * C;
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int sx = clampi(x + dx, 0, W - 1);
                        const int kv = K[(dy + radius) * ksize + (dx + radius)];
                        sum += kv * srow[sx * C + c];
                    }
                }
                if (normalize) sum = (sum + divisor / 2) / divisor;   // rounded division
                dstrow[x * C + c] = clampuc(sum);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sobel edge detection: Gx / Gy on integer luminance, |Gx| + |Gy| magnitude
// ---------------------------------------------------------------------------
static void sobel_window(const unsigned char* win, int W, int winH, int C,
                         unsigned char* dst, int y0, int y1, bool omp,
                         unsigned char* thread_map) {
    #pragma omp parallel for schedule(static) if(omp)
    for (int y = y0; y < y1; ++y) {
        if (thread_map) thread_map[y - y0] = (unsigned char)omp_get_thread_num();
        unsigned char* dstrow = dst + static_cast<std::size_t>(y - y0) * W * C;
        for (int x = 0; x < W; ++x) {
            int gx = 0, gy = 0;
            static const int SX[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
            static const int SY[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
            for (int dy = -1; dy <= 1; ++dy) {
                const int sy = clampi(y + dy, 0, winH - 1);
                const unsigned char* srow = win + static_cast<std::size_t>(sy) * W * C;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int sx = clampi(x + dx, 0, W - 1);
                    const int l = luma(srow[sx * C + 0], srow[sx * C + 1], srow[sx * C + 2]);
                    gx += SX[dy + 1][dx + 1] * l;
                    gy += SY[dy + 1][dx + 1] * l;
                }
            }
            const int mag = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
            const unsigned char v = clampuc(mag);
            dstrow[x * C + 0] = v;
            dstrow[x * C + 1] = v;
            dstrow[x * C + 2] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Median 3x3 (non-linear) - per channel, 9 values, insertion sort, take [4]
// ---------------------------------------------------------------------------
static void median_window(const unsigned char* win, int W, int winH, int C,
                          unsigned char* dst, int y0, int y1, bool omp,
                          unsigned char* thread_map) {
    #pragma omp parallel for schedule(static) if(omp)
    for (int y = y0; y < y1; ++y) {
        if (thread_map) thread_map[y - y0] = (unsigned char)omp_get_thread_num();
        unsigned char* dstrow = dst + static_cast<std::size_t>(y - y0) * W * C;
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                int v[9], n = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int sy = clampi(y + dy, 0, winH - 1);
                    const unsigned char* srow = win + static_cast<std::size_t>(sy) * W * C;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int sx = clampi(x + dx, 0, W - 1);
                        v[n++] = srow[sx * C + c];
                    }
                }
                for (int i = 1; i < 9; ++i) {                // insertion sort (tiny n)
                    const int key = v[i];
                    int j = i - 1;
                    while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; --j; }
                    v[j + 1] = key;
                }
                dstrow[x * C + c] = static_cast<unsigned char>(v[4]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grayscale - radius 0, output replicated to RGB so every filter keeps 3 channels
// ---------------------------------------------------------------------------
static void grayscale_window(const unsigned char* win, int W, int winH, int C,
                             unsigned char* dst, int y0, int y1, bool omp,
                             unsigned char* thread_map) {
    (void)winH;
    #pragma omp parallel for schedule(static) if(omp)
    for (int y = y0; y < y1; ++y) {
        if (thread_map) thread_map[y - y0] = (unsigned char)omp_get_thread_num();
        const unsigned char* srcrow = win + static_cast<std::size_t>(y) * W * C;
        unsigned char* dstrow = dst + static_cast<std::size_t>(y - y0) * W * C;
        for (int x = 0; x < W; ++x) {
            const int l = luma(srcrow[x * C + 0], srcrow[x * C + 1], srcrow[x * C + 2]);
            const unsigned char v = static_cast<unsigned char>(l);
            dstrow[x * C + 0] = v;
            dstrow[x * C + 1] = v;
            dstrow[x * C + 2] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// public entry points
// ---------------------------------------------------------------------------
void process_window(const unsigned char* win, int W, int winH, int C,
                    unsigned char* dst, int y0, int y1, FilterId f, bool omp,
                    unsigned char* thread_map) {
    switch (f) {
        case FILTER_GAUSSIAN:
            convolve_window(win, W, winH, C, dst, y0, y1, omp, &GAUSS_K[0][0], 2, 273, true,
                            thread_map);
            break;
        case FILTER_SHARPEN:
            convolve_window(win, W, winH, C, dst, y0, y1, omp, &SHARP_K[0][0], 1, 1, false,
                            thread_map);
            break;
        case FILTER_BOX: {
            const int box[3][3] = { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };
            convolve_window(win, W, winH, C, dst, y0, y1, omp, &box[0][0], 1, 9, true,
                            thread_map);
            break;
        }
        case FILTER_SOBEL:
            sobel_window(win, W, winH, C, dst, y0, y1, omp, thread_map);
            break;
        case FILTER_MEDIAN:
            median_window(win, W, winH, C, dst, y0, y1, omp, thread_map);
            break;
        case FILTER_GRAYSCALE:
            grayscale_window(win, W, winH, C, dst, y0, y1, omp, thread_map);
            break;
        default:
            break;
    }
}

bool apply_filter(const Image& src, Image& dst, FilterId f, bool omp,
                  unsigned char* thread_map) {
    if (src.empty()) return false;
    dst.width = src.width;
    dst.height = src.height;
    dst.channels = src.channels;
    dst.data.resize(src.byte_count());
    process_window(&src.data[0], src.width, src.height, src.channels,
                   &dst.data[0], 0, src.height, f, omp, thread_map);
    return true;
}

// ---------------------------------------------------------------------------
// Synchronization module: OpenMP checksum with a named critical section.
//
// Each thread accumulates a private partial sum over its share of rows and
// the partial sums are merged inside #pragma omp critical, so no two threads
// ever update the shared accumulator at the same time (mutual exclusion).
// ---------------------------------------------------------------------------
unsigned long long image_checksum(const Image& img, bool use_omp) {
    const unsigned char* p = img.data.empty() ? nullptr : &img.data[0];
    const long long n = static_cast<long long>(img.data.size());
    unsigned long long total = 0;

    #pragma omp parallel if(use_omp)
    {
        unsigned long long local = 0;
        #pragma omp for schedule(static)
        for (long long i = 0; i < n; ++i) {
            local += p[i];
        }
        #pragma omp critical(checksum_merge)
        {
            total += local;                 // only one thread writes at a time
        }
    }
    return total;
}
