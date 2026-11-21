// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  ImageIO.cpp - I/O and File Management module implementation.
//
//  Proposal mapping: "Reads standard uncompressed images (like BMP or PPM)
//  into memory and writes the final output to a file."
//
//  Only standard C library calls (fopen/fread/fwrite) are used, so the code
//  compiles unchanged with MSVC (Visual Studio) and GCC/MinGW.
// ============================================================================
#define _CRT_SECURE_NO_WARNINGS

#include "ImageIO.h"

#include <cstdio>
#include <cstring>
#include <cctype>

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
namespace {

unsigned int rd_le32(const unsigned char* p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) |
           ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

unsigned short rd_le16(const unsigned char* p) {
    return (unsigned short)((unsigned short)p[0] | ((unsigned short)p[1] << 8));
}

void wr_le32(unsigned char* p, unsigned int v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
    p[2] = (unsigned char)((v >> 16) & 0xFF);
    p[3] = (unsigned char)((v >> 24) & 0xFF);
}

void wr_le16(unsigned char* p, unsigned int v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
}

bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// PPM (P6 binary / P3 ascii) - header is "P6\n<comments>\nW H\nmaxval\n"
// ---------------------------------------------------------------------------
bool load_ppm(const std::string& path, Image& img, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open file: " + path; return false; }

    char magic[3] = { 0, 0, 0 };
    if (std::fread(magic, 1, 2, f) != 2) { std::fclose(f); err = "short read (magic)"; return false; }

    const bool binary = (magic[0] == 'P' && magic[1] == '6');
    const bool ascii  = (magic[0] == 'P' && magic[1] == '3');
    if (!binary && !ascii) {
        std::fclose(f);
        err = "unsupported PPM variant (need P6 or P3)";
        return false;
    }

    // header helper: read next integer, skipping whitespace and '#' comments
    auto next_int = [&f](bool& ok) -> long {
        int c = std::fgetc(f);
        while (c != EOF) {
            if (c == '#') {                       // comment to end of line
                while (c != EOF && c != '\n') c = std::fgetc(f);
            } else if (std::isspace(c)) {
                c = std::fgetc(f);
            } else {
                break;
            }
        }
        if (c == EOF || !std::isdigit(c)) { ok = false; return 0; }
        long v = 0;
        while (c != EOF && std::isdigit(c)) {
            v = v * 10 + (c - '0');
            c = std::fgetc(f);
        }
        ok = true;
        return v;
    };

    bool ok = true;
    const long w = next_int(ok);
    const long h = ok ? next_int(ok) : 0;
    const long maxval = ok ? next_int(ok) : 0;
    if (!ok || w <= 0 || h <= 0 || maxval <= 0 || maxval > 255) {
        std::fclose(f);
        err = "invalid PPM header (size or maxval; maxval>255 unsupported)";
        return false;
    }

    img.width = static_cast<int>(w);
    img.height = static_cast<int>(h);
    img.channels = 3;
    img.data.resize(img.byte_count());

    if (binary) {
        // next_int() already consumed the single whitespace after maxval.
        // Tolerate an extra separator (CRLF files) without eating pixel data.
        int c = std::fgetc(f);
        if (c == '\r') {
            int c2 = std::fgetc(f); (void)c2;            // skip LF after CR
        } else if (c == '\n' || c == ' ' || c == '\t') {
            // consumed: was an extra separator
        } else if (c != EOF) {
            std::ungetc(c, f);                           // real pixel byte, push back
        }
        const std::size_t n = img.data.size();
        const std::size_t got = std::fread(&img.data[0], 1, n, f);
        std::fclose(f);
        if (got != n) { err = "truncated PPM pixel data"; return false; }
    } else {
        const std::size_t n = img.data.size();
        for (std::size_t i = 0; i < n; ++i) {
            bool tok_ok = true;
            long v = next_int(tok_ok);
            if (!tok_ok || v > 255) { std::fclose(f); err = "invalid P3 pixel data"; return false; }
            img.data[i] = static_cast<unsigned char>(v);
        }
        std::fclose(f);
    }
    return true;
}

bool save_ppm(const std::string& path, const Image& img, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot open file for writing: " + path; return false; }
    std::fprintf(f, "P6\n# Hybrid Distributed Image Processing Engine\n%d %d\n255\n",
                 img.width, img.height);
    const std::size_t n = img.data.size();
    const std::size_t put = n ? std::fwrite(&img.data[0], 1, n, f) : 0;
    std::fclose(f);
    if (put != n) { err = "short write on PPM"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// BMP 24-bit uncompressed (BI_RGB)
// ---------------------------------------------------------------------------
bool load_bmp(const std::string& path, Image& img, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open file: " + path; return false; }

    // read the whole file (BMP files in this project are modest in size)
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); err = "seek failed"; return false; }
    long fsize = std::ftell(f);
    if (fsize < 54) { std::fclose(f); err = "file too small to be a BMP"; return false; }
    std::fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf(static_cast<std::size_t>(fsize));
    if (std::fread(&buf[0], 1, buf.size(), f) != buf.size()) {
        std::fclose(f); err = "short read on BMP"; return false;
    }
    std::fclose(f);

    if (buf[0] != 'B' || buf[1] != 'M') { err = "not a BMP file (missing BM magic)"; return false; }
    const unsigned int data_offset = rd_le32(&buf[10]);
    const unsigned int header_size = rd_le32(&buf[14]);
    if (header_size < 40) { err = "unsupported BMP header (need BITMAPINFOHEADER)"; return false; }

    const int  w = static_cast<int>(rd_le32(&buf[18]));
    const int  h_raw = static_cast<int>(rd_le32(&buf[22]));     // may be negative (top-down)
    const unsigned short planes = rd_le16(&buf[26]);
    const unsigned short bpp    = rd_le16(&buf[28]);
    const unsigned int   comp   = rd_le32(&buf[30]);

    if (planes != 1)                { err = "unsupported BMP (planes != 1)"; return false; }
    if (bpp != 24)                  { err = "unsupported BMP (only 24-bit uncompressed is supported)"; return false; }
    if (comp != 0)                  { err = "unsupported BMP (compressed; only BI_RGB supported)"; return false; }
    if (w <= 0 || h_raw == 0 || w > 1000000 || (h_raw > 0 ? h_raw : -h_raw) > 1000000) {
        err = "invalid BMP dimensions"; return false;
    }
    const bool top_down = (h_raw < 0);
    const int  h = top_down ? -h_raw : h_raw;
    const int  stride = ((w * 3 + 3) / 4) * 4;                   // rows padded to 4 bytes

    const std::size_t needed = static_cast<std::size_t>(data_offset) +
                               static_cast<std::size_t>(stride) * static_cast<std::size_t>(h);
    if (buf.size() < needed) { err = "truncated BMP pixel data"; return false; }

    img.width = w; img.height = h; img.channels = 3;
    img.data.resize(img.byte_count());

    for (int y = 0; y < h; ++y) {
        const int src_row = top_down ? y : (h - 1 - y);          // BMP rows are bottom-up
        const unsigned char* src = &buf[data_offset] + static_cast<std::size_t>(src_row) * stride;
        unsigned char* dst = &img.data[static_cast<std::size_t>(y) * w * 3];
        for (int x = 0; x < w; ++x) {                            // BGR -> RGB
            dst[x * 3 + 0] = src[x * 3 + 2];
            dst[x * 3 + 1] = src[x * 3 + 1];
            dst[x * 3 + 2] = src[x * 3 + 0];
        }
    }
    return true;
}

bool save_bmp(const std::string& path, const Image& img, std::string& err) {
    const int w = img.width;
    const int h = img.height;
    const int stride = ((w * 3 + 3) / 4) * 4;
    const unsigned int data_size = static_cast<unsigned int>(stride) * static_cast<unsigned int>(h);
    const unsigned int file_size = 14 + 40 + data_size;

    unsigned char fh[14]; unsigned char ih[40];
    std::memset(fh, 0, sizeof(fh)); std::memset(ih, 0, sizeof(ih));

    fh[0] = 'B'; fh[1] = 'M';
    wr_le32(fh + 2, file_size);
    wr_le32(fh + 10, 14 + 40);                                   // pixel data offset

    wr_le32(ih + 0, 40);                                         // biSize
    wr_le32(ih + 4, static_cast<unsigned int>(w));               // biWidth
    wr_le32(ih + 8, static_cast<unsigned int>(h));               // biHeight (positive = bottom-up)
    wr_le16(ih + 12, 1);                                         // biPlanes
    wr_le16(ih + 14, 24);                                        // biBitCount
    wr_le32(ih + 16, 0);                                         // biCompression = BI_RGB
    wr_le32(ih + 20, data_size);                                 // biSizeImage
    wr_le32(ih + 24, 2835);                                      // 72 dpi
    wr_le32(ih + 28, 2835);

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot open file for writing: " + path; return false; }
    bool ok = (std::fwrite(fh, 1, 14, f) == 14) && (std::fwrite(ih, 1, 40, f) == 40);
    std::vector<unsigned char> row(static_cast<std::size_t>(stride), 0);
    for (int y = h - 1; ok && y >= 0; --y) {                     // bottom-up, BGR
        const unsigned char* src = &img.data[static_cast<std::size_t>(y) * w * 3];
        for (int x = 0; x < w; ++x) {
            row[static_cast<std::size_t>(x) * 3 + 0] = src[x * 3 + 2];
            row[static_cast<std::size_t>(x) * 3 + 1] = src[x * 3 + 1];
            row[static_cast<std::size_t>(x) * 3 + 2] = src[x * 3 + 0];
        }
        ok = (std::fwrite(&row[0], 1, static_cast<std::size_t>(stride), f) == static_cast<std::size_t>(stride));
    }
    std::fclose(f);
    if (!ok) { err = "short write on BMP"; return false; }
    return true;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------
std::string file_extension(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return "";
    std::string ext = path.substr(dot);
    for (std::size_t i = 0; i < ext.size(); ++i) ext[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(ext[i])));
    return ext;
}

bool load_image(const std::string& path, Image& img, std::string& err) {
    if (!file_exists(path)) { err = "file not found: " + path; return false; }
    const std::string ext = file_extension(path);
    if (ext == ".ppm")      return load_ppm(path, img, err);
    if (ext == ".bmp")      return load_bmp(path, img, err);
    err = "unsupported input format \"" + ext + "\" (use .ppm or .bmp)";
    return false;
}

bool save_image(const std::string& path, const Image& img, std::string& err) {
    if (img.empty()) { err = "refusing to save empty image"; return false; }
    const std::string ext = file_extension(path);
    if (ext == ".ppm")      return save_ppm(path, img, err);
    if (ext == ".bmp")      return save_bmp(path, img, err);
    err = "unsupported output format \"" + ext + "\" (use .ppm or .bmp)";
    return false;
}

std::string make_output_path(const std::string& input, const std::string& filter,
                             const std::string& mode) {
    const std::size_t slash = input.find_last_of("/\\");
    const std::size_t dot = input.find_last_of('.');
    const std::size_t stem_end = (dot != std::string::npos &&
                                  (slash == std::string::npos || dot > slash)) ? dot : input.size();
    const std::size_t stem_start = (slash != std::string::npos) ? slash + 1 : 0;
    const std::string dir = (slash != std::string::npos) ? input.substr(0, slash + 1) : "";
    const std::string stem = input.substr(stem_start, stem_end - stem_start);
    std::string ext = file_extension(input);
    if (ext.empty()) ext = ".ppm";   // fallback; caller normally validates first
    return dir + stem + "_" + filter + "_" + mode + ext;
}

// ---------------------------------------------------------------------------
// make_test_image - synthetic demo picture, pure deterministic integer math.
//
// Composition (chosen so every filter has something to work on):
//   * diagonal two-color gradient background
//   * soft diagonal stripes (48 px period)
//   * checkerboard in the bottom-right quadrant (24 px blocks)
//   * white ring (outlines survive blur, edges light up in Sobel)
//   * solid teal disc + small yellow disc (flat regions for median/box)
// ---------------------------------------------------------------------------
bool make_test_image(const std::string& out_path, int W, int H, std::string& err) {
    if (W < 1 || H < 1 || W > 8192 || H > 8192) {
        err = "test image size must be between 1x1 and 8192x8192";
        return false;
    }

    Image img;
    img.width = W;
    img.height = H;
    img.channels = 3;
    img.data.resize(img.byte_count());

    const long long minWH = (W < H ? W : H);
    const long long ring_cx = W * 32 / 100, ring_cy = H * 38 / 100;
    const long long ring_r  = minWH * 22 / 100;
    const long long ring_r2 = ring_r * ring_r;
    const long long disc_cx = W * 72 / 100, disc_cy = H * 62 / 100;
    const long long disc_r2 = (minWH * 12 / 100) * (minWH * 12 / 100);
    const long long dot_cx  = W * 55 / 100, dot_cy = H * 25 / 100;
    const long long dot_r2  = (minWH * 6 / 100) * (minWH * 6 / 100);
    int block = minWH < 48 ? static_cast<int>(minWH / 4) : 24;   // checker/stripes scale down
    if (block < 1) block = 1;
    const int denom_w = W > 1 ? W - 1 : 1;           // avoid /0 on degenerate sizes
    const int denom_h = H > 1 ? H - 1 : 1;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // base: diagonal two-color gradient
            const int t = (x * 255) / denom_w;
            const int u = (y * 255) / denom_h;
            int r = t, g = u, b = (t + u) / 2;

            // soft diagonal stripes
            const int period = block * 2 > 0 ? block * 2 : 2;
            if (((x + y) / period) % 2) {
                r = (r + 230) / 2; g = (g + 40) / 2; b = (b + 160) / 2;
            }

            // checkerboard bottom-right quadrant
            if (x >= W / 2 && y >= H / 2) {
                if (((x / block) + (y / block)) % 2) { r = 235; g = 235; b = 235; }
                else                                 { r = 30;  g = 60;  b = 120; }
            }

            // shapes (overwrite, checked back-to-front: ring, disc, dot)
            const long long dx1 = x - ring_cx, dy1 = y - ring_cy;
            const long long d1 = dx1 * dx1 + dy1 * dy1;
            if (d1 >= (ring_r2 * 88) / 100 && d1 <= (ring_r2 * 112) / 100) {
                r = 255; g = 255; b = 255;
            }
            const long long dx2 = x - disc_cx, dy2 = y - disc_cy;
            if (dx2 * dx2 + dy2 * dy2 <= disc_r2) { r = 0; g = 140; b = 140; }
            const long long dx3 = x - dot_cx, dy3 = y - dot_cy;
            if (dx3 * dx3 + dy3 * dy3 <= dot_r2)  { r = 240; g = 200; b = 40; }

            const std::size_t idx = (static_cast<std::size_t>(y) * W + x) * 3;
            img.data[idx + 0] = static_cast<unsigned char>(r < 0 ? 0 : (r > 255 ? 255 : r));
            img.data[idx + 1] = static_cast<unsigned char>(g < 0 ? 0 : (g > 255 ? 255 : g));
            img.data[idx + 2] = static_cast<unsigned char>(b < 0 ? 0 : (b > 255 ? 255 : b));
        }
    }

    if (!save_image(out_path, img, err)) return false;
    return true;
}
