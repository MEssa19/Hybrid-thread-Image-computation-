// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  PartitionMap.cpp - Live console visualizer (--visualize).
//
//  Console-only: the master's split map, the live MPI message timeline and
//  the final recap all go to stdout. No image or data file is written.
// ============================================================================
#include "PartitionMap.h"

#include <cstdio>
#include <cstring>

#include "DistributedEngine.h"

// ---------------------------------------------------------------------------
// PartitionMap
// ---------------------------------------------------------------------------
void PartitionMap::reset(int w, int h) {
    width = w;
    height = h;
    rank_of_row.assign(static_cast<std::size_t>(h), 0);
    thread_of_row.assign(static_cast<std::size_t>(h), 0);
}

int PartitionMap::max_rank() const {
    int m = 0;
    for (std::size_t i = 0; i < rank_of_row.size(); ++i)
        if (rank_of_row[i] > m) m = rank_of_row[i];
    return m;
}

int PartitionMap::max_thread() const {
    int m = 0;
    for (std::size_t i = 0; i < thread_of_row.size(); ++i)
        if (thread_of_row[i] > m) m = thread_of_row[i];
    return m;
}

// ---------------------------------------------------------------------------
// per-row owner map -> contiguous runs (shortest representation)
// ---------------------------------------------------------------------------
int compress_chunks(const unsigned char* map, int n, Chunk* out, int max_out) {
    if (n <= 0 || !map || !out || max_out <= 0) return 0;
    int count = 0;
    int begin = 0;
    for (int i = 1; i <= n; ++i) {
        if (i == n || map[i] != map[begin]) {
            if (count < max_out) {
                out[count].begin = begin;
                out[count].end   = i;              // exclusive
                out[count].id    = map[begin];
                ++count;
            } else {                               // overflow: extend last run
                out[max_out - 1].end = i;
            }
            begin = i;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// ASCII helpers
// ---------------------------------------------------------------------------
static const int BOX_W = 64;

static void box_border(int row) {                       // row < 0 -> no label
    char lbl[32];
    if (row < 0) std::snprintf(lbl, sizeof(lbl), "%10s", "");
    else         std::snprintf(lbl, sizeof(lbl), "row %5d", row);
    std::printf("  %s  +", lbl);
    for (int i = 0; i < BOX_W; ++i) std::printf("-");
    std::printf("+\n");
}

static void box_line(const char* text) {
    const int n = static_cast<int>(std::strlen(text));
    int pad = BOX_W - n - 1;                       // 1 leading space inside the box
    if (pad < 0) pad = 0;
    std::printf("  %10s  | %s%*s|\n", "", text, pad, "");
}

static void print_bar(int width, int rows, int total) {
    int k = total > 0 ? (rows * width + total / 2) / total : 0;
    if (k < 1) k = 1;
    if (k > width) k = width;
    std::printf("[");
    for (int i = 0; i < width; ++i) std::printf("%c", i < k ? '#' : '.');
    std::printf("]");
}

// ---------------------------------------------------------------------------
// master's split map (printed before any message is sent)
// ---------------------------------------------------------------------------
void print_split_map(int H, int size, int radius, const char* filter) {
    std::printf("\n  MASTER SPLITS  %d rows  ->  %d BAND(S)   (%s, halo = %d row%s)\n",
                H, size, filter, radius, (radius == 1 ? "" : "s"));

    for (int r = 0; r < size; ++r) {
        int start = 0, rows = 0;
        compute_band_layout(r, size, H, start, rows);
        if (rows <= 0) {
            char txt[96];
            std::snprintf(txt, sizeof(txt),
                          "BAND %-2d -> RANK %-2d  (0 rows - rank stays idle)", r + 1, r);
            box_line(txt);
            continue;
        }
        const int y0 = start - radius < 0 ? 0 : start - radius;
        const int y1 = start + rows + radius > H ? H : start + rows + radius;
        const int halo_top = start - y0;
        const int halo_bot = y1 - (start + rows);

        box_border(start);
        char title[96];
        std::snprintf(title, sizeof(title), "BAND %-2d -> RANK %-2d%s | rows %d..%d (%d %s)",
                      r + 1, r, (r == 0 ? " (master keeps it)" : ""),
                      start, start + rows - 1, rows, rows_word(rows));
        box_line(title);
        if (radius == 0) {
            box_line("halo: none needed (radius-0 filter)");
        } else {
            char top[64], bot[64], line[192];
            if (halo_top == 0) std::snprintf(top, sizeof(top), "top none (image edge)");
            else std::snprintf(top, sizeof(top), "top %d (rows %d-%d)", halo_top, y0, start - 1);
            if (halo_bot == 0) std::snprintf(bot, sizeof(bot), "bottom none (image edge)");
            else std::snprintf(bot, sizeof(bot), "bottom %d (rows %d-%d)", halo_bot, start + rows, y1 - 1);
            std::snprintf(line, sizeof(line), "halo: %s | %s", top, bot);
            box_line(line);
        }
    }
    box_border(H - 1);
    if (size > 1)
        std::printf("  RANK 0 processes band 1 locally; bands 2..%d travel to the workers.\n",
                    size);
    else
        std::printf("  Only RANK 0 exists - it processes the single band locally.\n");
}

// ---------------------------------------------------------------------------
// final recap: who processed which rows (console table with bars)
// ---------------------------------------------------------------------------
void print_partition_summary(const PartitionMap& pm, int ranks, int threads) {
    std::printf("\n  WORK RECAP - who processed which rows (ranks=%d, threads/rank=%d)\n",
                ranks, threads);

    std::vector<char> seen(static_cast<std::size_t>(ranks > 0 ? ranks : 1), 0);

    int y = 0;
    while (y < pm.height) {
        const int rk = pm.rank_of_row[static_cast<std::size_t>(y)];
        int end = y;
        while (end < pm.height && pm.rank_of_row[end] == rk) ++end;
        const int rows = end - y;
        if (rk < ranks) seen[static_cast<std::size_t>(rk)] = 1;

        std::printf("  RANK %d : rows %5d - %5d (%d %s)\n", rk, y, end - 1, rows, rows_word(rows));

        Chunk chunks[64];
        const int n = compress_chunks(&pm.thread_of_row[static_cast<std::size_t>(y)],
                                      rows, chunks, 64);
        for (int c = 0; c < n; ++c) {
            const int crows = chunks[c].end - chunks[c].begin;
            std::printf("          T%-2d ", chunks[c].id);
            print_bar(24, crows, rows);
            std::printf(" rows %5d - %5d (%d %s)\n",
                        y + chunks[c].begin, y + chunks[c].end - 1, crows, rows_word(crows));
        }
        if (n >= 64) std::printf("          ...\n");
        y = end;
    }

    for (int r = 0; r < ranks; ++r)
        if (!seen[static_cast<std::size_t>(r)])
            std::printf("  RANK %d : idle (0 rows assigned)\n", r);

    std::printf("  Legend: '#' bar length = share of the band's rows, Tn = OpenMP thread n.\n");
}
