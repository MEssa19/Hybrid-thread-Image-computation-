// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  Benchmark.cpp - Benchmarking Module implementation.
// ============================================================================
#define _CRT_SECURE_NO_WARNINGS

#include "Benchmark.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <omp.h>

double chrono_now_ms() {
    using clk = std::chrono::steady_clock;
    const clk::time_point t = clk::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

double mpi_wtime_ms() {
    return MPI_Wtime() * 1000.0;
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------
static const char* CSV_HEADER =
    "mode,filter,width,height,ranks,threads,load_ms,process_ms,save_ms,total_ms,checksum";

bool append_csv(const std::string& path, const RunRecord& r, std::string& err) {
    bool need_header = true;
    {   // check whether the file already has content
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f) {
            need_header = false;
            std::fclose(f);
        }
    }
    FILE* f = std::fopen(path.c_str(), "ab");                    // append (binary: Windows)
    if (!f) { err = "cannot open CSV for appending: " + path; return false; }
    if (need_header) std::fprintf(f, "%s\n", CSV_HEADER);
    std::fprintf(f, "%s,%s,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%llu\n",
                 r.mode.c_str(), r.filter.c_str(), r.width, r.height,
                 r.ranks, r.threads, r.load_ms, r.process_ms, r.save_ms, r.total_ms,
                 r.checksum);
    std::fclose(f);
    return true;
}

void print_run(const RunRecord& r) {
    std::printf("  ------------------------------------------------------------\n");
    std::printf("  Mode     : %-12s Filter   : %s\n", r.mode.c_str(), r.filter.c_str());
    std::printf("  Image    : %dx%d  Ranks    : %d  x %d thread(s)/rank\n",
                r.width, r.height, r.ranks, r.threads);
    std::printf("  Load     : %9.2f ms      Save     : %9.2f ms\n", r.load_ms, r.save_ms);
    std::printf("  Process  : %9.2f ms      Total    : %9.2f ms\n", r.process_ms, r.total_ms);
    std::printf("  Checksum : %llu  (must match across all modes)\n", r.checksum);
    std::printf("  ------------------------------------------------------------\n");
}

// ---------------------------------------------------------------------------
// report: parse CSV, group by (filter, width, height), aggregate by mode
// ---------------------------------------------------------------------------
struct AggRow {
    std::string mode;
    int ranks = 1;
    int threads = 1;
    int runs = 0;
    double sum_process = 0.0;
    double sum_total = 0.0;
};

bool print_report(const std::string& path, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open CSV: " + path; return false; }

    char line[1024];
    bool first = true;
    bool any = false;

    struct Group {
        std::string filter;
        int width = 0, height = 0;
        std::vector<AggRow> rows;
        double seq_mean = -1.0;
    };
    std::vector<Group> groups;

    while (std::fgets(line, sizeof(line), f)) {
        if (first) { first = false; if (std::strncmp(line, "mode,", 5) == 0) continue; }
        char mode[32] = {0}, filter[32] = {0};
        int width = 0, height = 0, ranks = 0, threads = 0;
        double load = 0, process = 0, save = 0, total = 0;
        unsigned long long checksum = 0;
        const int fields = std::sscanf(line, "%31[^,],%31[^,],%d,%d,%d,%d,%lf,%lf,%lf,%lf,%llu",
                                       mode, filter, &width, &height, &ranks, &threads,
                                       &load, &process, &save, &total, &checksum);
        if (fields != 11) continue;                              // skip malformed lines
        any = true;

        Group* g = nullptr;
        for (std::size_t i = 0; i < groups.size(); ++i) {
            if (groups[i].filter == filter && groups[i].width == width && groups[i].height == height) {
                g = &groups[i];
                break;
            }
        }
        if (!g) {
            Group ng;
            ng.filter = filter; ng.width = width; ng.height = height;
            groups.push_back(ng);
            g = &groups.back();
        }
        AggRow* row = nullptr;
        for (std::size_t i = 0; i < g->rows.size(); ++i) {
            if (g->rows[i].mode == mode && g->rows[i].ranks == ranks && g->rows[i].threads == threads) {
                row = &g->rows[i];
                break;
            }
        }
        if (!row) {
            AggRow nr;
            nr.mode = mode; nr.ranks = ranks; nr.threads = threads;
            g->rows.push_back(nr);
            row = &g->rows.back();
        }
        row->runs++;
        row->sum_process += process;
        row->sum_total += total;
    }
    std::fclose(f);

    if (!any) { err = "no usable rows in " + path; return false; }

    // mean per group row + remember sequential mean for speedup
    for (std::size_t i = 0; i < groups.size(); ++i) {
        for (std::size_t j = 0; j < groups[i].rows.size(); ++j) {
            AggRow& r = groups[i].rows[j];
            r.sum_process /= r.runs;
            r.sum_total /= r.runs;
            if (r.mode == "sequential") groups[i].seq_mean = r.sum_process;
        }
    }

    std::printf("\n=== Performance comparison report (%s) ===\n", path.c_str());
    for (std::size_t i = 0; i < groups.size(); ++i) {
        const Group& g = groups[i];
        std::printf("\nImage %dx%d   Filter: %s\n", g.width, g.height, g.filter.c_str());
        std::printf("  %-11s %-7s %-7s %-13s %-10s %-10s\n",
                    "Mode", "Ranks", "Threads", "Process(ms)", "Total(ms)", "Speedup");
        std::printf("  %-11s %-7s %-7s %-13s %-10s %-10s\n",
                    "----", "-----", "-------", "-----------", "---------", "-------");
        for (std::size_t j = 0; j < g.rows.size(); ++j) {
            const AggRow& r = g.rows[j];
            char speed[16] = "-";
            if (g.seq_mean > 0.0 && r.sum_process > 0.0) {
                std::snprintf(speed, sizeof(speed), "%.2fx", g.seq_mean / r.sum_process);
            }
            std::printf("  %-11s %-7d %-7d %-13.2f %-10.2f %-10s\n",
                        r.mode.c_str(), r.ranks, r.threads, r.sum_process, r.sum_total, speed);
        }
    }
    std::printf("\nSpeedup = mean sequential process time / mean parallel process time\n");
    return true;
}
