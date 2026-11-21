// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  Benchmark.h - Benchmarking Module.
//
//  Proposal mapping: "Tracks and prints the execution time for different
//  modes (Sequential, OpenMP only, MPI only, and Hybrid)." and
//  "A detailed performance comparison table."
//
//  * append_csv()  - appends one row per run to times.csv
//  * print_report()- reads the CSV and prints a comparison table with
//                    mean times and speedup versus the sequential mode
//  * Timer / mpi_now() - std::chrono for local modes, MPI_Wtime for
//                    distributed modes (both millisecond precision)
// ============================================================================
#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include "Image.h"

struct RunRecord {
    std::string mode;                  // sequential | openmp | mpi | hybrid
    std::string filter;                // gaussian | sobel | ...
    int width = 0;
    int height = 0;
    int ranks = 1;
    int threads = 1;
    double load_ms = 0.0;
    double process_ms = 0.0;
    double save_ms = 0.0;
    double total_ms = 0.0;
    unsigned long long checksum = 0;
};

// Millisecond precision timers.
double chrono_now_ms();                // std::chrono::steady_clock (always usable)
double mpi_wtime_ms();                 // MPI_Wtime (only call when MPI is initialized)

// Append one CSV row (header written automatically for a new file). Rank 0 only.
bool append_csv(const std::string& path, const RunRecord& r, std::string& err);

// Print a human-readable summary of a single run to stdout.
void print_run(const RunRecord& r);

// Read the CSV and print the comparison table (mean + speedup per mode).
bool print_report(const std::string& path, std::string& err);

#endif // BENCHMARK_H
