// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  main.cpp - CLI front end and mode dispatch.
//
//  Usage:
//    himg <input.(ppm|bmp)> <filter> [options]
//    himg --make-test <out.(ppm|bmp)> [width] [height]   (default 1920x1080)
//
//    Modes:       --mode sequential | openmp | mpi | hybrid   (default: hybrid)
//    Threads:     --threads N          OpenMP threads per rank (hybrid/openmp)
//    Output:      --out PATH           default: <input>_<filter>_<mode>.<ext>
//    Visualize:   --visualize          LIVE console view of the master/worker
//                                      split + thread sharing (no extra files)
//    CSV record:  --csv PATH           append one benchmark row (default times.csv is NOT written unless flag given)
//    Report:      himg --report times.csv
//    Filter list: himg --list-filters
//
//  Examples:
//    himg --make-test test.ppm 1920 1080
//    himg test.ppm gaussian --mode sequential --csv times.csv
//    himg input.bmp sobel --mode openmp --threads 4 --csv times.csv
//    mpiexec -n 2 himg input.ppm gaussian --mode mpi --csv times.csv
//    mpiexec -n 3 himg test.ppm gaussian --mode hybrid --threads 2 --visualize
// ============================================================================
#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mpi.h>
#include <omp.h>

#include "Benchmark.h"
#include "DistributedEngine.h"
#include "Filters.h"
#include "Image.h"
#include "ImageIO.h"
#include "PartitionMap.h"

static void print_usage(const char* exe) {
    std::printf(
        "Hybrid Distributed Image Processing Engine\n"
        "C++ / MPI / OpenMP master-worker image filtering (PDC course project)\n\n"
        "Usage:\n"
        "  %s <input.(ppm|bmp)> <filter> [options]\n"
        "  %s --make-test <out.(ppm|bmp)> [width] [height]   (default 1920x1080)\n"
        "  %s --report <times.csv>\n"
        "  %s --list-filters\n\n"
        "Options:\n"
        "  --mode M      sequential | openmp | mpi | hybrid   (default: hybrid)\n"
        "  --threads N   OpenMP threads per rank              (default: OMP_NUM_THREADS or hardware)\n"
        "  --out PATH    output image path                    (default: <input>_<filter>_<mode>.<ext>)\n"
        "  --visualize   print a LIVE master/worker view on this console:\n"
        "                band split, halo rows, message timeline, per-thread row\n"
        "                shares (no extra files are written)\n"
        "  --csv PATH    append benchmark row to PATH\n\n"
        "Notes:\n"
        "  * No image handy?  Create one:  %s --make-test test.ppm 1920 1080\n"
        "  * sequential and openmp modes run the exe directly (no mpiexec needed).\n"
        "  * mpi and hybrid modes must be launched with mpiexec, e.g.:\n"
        "      mpiexec -n 2 %s input.ppm gaussian --mode hybrid --threads 4\n"
        "  * With -n 1 the distributed modes degenerate to the OpenMP-only mode.\n",
        exe, exe, exe, exe, exe, exe);
}

// ---------------------------------------------------------------------------
// sequential / openmp-only processing (no MPI distribution involved)
// ---------------------------------------------------------------------------
static int run_local(const std::string& in_path, const std::string& out_path,
                     FilterId filter, Mode mode, int threads_arg,
                     const std::string& csv_path, bool visualize) {
    // Safety net: if someone launches a local-only mode under mpiexec with
    // several ranks, only rank 0 proceeds and the rest exit cleanly.
    const char* ompi_rank = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* pmi_rank  = std::getenv("PMI_RANK");
    if (ompi_rank || pmi_rank) {
        const int r = std::atoi(ompi_rank ? ompi_rank : pmi_rank);
        if (r > 0) return 0;
        std::printf("NOTE: running under mpiexec but mode '%s' does not need it -\n"
                    "      all ranks except 0 idle. Use '-n 1' or --mode mpi|hybrid.\n",
                    mode_to_string(mode));
    }

    if (mode == MODE_OPENMP) {
        if (threads_arg > 0) omp_set_num_threads(threads_arg);
    } else {
        omp_set_num_threads(1);                  // strictly single-core for sequential
    }
    const int threads = omp_get_max_threads();

    std::string err;
    Image img;
    const double t_load0 = chrono_now_ms();
    if (!load_image(in_path, img, err)) {
        std::printf("ERROR: %s\n", err.c_str());
        return 1;
    }
    const double load_ms = chrono_now_ms() - t_load0;

    std::printf("Local Engine - mode=%s filter=%s threads=%d\n",
                mode_to_string(mode), filter_name(filter), threads);
    if (visualize) {
        if (mode == MODE_OPENMP)
            std::printf("[LIVE] %d OpenMP thread(s) will share all %d rows of the image\n",
                        threads, img.height);
        else
            std::printf("[LIVE] one thread processes rows 0..%d strictly in order\n",
                        img.height - 1);
    }

    Image out;
    std::vector<unsigned char> thread_rows;          // --visualize: row -> thread id
    if (visualize) thread_rows.assign(static_cast<std::size_t>(img.height), 0);
    const double t0 = chrono_now_ms();
    apply_filter(img, out, filter, /*omp*/ mode == MODE_OPENMP,
                 thread_rows.empty() ? nullptr : &thread_rows[0]);
    const double process_ms = chrono_now_ms() - t0;

    const double t_save0 = chrono_now_ms();
    if (!save_image(out_path, out, err)) {
        std::printf("ERROR: %s\n", err.c_str());
        return 1;
    }
    const double save_ms = chrono_now_ms() - t_save0;

    RunRecord rec;
    rec.mode = mode_to_string(mode);
    rec.filter = filter_name(filter);
    rec.width = img.width;
    rec.height = img.height;
    rec.ranks = 1;
    rec.threads = threads;
    rec.load_ms = load_ms;
    rec.process_ms = process_ms;
    rec.save_ms = save_ms;
    rec.total_ms = load_ms + process_ms + save_ms;
    rec.checksum = image_checksum(out, mode == MODE_OPENMP);
    print_run(rec);
    if (!csv_path.empty()) {
        if (append_csv(csv_path, rec, err)) {
            std::printf("Benchmark row appended to %s\n", csv_path.c_str());
        } else {
            std::printf("WARNING: %s\n", err.c_str());
        }
    }
    if (visualize) {
        // Console-only recap: which thread produced which rows. No file output.
        PartitionMap pm;
        pm.reset(img.width, img.height);             // rank_of_row stays all 0 (no MPI)
        if (!thread_rows.empty())
            std::memcpy(&pm.thread_of_row[0], &thread_rows[0],
                        static_cast<std::size_t>(img.height));
        print_partition_summary(pm, 1, threads);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// argument parsing
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string input, filter_name_arg, out_path, csv_path, report_path;
    Mode mode = MODE_HYBRID;
    int threads_arg = 0;
    bool have_filter = false;
    bool visualize = false;

    if (argc < 2) { print_usage(argv[0]); return 0; }

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
        else if (a == "--list-filters") {
            std::printf("Available filters:\n");
            print_filter_menu();
            return 0;
        }
        else if (a == "--make-test") {
            if (i + 1 >= argc) { std::printf("ERROR: --make-test needs an output path (.ppm or .bmp)\n"); return 1; }
            const std::string test_path = argv[++i];
            int w = 1920, h = 1080;
            if (i + 1 < argc && argv[i + 1][0] != '-') w = std::atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') h = std::atoi(argv[++i]);
            std::string err;
            if (!make_test_image(test_path, w, h, err)) {
                std::printf("ERROR: %s\n", err.c_str());
                return 1;
            }
            std::printf("Test image %dx%d written to %s\n\n", w, h, test_path.c_str());
            std::printf("Now try:\n");
            std::printf("  %s %s gaussian --mode openmp --threads 4 --visualize\n", argv[0], test_path.c_str());
            std::printf("  mpiexec -n 3 %s %s gaussian --mode hybrid --threads 2 --visualize\n", argv[0], test_path.c_str());
            return 0;
        }
        else if (a == "--report") {
            if (i + 1 >= argc) { std::printf("ERROR: --report needs a file path\n"); return 1; }
            report_path = argv[++i];
            std::string err;
            if (!print_report(report_path, err)) { std::printf("ERROR: %s\n", err.c_str()); return 1; }
            return 0;
        }
        else if (a == "--mode") {
            if (i + 1 >= argc) { std::printf("ERROR: --mode needs a value\n"); return 1; }
            mode = mode_from_string(argv[++i]);
            if (mode == MODE_COUNT) { std::printf("ERROR: unknown mode (use sequential|openmp|mpi|hybrid)\n"); return 1; }
        }
        else if (a == "--threads") {
            if (i + 1 >= argc) { std::printf("ERROR: --threads needs a number\n"); return 1; }
            threads_arg = std::atoi(argv[++i]);
            if (threads_arg < 1) { std::printf("ERROR: --threads must be >= 1\n"); return 1; }
        }
        else if (a == "--out") {
            if (i + 1 >= argc) { std::printf("ERROR: --out needs a path\n"); return 1; }
            out_path = argv[++i];
        }
        else if (a == "--csv") {
            if (i + 1 >= argc) { std::printf("ERROR: --csv needs a path\n"); return 1; }
            csv_path = argv[++i];
        }
        else if (a == "--visualize") {
            visualize = true;
        }
        else if (a.length() > 0 && a[0] == '-') {
            std::printf("ERROR: unknown option %s (see --help)\n", a.c_str());
            return 1;
        }
        else if (input.empty()) input = a;
        else if (!have_filter)  { filter_name_arg = a; have_filter = true; }
        else { std::printf("ERROR: unexpected extra argument '%s'\n", a.c_str()); return 1; }
    }

    if (input.empty() || !have_filter) {
        std::printf("ERROR: an input image and a filter are required.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    const FilterId filter = filter_from_string(filter_name_arg);
    if (filter == FILTER_COUNT) {
        std::printf("ERROR: unknown filter '%s'. Available filters:\n", filter_name_arg.c_str());
        print_filter_menu();
        return 1;
    }

    if (out_path.empty()) out_path = make_output_path(input, filter_name(filter), mode_to_string(mode));

    if (!mode_is_distributed(mode)) {
        return run_local(input, out_path, filter, mode, threads_arg, csv_path, visualize);
    }

    // ---- MPI modes --------------------------------------------------------
    MPI_Init(&argc, &argv);
    const int rc = run_distributed(input, out_path, filter, mode, threads_arg,
                                   csv_path, visualize);
    MPI_Finalize();
    return rc;
}
