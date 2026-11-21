// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  DistributedEngine.cpp - Distributed Engine (MPI) implementation.
//
//  Workflow (matches proposal section 4):
//    1. Rank 0 (master) loads the image and slices it into horizontal bands.
//    2. Each band is sent to a worker process with MPI_Send, extended by
//       'radius' halo rows above and below so borders filter correctly.
//    3. Each worker applies the filter to its band with an OpenMP parallel
//       loop over the band's rows (shared memory inside the process).
//    4. Workers send the processed band back with MPI_Send.
//    5. Rank 0 gathers all bands with MPI_Recv, stitches them into the final
//       image, saves it, and prints the measured execution times.
//
//  Message protocol (point-to-point, blocking):
//    tag 0 : work header  int[6] = {start_row, band_rows, y0, y1, W, C}
//    tag 1 : window pixels (unsigned char)
//    tag 2 : unused (reserved)
//    tag 3 : result header int[2] = {start_row, band_rows}
//    tag 4 : result pixels (unsigned char)
//    tag 5 : result thread map, band_rows bytes (row -> OMP thread id),
//            sent only when --visualize is active (PartitionMap module)
//    tag 6 : live log line (text), sent only when --visualize is active.
//            Workers report "band received / threads started / rows shared /
//            result on its way" to the master, which prints everything as it
//            arrives - the live command-interface view uses NO extra files.
//
//  Workers with band_rows == 0 (e.g. more ranks than image rows) reply with
//  an empty result so no rank is ever left waiting.
// ============================================================================
#include "DistributedEngine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <chrono>
#include <thread>
#include <vector>
#include <mpi.h>
#include <omp.h>

#include "Benchmark.h"
#include "Filters.h"
#include "ImageIO.h"
#include "PartitionMap.h"

Mode mode_from_string(const std::string& s) {
    if (s == "sequential") return MODE_SEQUENTIAL;
    if (s == "openmp")     return MODE_OPENMP;
    if (s == "mpi")        return MODE_MPI;
    if (s == "hybrid")     return MODE_HYBRID;
    return MODE_COUNT;
}

const char* mode_to_string(Mode m) {
    switch (m) {
        case MODE_SEQUENTIAL: return "sequential";
        case MODE_OPENMP:     return "openmp";
        case MODE_MPI:        return "mpi";
        case MODE_HYBRID:     return "hybrid";
        default:              return "unknown";
    }
}

bool mode_is_distributed(Mode m) {
    return m == MODE_MPI || m == MODE_HYBRID;
}

// ---------------------------------------------------------------------------
// band layout shared by master, worker and the console split map
// ---------------------------------------------------------------------------
void compute_band_layout(int rank, int size, int H, int& start, int& rows) {
    const int base = H / size;
    const int rem  = H % size;
    rows  = base + (rank < rem ? 1 : 0);
    start = rank * base + (rank < rem ? rank : rem);
}

static const int TAG_WORK_HEADER = 0;
static const int TAG_WORK_DATA   = 1;
static const int TAG_RESULT_HEADER = 3;
static const int TAG_RESULT_DATA   = 4;
static const int TAG_RESULT_THREADS = 5;   // --visualize only
static const int TAG_LOG            = 6;   // --visualize only: live log line

// ---------------------------------------------------------------------------
// live-view helpers (master side)
//   ts_print  : one timestamped console line, flushed immediately so the
//               command interface shows events the moment they happen.
//   drain_logs: collect whatever worker log lines (tag 6) have arrived and
//               print them - this is what makes the view "live".
// ---------------------------------------------------------------------------
static void ts_print(double t0, const char* fmt, ...) {
    const double ms = (MPI_Wtime() - t0) * 1000.0;
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    std::printf("[t+%9.1f ms] %s\n", ms, msg);
    std::fflush(stdout);
}

static void drain_logs(double t0) {
    int flag = 0;
    MPI_Status st;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_LOG, MPI_COMM_WORLD, &flag, &st);
    while (flag) {
        int len = 0;
        MPI_Get_count(&st, MPI_CHAR, &len);
        char buf[512];
        MPI_Recv(buf, len, MPI_CHAR, st.MPI_SOURCE, TAG_LOG, MPI_COMM_WORLD, &st);
        ts_print(t0, "RANK %d | %s", st.MPI_SOURCE, buf);
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_LOG, MPI_COMM_WORLD, &flag, &st);
    }
}

// ---------------------------------------------------------------------------
// worker-side live log (only called when --visualize is active)
// ---------------------------------------------------------------------------
static void send_log(const char* fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    MPI_Send(msg, static_cast<int>(std::strlen(msg)) + 1, MPI_CHAR, 0,
             TAG_LOG, MPI_COMM_WORLD);
}

// ---------------------------------------------------------------------------
// master (rank 0)
// ---------------------------------------------------------------------------
static int master_loop(int size, const Image& img, FilterId filter, int threads,
                       Image& out, double& process_ms, PartitionMap* pm) {
    const int W = img.width;
    const int H = img.height;
    const int C = img.channels;
    const int R = filter_radius(filter);

    out = img;                                   // allocate + keep rank 0's own band later
    out.data.assign(img.byte_count(), 0);

    const double t0 = MPI_Wtime();

    // ---- record the band layout + draw the split map (visualization) -----
    if (pm) {
        pm->reset(W, H);
        for (int r = 0; r < size; ++r) {
            int start = 0, rows = 0;
            compute_band_layout(r, size, H, start, rows);
            for (int y = start; y < start + rows; ++y)
                pm->rank_of_row[y] = static_cast<unsigned char>(r);
        }
        print_split_map(H, size, R, filter_name(filter));
        std::fflush(stdout);
    }

    // ---- send work to every worker (including empty bands) ---------------
    for (int r = 1; r < size; ++r) {
        int start = 0, rows = 0;
        compute_band_layout(r, size, H, start, rows);
        const int y0 = start - R < 0 ? 0 : start - R;
        const int y1 = start + rows + R > H ? H : start + rows + R;
        int hdr[6] = { start, rows, y0, y1, W, C };
        MPI_Send(hdr, 6, MPI_INT, r, TAG_WORK_HEADER, MPI_COMM_WORLD);
        if (rows > 0) {
            const long count = static_cast<long>(y1 - y0) * W * C;
            MPI_Send(&img.data[static_cast<std::size_t>(y0) * W * C], (int)count,
                     MPI_UNSIGNED_CHAR, r, TAG_WORK_DATA, MPI_COMM_WORLD);
            if (pm)
                ts_print(t0, "MASTER -> RANK %d | band %d sent: rows %d..%d "
                             "(%d %s, halo +%d/+%d) = %.1f KB",
                         r, r + 1, start, start + rows - 1, rows, rows_word(rows),
                         start - y0, y1 - (start + rows),
                         static_cast<double>(count) / 1024.0);
        } else if (pm) {
            ts_print(t0, "MASTER -> RANK %d | band %d is empty (0 rows) - idle notice sent",
                     r, r + 1);
        }
    }
    if (pm) drain_logs(t0);

    // ---- process rank 0's own band locally (OpenMP threads) --------------
    {
        int start = 0, rows = 0;
        compute_band_layout(0, size, H, start, rows);
        if (rows > 0) {
            const int y0 = start - R < 0 ? 0 : start - R;
            const int y1 = start + rows + R > H ? H : start + rows + R;
            std::vector<unsigned char> band(static_cast<std::size_t>(rows) * W * C, 0);
            std::vector<unsigned char> local_threads(
                static_cast<std::size_t>(rows), 0);
            if (pm) {
                ts_print(t0, "MASTER (RANK 0) | band 1: %d OpenMP thread(s) start on "
                             "rows %d..%d (%d %s)", threads, start, start + rows - 1,
                         rows, rows_word(rows));
            }
            const double t_own = MPI_Wtime();
            process_window(&img.data[static_cast<std::size_t>(y0) * W * C],
                           W, y1 - y0, C, &band[0], start - y0, start - y0 + rows,
                           filter, /*omp*/ true,
                           pm ? &local_threads[0] : nullptr);
            if (pm) {
                ts_print(t0, "MASTER (RANK 0) | band 1 done in %.1f ms",
                         (MPI_Wtime() - t_own) * 1000.0);
                Chunk chunks[64];
                const int n = compress_chunks(&local_threads[0], rows, chunks, 64);
                char line[512];
                std::snprintf(line, sizeof(line), "band 1 shared %d %s:", rows, rows_word(rows));
                std::size_t pos = std::strlen(line);
                for (int c = 0; c < n && pos < sizeof(line) - 24; ++c) {
                    const int wr = std::snprintf(line + pos, sizeof(line) - pos,
                                                 " T%d: %d-%d", chunks[c].id,
                                                 start + chunks[c].begin,
                                                 start + chunks[c].end - 1);
                    if (wr > 0) pos += static_cast<std::size_t>(wr);
                }
                ts_print(t0, "MASTER (RANK 0) | %s", line);
                std::memcpy(&pm->thread_of_row[start], &local_threads[0],
                            static_cast<std::size_t>(rows));
            }
            std::memcpy(&out.data[static_cast<std::size_t>(start) * W * C], &band[0],
                        static_cast<std::size_t>(rows) * W * C);
        }
    }
    if (pm) drain_logs(t0);

    // ---- gather results ---------------------------------------------------
    if (pm) {
        // LIVE gather: poll for any message, print worker logs the moment
        // they arrive and stitch every band as soon as it is back.
        int pending = size - 1;
        while (pending > 0) {
            drain_logs(t0);
            int flag = 0;
            MPI_Status st;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
            if (!flag) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            if (st.MPI_TAG == TAG_RESULT_HEADER) {
                int rhdr[2] = { 0, 0 };
                MPI_Recv(rhdr, 2, MPI_INT, st.MPI_SOURCE, TAG_RESULT_HEADER,
                         MPI_COMM_WORLD, &st);
                if (rhdr[1] > 0) {
                    const long count = static_cast<long>(rhdr[1]) * W * C;
                    MPI_Recv(&out.data[static_cast<std::size_t>(rhdr[0]) * W * C],
                             (int)count, MPI_UNSIGNED_CHAR, st.MPI_SOURCE,
                             TAG_RESULT_DATA, MPI_COMM_WORLD, &st);
                    MPI_Recv(&pm->thread_of_row[rhdr[0]], rhdr[1], MPI_UNSIGNED_CHAR,
                             st.MPI_SOURCE, TAG_RESULT_THREADS, MPI_COMM_WORLD, &st);
                    ts_print(t0, "RANK %d -> MASTER | band %d stitched into rows "
                                 "%d..%d (%d %s)", st.MPI_SOURCE, st.MPI_SOURCE + 1,
                             rhdr[0], rhdr[0] + rhdr[1] - 1, rhdr[1], rows_word(rhdr[1]));
                } else {
                    ts_print(t0, "RANK %d -> MASTER | band %d empty - rank was idle",
                             st.MPI_SOURCE, st.MPI_SOURCE + 1);
                }
                --pending;
            } else if (st.MPI_TAG == TAG_LOG) {
                drain_logs(t0);                   // probed a log line: print it
            } else {
                // defensive: consume an unexpected message, never happen in practice
                int len = 0;
                MPI_Get_count(&st, MPI_CHAR, &len);
                std::vector<char> tmp(static_cast<std::size_t>(len > 0 ? len : 1));
                MPI_Recv(&tmp[0], len, MPI_CHAR, st.MPI_SOURCE, st.MPI_TAG,
                         MPI_COMM_WORLD, &st);
            }
        }
        drain_logs(t0);
        print_partition_summary(*pm, size, threads);
        std::fflush(stdout);
    } else {
        // benchmark path (no visualization): plain blocking receives
        for (int r = 1; r < size; ++r) {
            int rhdr[2] = { 0, 0 };
            MPI_Status st;
            MPI_Recv(rhdr, 2, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT_HEADER, MPI_COMM_WORLD, &st);
            if (rhdr[1] > 0) {
                const long count = static_cast<long>(rhdr[1]) * W * C;
                MPI_Recv(&out.data[static_cast<std::size_t>(rhdr[0]) * W * C], (int)count,
                         MPI_UNSIGNED_CHAR, st.MPI_SOURCE, TAG_RESULT_DATA, MPI_COMM_WORLD, &st);
            }
        }
    }

    process_ms = (MPI_Wtime() - t0) * 1000.0;
    return 0;
}

// ---------------------------------------------------------------------------
// worker (rank > 0)
// ---------------------------------------------------------------------------
static void worker_loop(FilterId filter, bool omp, bool want_log) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int hdr[6] = { 0, 0, 0, 0, 0, 0 };
    MPI_Status st;
    MPI_Recv(hdr, 6, MPI_INT, 0, TAG_WORK_HEADER, MPI_COMM_WORLD, &st);

    const int start = hdr[0], rows = hdr[1], y0 = hdr[2], y1 = hdr[3];
    const int W = hdr[4], C = hdr[5];

    if (rows <= 0) {
        if (want_log)
            send_log("band %d is empty (0 rows) - staying idle, answering now", rank + 1);
        int empty[2] = { start, 0 };
        MPI_Send(empty, 2, MPI_INT, 0, TAG_RESULT_HEADER, MPI_COMM_WORLD);
        return;
    }

    std::vector<unsigned char> win(static_cast<std::size_t>(y1 - y0) * W * C);
    MPI_Recv(&win[0], (int)win.size(), MPI_UNSIGNED_CHAR, 0, TAG_WORK_DATA,
             MPI_COMM_WORLD, &st);
    const double t_band = MPI_Wtime();

    if (want_log)
        send_log("band %d received: rows %d..%d (%d %s, halo +%d/+%d), "
                 "window %.1f KB", rank + 1, start, start + rows - 1, rows,
                 rows_word(rows), start - y0, y1 - (start + rows),
                 static_cast<double>(win.size()) / 1024.0);

    std::vector<unsigned char> band(static_cast<std::size_t>(rows) * W * C);
    std::vector<unsigned char> my_threads(static_cast<std::size_t>(rows), 0);

    if (want_log)
        send_log("%d OpenMP thread(s) start on the band's %d %s (shared memory)",
                 omp_get_max_threads(), rows, rows_word(rows));

    process_window(&win[0], W, y1 - y0, C, &band[0],
                   start - y0, start - y0 + rows, filter, omp,
                   want_log ? &my_threads[0] : nullptr);
    const double compute_ms = (MPI_Wtime() - t_band) * 1000.0;

    if (want_log) {
        // report how the work-sharing loop actually divided the rows
        Chunk chunks[16];
        const int n = compress_chunks(&my_threads[0], rows, chunks, 16);
        char line[512];
        std::snprintf(line, sizeof(line),
                      "band done in %.1f ms | shared %d %s:", compute_ms, rows, rows_word(rows));
        std::size_t pos = std::strlen(line);
        for (int c = 0; c < n && pos < sizeof(line) - 24; ++c) {
            const int wr = std::snprintf(line + pos, sizeof(line) - pos,
                                         " T%d: %d-%d", chunks[c].id,
                                         start + chunks[c].begin,
                                         start + chunks[c].end - 1);
            if (wr > 0) pos += static_cast<std::size_t>(wr);
        }
        if (n >= 16) {
            const int wr = std::snprintf(line + pos, sizeof(line) - pos, " ...");
            if (wr > 0) pos += static_cast<std::size_t>(wr);
        }
        send_log("%s", line);
        send_log("sending %d result %s back to the master", rows, rows_word(rows));
    }

    int rhdr[2] = { start, rows };
    MPI_Send(rhdr, 2, MPI_INT, 0, TAG_RESULT_HEADER, MPI_COMM_WORLD);
    MPI_Send(&band[0], (int)band.size(), MPI_UNSIGNED_CHAR, 0, TAG_RESULT_DATA,
             MPI_COMM_WORLD);
    if (want_log)
        MPI_Send(&my_threads[0], rows, MPI_UNSIGNED_CHAR, 0, TAG_RESULT_THREADS,
                 MPI_COMM_WORLD);
}

// ---------------------------------------------------------------------------
// distributed entry point (MPI already initialized)
// ---------------------------------------------------------------------------
int run_distributed(const std::string& in_path, const std::string& out_path,
                    FilterId filter, Mode mode, int threads_arg,
                    const std::string& csv_path, bool visualize) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Synchronization: thread count per rank depends on the benchmark mode.
    //   mpi    -> exactly 1 thread (pure distributed memory)
    //   hybrid -> threads_arg, OMP_NUM_THREADS, or hardware default
    if (mode == MODE_MPI) {
        omp_set_num_threads(1);
    } else if (threads_arg > 0) {
        omp_set_num_threads(threads_arg);
    }
    const int threads = omp_get_max_threads();

    int rc = 0;
    if (rank == 0) {
        if (visualize) {
            std::printf("\n==============================================================\n");
            std::printf("  LIVE MASTER/WORKER VIEW - who owns which rows, in real time\n");
            std::printf("  mode=%s  filter=%s  ranks=%d  threads/rank=%d\n",
                        mode_to_string(mode), filter_name(filter), size, threads);
            std::printf("==============================================================\n");
            std::fflush(stdout);
        }
        std::printf("[Rank 0] Distributed Engine - mode=%s filter=%s ranks=%d threads/rank=%d\n",
                    mode_to_string(mode), filter_name(filter), size, threads);

        Image img;
        std::string err;
        const double t_load0 = MPI_Wtime();
        if (!load_image(in_path, img, err)) {
            std::printf("[Rank 0] ERROR: %s\n", err.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        const double load_ms = (MPI_Wtime() - t_load0) * 1000.0;
        std::printf("[Rank 0] Loaded %dx%d image (%.1f KB), slicing into %d band(s)\n",
                    img.width, img.height,
                    static_cast<double>(img.data.size()) / 1024.0, size);

        Image out;
        double process_ms = 0.0;
        PartitionMap pm;
        PartitionMap* pm_ptr = visualize ? &pm : nullptr;
        master_loop(size, img, filter, threads, out, process_ms, pm_ptr);

        const double t_save0 = MPI_Wtime();
        if (!save_image(out_path, out, err)) {
            std::printf("[Rank 0] ERROR: %s\n", err.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        const double save_ms = (MPI_Wtime() - t_save0) * 1000.0;

        RunRecord rec;
        rec.mode = mode_to_string(mode);
        rec.filter = filter_name(filter);
        rec.width = img.width;
        rec.height = img.height;
        rec.ranks = size;
        rec.threads = threads;
        rec.load_ms = load_ms;
        rec.process_ms = process_ms;
        rec.save_ms = save_ms;
        rec.total_ms = load_ms + process_ms + save_ms;
        rec.checksum = image_checksum(out, /*omp*/ true);
        print_run(rec);
        if (!csv_path.empty()) {
            if (append_csv(csv_path, rec, err)) {
                std::printf("[Rank 0] Benchmark row appended to %s\n", csv_path.c_str());
            } else {
                std::printf("[Rank 0] WARNING: %s\n", err.c_str());
            }
        }
    } else {
        worker_loop(filter, /*omp*/ true, /*want_log*/ visualize);
    }

    MPI_Barrier(MPI_COMM_WORLD);                 // all ranks finish together
    return rc;
}
