// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  PartitionMap.h - Live console visualizer (--visualize).
//
//  Answers the demo question "who processed which part of the image?"
//  LIVE in the command interface while the run happens. Nothing is written
//  to disk: no partition image, no trace file - only console output.
//
//    * print_split_map          master's band split + halo rows (ASCII)
//    * compress_chunks          collapses per-row owner maps into runs
//    * print_partition_summary  final recap table with per-thread bars
//
//  During a distributed run the workers additionally ship short log lines to
//  the master over MPI (tag 6) and the master prints them as they arrive,
//  so the console shows the whole master-worker protocol in real time.
//
//  The engine's pixel pipeline is untouched: the map is only recorded on the
//  side while the normal (bitwise-reproducible) processing runs.
// ============================================================================
#ifndef PARTITION_MAP_H
#define PARTITION_MAP_H

#include <string>
#include <vector>
#include "Image.h"

// Per-row ownership of the work.
struct PartitionMap {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rank_of_row;    // [H] MPI rank that owned row y
    std::vector<unsigned char> thread_of_row;  // [H] OMP thread id that produced row y

    void reset(int w, int h);
    // Number of distinct ranks/threads that actually appear in the map.
    int max_rank() const;
    int max_thread() const;
};

// A contiguous run of rows handled by one owner (rank or thread).
// Rows [begin, end) were produced by owner 'id'.
struct Chunk {
    int begin;
    int end;
    int id;
};

// "row" / "rows" - used by the live console output.
inline const char* rows_word(int n) { return n == 1 ? "row" : "rows"; }

// Collapse a per-row owner map into the shortest list of contiguous runs.
// Returns the number of chunks written (<= max_out).
int compress_chunks(const unsigned char* map, int n, Chunk* out, int max_out);

// ASCII split map printed by the master right after loading the image:
// one block per horizontal band, showing the row range, the owning rank
// and the halo rows that travel with the band.
void print_split_map(int H, int size, int radius, const char* filter);

// Final recap table (console only): per-rank row ranges and the per-thread
// chunk each OpenMP loop produced inside that rank, drawn as bar charts.
void print_partition_summary(const PartitionMap& pm, int ranks, int threads);

#endif // PARTITION_MAP_H
