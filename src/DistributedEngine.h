// ============================================================================
//  Hybrid Distributed Image Processing Engine
//  DistributedEngine.h - Distributed Engine (MPI) + execution modes.
//
//  Proposal mapping:
//    "Set up multiple worker processes using MPI."
//    "Implement a master-worker setup where the main controller (Rank 0)
//     divides the image, assigns work, and gathers the final results."
//
//  Four execution modes (used by the Benchmarking Module):
//    sequential : single core loop, no OpenMP, no MPI distribution
//    openmp     : shared-memory threads only (run with one MPI rank)
//    mpi        : distributed ranks, exactly one CPU thread per rank
//    hybrid     : distributed ranks, OpenMP threads inside every rank
// ============================================================================
#ifndef DISTRIBUTED_ENGINE_H
#define DISTRIBUTED_ENGINE_H

#include <string>
#include "Image.h"

enum Mode {
    MODE_SEQUENTIAL = 0,
    MODE_OPENMP,
    MODE_MPI,
    MODE_HYBRID,
    MODE_COUNT
};

Mode        mode_from_string(const std::string& s);   // MODE_COUNT if unknown
const char* mode_to_string(Mode m);
bool        mode_is_distributed(Mode m);              // needs mpiexec with ranks > 1

// Run one filter pass in a distributed (MPI) mode.
// Must be called after MPI_Init; finalizes MPI itself.
//   threads_arg : 0 = use OMP_NUM_THREADS / hardware default
//   csv_path    : empty = do not record the run
//   visualize   : print a LIVE master/worker view on the console (split map,
//                 message timeline, per-thread row shares). No extra files
//                 are written - everything happens in the command interface.
int run_distributed(const std::string& in_path, const std::string& out_path,
                    FilterId filter, Mode mode, int threads_arg,
                    const std::string& csv_path, bool visualize = false);

// Horizontal band layout of the master's split: rank r owns rows
// [start, start+rows) of an H-row image. Shared with PartitionMap so the
// console split map always matches what the engine actually sends.
void compute_band_layout(int rank, int size, int H, int& start, int& rows);

#endif // DISTRIBUTED_ENGINE_H
