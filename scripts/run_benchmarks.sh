#!/usr/bin/env bash
# ============================================================================
#  Hybrid Distributed Image Processing Engine
#  run_benchmarks.sh - 4-mode benchmark runner (Linux, reference script)
#
#  Usage:
#    ./run_benchmarks.sh <image.ppm|bmp> <filter> [threads] [ranks]
#    e.g.  ./run_benchmarks.sh test_4k.ppm gaussian 2 2
#
#  Produces times.csv and prints the comparison report.
# ============================================================================
set -u

IMG="${1:-}"
FILTER="${2:-gaussian}"
T="${3:-2}"
R="${4:-2}"
CSV="times.csv"

if [ -z "$IMG" ]; then
    echo "Usage: $0 <image.ppm|bmp> <filter> [threads] [ranks]"
    exit 1
fi

command -v himg >/dev/null 2>&1 || { echo "[ERROR] himg not on PATH. Build first (see README)."; exit 1; }
command -v mpirun >/dev/null 2>&1 || { echo "[ERROR] mpirun not found. Install OpenMPI/MPICH."; exit 1; }

MPIRUN="mpirun --oversubscribe"
# inside docker/WSL root shells OpenMPI refuses to run without this flag
if [ "$(id -u)" = "0" ]; then
    MPIRUN="mpirun --allow-run-as-root --oversubscribe"
fi

rm -f "$CSV"

echo "[1/4] Sequential (single core) ..."
himg "$IMG" "$FILTER" --mode sequential --csv "$CSV" || exit 1

echo "[2/4] OpenMP only ($T threads, 1 process) ..."
himg "$IMG" "$FILTER" --mode openmp --threads "$T" --csv "$CSV" || exit 1

echo "[3/4] MPI only ($R ranks, 1 thread each) ..."
$MPIRUN -np "$R" himg "$IMG" "$FILTER" --mode mpi --csv "$CSV" || exit 1

echo "[4/4] Hybrid MPI + OpenMP ($R ranks x $T threads) ..."
$MPIRUN -np "$R" himg "$IMG" "$FILTER" --mode hybrid --threads "$T" --csv "$CSV" || exit 1

echo
himg --report "$CSV"
echo
echo "Done. Raw data in $CSV. Checksums printed above must match across modes."
