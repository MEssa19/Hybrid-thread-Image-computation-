# Hybrid Distributed Image Processing Engine

**Course:** Parallel and Distributed Computing
**Language / IDE:** C++17 / cmd terminal (MinGW-w64 or Visual Studio Build Tools), MS-MPI + OpenMP
**GUI:** Console application (CLI)
**Project type:** Academic software implementation
**Team:** Muhammad Fawad Hassan Ali (67011), Muhammad Essa (67201), Hamza Ashfaq

A high-performance C++ image filtering engine that demonstrates **MPI** (distributed
memory), **OpenMP** (shared memory) and a **master-worker** architecture. The master
process (Rank 0) slices an image into horizontal bands, sends them to worker processes
with `MPI_Send`, every worker filters its band with OpenMP threads, and the master
stitches the results back together and reports the measured execution time.

Four execution modes let you quantify every layer of parallelism:

| Mode | Distribution | Threads | Run as |
|---|---|---|---|
| `sequential` | none (single core) | 1 | `himg ...` |
| `openmp` | none | N threads | `himg ... --threads N` |
| `mpi` | N ranks | 1 per rank | `mpiexec -n N himg ... --mode mpi` |
| `hybrid` | N ranks | M threads per rank | `mpiexec -n N himg ... --mode hybrid --threads M` |

---

## 1. Module map

| Module (proposal) | Source file | What it does |
|---|---|---|
| I/O and File Management | `src/ImageIO.cpp` | PPM (P6/P3) and 24-bit BMP readers/writers, no external libraries |
| Filter Module | `src/Filters.cpp` | Gaussian 5x5, Sobel, Sharpen, Box 3x3, Grayscale, Median 3x3 — all in OpenMP parallel loops |
| Distributed Engine (MPI) | `src/DistributedEngine.cpp` | Master-worker band distribution, halo rows, gather & stitch |
| Partition Visualizer | `src/PartitionMap.cpp` | `--visualize`: LIVE console view of the band split, the MPI message timeline and each thread's row share (no extra files) |
| Benchmarking Module | `src/Benchmark.cpp` | `MPI_Wtime` / `std::chrono` timing, `times.csv` recorder, comparison report with speedup |
| Synchronization | `src/Filters.cpp` (`image_checksum`) | `#pragma omp critical` merge + implicit OpenMP barriers; used to prove outputs are pixel-exact |
| CLI front end | `src/main.cpp` | Argument parsing, mode dispatch, error handling |
| Shared container | `src/Image.h` | Simple interleaved-RGB image struct + filter metadata |

## 2. How the distributed engine works

```
             Rank 0 (master)                         Rank 1..N-1 (workers)
  ┌─────────────────────────────────┐      ┌────────────────────────────────┐
  │ load image (PPM/BMP)            │      │ MPI_Recv work header  (tag 0)  │
  │ split into horizontal bands     │      │ MPI_Recv window pixels (tag 1) │
  │  + radius halo rows per band    │ ───▶ │                                │
  │ MPI_Send header + window        │      │ OpenMP parallel for over rows  │
  │                                 │      │  → filter the band             │
  │ process own band (OpenMP)       │      │ MPI_Send result header (tag 3) │
  │ MPI_Recv results (any order)    │ ◀─── │ MPI_Send result pixels (tag 4) │
  │ stitch bands → save → print time│      │                                │
  └─────────────────────────────────┘      └────────────────────────────────┘
```

* **Halo rows:** a band is extended by the kernel radius (2 rows for Gaussian, 1 for
  the 3x3 filters) on each side, so interior band borders filter with *real*
  neighbouring pixels; only the true image borders are clamped.
  No thread or process ever writes outside its own output band, so there is no data
  race and the stitching is seamless by construction.
* **Synchronization:** inside each rank, `#pragma omp parallel for` over rows provides
  the shared-memory parallelism (implicit barriers at region start/end); the
  integrity checksum is merged with `#pragma omp critical` — a textbook
  mutual-exclusion demo that also serves as a correctness check.
* **All pixel math is integer-only**, so sequential, OpenMP, MPI and Hybrid runs
  produce *bitwise identical* outputs. The printed checksum must match across modes —
  this is the built-in "no tearing, no corruption" test.

## 3. Filters

| Name | Kernel | Notes |
|---|---|---|
| `gaussian` | 5x5, sum 273 (`[[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],...]`) | smooth blur |
| `sobel` | Gx/Gy 3x3 on integer luminance, output `\|Gx\|+\|Gy\|` | edge detection |
| `sharpen` | 3x3 `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | edge enhancement |
| `box` | 3x3 all ones / 9 | mean blur |
| `grayscale` | luminance `(77R+151G+28B)>>8` | replicated to RGB |
| `median` | non-linear 3x3, per channel | noise removal (heaviest filter) |

## 4. Requirements

**Windows (primary target)**
* MS-MPI **SDK** (`msmpisdk.msi`) — provides `mpi.h` + `msmpi.lib` and sets the
  `MSMPI_INC` / `MSMPI_LIB64` environment variables used by the build
* MS-MPI **runtime** (`msmpisetup.exe`) — provides `mpiexec`
* Both installers: <https://learn.microsoft.com/en-us/message-passing-interface/message-passing-interface-in-microsoft-vc>
* A C++ compiler — **any one** of:
  * MinGW-w64 (no Visual Studio needed): MSYS2 (`pacman -S mingw-w64-ucrt-x86_64-gcc`)
    or w64devkit, with its `bin` folder on PATH — the `build.bat` cmd path uses this
  * Visual Studio 2019/2022 with "Desktop development with C++" (or just the Build
    Tools, which give you the *x64 Native Tools Command Prompt*)

**Linux (optional, used to verify this code):** `g++`, OpenMP, OpenMPI or MPICH
(`sudo apt install openmpi-bin libopenmpi-dev`).

## 5. Building

### 5.1 From the cmd terminal (recommended, no Visual Studio)

A `build.bat` ships with the project. Open a plain **cmd** window in the project
folder and run:

```bat
build.bat
```

The script:
1. checks `MSMPI_INC` / `MSMPI_LIB64` (set by the MS-MPI SDK installer),
2. picks whatever compiler is in PATH — **MinGW `g++` first, MSVC `cl` second**,
3. compiles every `src/*.cpp` with `-O2 -fopenmp` and links `msmpi.lib` into
   `himg.exe` in the project root,
4. if the MinGW link against `msmpi.lib` fails, converts `msmpi.dll` automatically
   with `gendef` + `dlltool` and retries,
5. prints the quick-start commands and warns when `mpiexec` is missing.

`build.bat clean` removes `himg.exe` and the conversion leftovers. If you build
with MinGW, keep MinGW's `bin` on PATH when you run the engine (it provides the
OpenMP runtime DLL `libgomp-1.dll`).

### 5.2 Visual Studio (optional)

**Option A — open the shipped solution**
1. Install the two MS-MPI installers above.
2. Open `HybridImageEngine.sln` in Visual Studio, accept the toolset retarget prompt.
3. Select *Release | x64*, build (Ctrl+Shift+B). Output: `bin\Release\himg.exe`.

**Option B — CMake (VS "Open Folder")**
1. *File → Open → Folder…* and pick this project folder.
2. VS auto-configures with `CMakeLists.txt` (FindMPI detects MS-MPI automatically).
3. Choose *x64-Release*, *Build → Build All*. Output: `build\x64-Release\himg.exe`.

**Option C — manual empty project**
1. *Create a new project → Console App (empty project)*, add all `src/*.cpp`.
2. Project *Properties → C/C++ → Language → C++ Language Standard:* **ISO C++17**;
   *C/C++ → All Options → Open MP Support:* **Yes (/openmp)**.
3. *C/C++ → General → Additional Include Directories:* `$(MSMPI_INC)`
   *Linker → General → Additional Library Directories:* `$(MSMPI_LIB64)`
   *Linker → Input → Additional Dependencies:* `msmpi.lib`
4. Build as x64.

## 6. Running

### 6.0 Interactive Studio (Custom Image Upload & Live Worker Visualization)

To process your own picture with a guided interactive menu:

```bat
run_interactive.bat
```

* **Custom Image Upload:** Drag & drop any image file (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.ppm`) directly into the terminal or type its path (or press Enter for an auto-generated sample image).
* **Automatic Format Conversion:** Automatically converts JPG/PNG to 24-bit RGB for the processing engine, and saves the final filtered result in your original image format.
* **Filter Selection:** Choose between `gaussian`, `sobel`, `sharpen`, `box`, `grayscale`, or `median`.
* **Parallelism Controls:** Select execution mode (`sequential`, `openmp`, `mpi`, or `hybrid`) and customize OpenMP thread counts and MPI rank counts.
* **Worker Partition & Transfer Breakdown:** Displays an exact visual blueprint of how the image height is partitioned into horizontal bands, halo row margins for boundary preservation, and the MPI data payload sizes sent from Rank 0 (Master) to each Worker rank.
* **Live Execution & Timeline:** Prints real-time transmission timestamps (`MASTER -> RANK N`), worker thread assignments, stitching logs, and integrity checksums.
* **Direct Image Viewing:** Automatically prompts to open and view the finished filtered image in your default image viewer.

### 6.1 One-command demo from cmd

```bat
REM builds himg.exe if needed, creates test.ppm if needed, then runs
REM the live master/worker view: mpiexec -n 3 hybrid gaussian
run_demo.bat

REM or with your own image / filter / rank count:
run_demo.bat input.bmp sobel 2
```

### 6.1 CLI reference

```
himg <input.(ppm|bmp)> <filter> [options]
himg --make-test <out.(ppm|bmp)> [width] [height]   (default 1920x1080)
himg --list-filters
himg --report times.csv
```

No image handy? `himg --make-test test.ppm 1920 1080` generates a colorful
synthetic picture (gradient, stripes, checkerboard, ring and discs) - exactly
the kind of content that shows blur, edges and flat regions well. It also works
tiny (`himg --make-test tiny.ppm 8 3`) for the idle-rank edge case.

Options: `--mode sequential|openmp|mpi|hybrid` (default **hybrid**),
`--threads N`, `--out PATH`, `--csv PATH`, `--visualize` (see section 6.3).

```bat
REM single-core baseline
himg test_4k.ppm gaussian --mode sequential --csv times.csv

REM shared-memory only (2 threads)
himg test_4k.ppm gaussian --mode openmp --threads 2 --csv times.csv

REM distributed, one thread per rank
mpiexec -n 2 himg test_4k.ppm gaussian --mode mpi --csv times.csv

REM hybrid: 2 processes x 2 threads each
mpiexec -n 2 himg test_4k.ppm gaussian --mode hybrid --threads 2 --csv times.csv

REM comparison table with speedups
himg --report times.csv
```

Sample output of one run:

```
[Rank 0] Distributed Engine - mode=hybrid filter=gaussian ranks=2 threads/rank=2
[Rank 0] Loaded 3840x2160 image (24300.0 KB), slicing into 2 band(s)
  ------------------------------------------------------------
  Mode     : hybrid       Filter   : gaussian
  Image    : 3840x2160  Ranks    : 2  x 2 thread(s)/rank
  Load     :      6.86 ms      Save     :     10.08 ms
  Process  :    447.21 ms      Total    :    464.15 ms
  Checksum : 2917518672  (must match across all modes)
  ------------------------------------------------------------
```

Sequential and OpenMP modes run `himg.exe` directly; MPI/Hybrid modes are launched
through `mpiexec` (with `-n 1` they degenerate to the OpenMP mode gracefully).

### 6.3 Watching the split live: `--visualize`

Add `--visualize` to any run and the **command interface itself** shows how the
master divides the image and how the workers/threads share it — printed in real
time while the run happens. **No extra files are written** (no partition image,
no trace file): everything is console output.

```bat
REM hybrid demo: 2 workers, 2 threads each (the classic demo line)
mpiexec -n 3 himg input.ppm gaussian --mode hybrid --threads 2 --visualize

REM works in every mode:
himg        input.ppm gaussian --mode openmp   --threads 4 --visualize
mpiexec -n 4 himg input.ppm gaussian --mode mpi                 --visualize
```

What you see (real transcript of the hybrid run above, 1920x1080 image):

```
==============================================================
  LIVE MASTER/WORKER VIEW - who owns which rows, in real time
  mode=hybrid  filter=gaussian  ranks=3  threads/rank=2
==============================================================
[Rank 0] Loaded 1920x1080 image (6075.0 KB), slicing into 3 band(s)

  MASTER SPLITS  1080 rows  ->  3 BAND(S)   (gaussian, halo = 2 rows)
  row     0  +----------------------------------------------------------------+
              | BAND 1  -> RANK 0  (master keeps it) | rows 0..359 (360 rows)  |
              | halo: top none (image edge) | bottom 2 (rows 360-361)          |
  row   360  +----------------------------------------------------------------+
              | BAND 2  -> RANK 1  | rows 360..719 (360 rows)                  |
              | halo: top 2 (rows 358-359) | bottom 2 (rows 720-721)           |
  row   720  +----------------------------------------------------------------+
              | BAND 3  -> RANK 2  | rows 720..1079 (360 rows)                 |
              | halo: top 2 (rows 718-719) | bottom none (image edge)          |
  row  1079  +----------------------------------------------------------------+
  RANK 0 processes band 1 locally; bands 2..3 travel to the workers.
[t+     10.5 ms] MASTER -> RANK 1 | band 2 sent: rows 360..719 (360 rows, halo +2/+2) = 2047.5 KB
[t+     11.7 ms] MASTER -> RANK 2 | band 3 sent: rows 720..1079 (360 rows, halo +2/+0) = 2036.2 KB
[t+     11.7 ms] RANK 1 | band 2 received: rows 360..719 (360 rows, halo +2/+2), window 2047.5 KB
[t+     11.7 ms] RANK 1 | 2 OpenMP thread(s) start on the band's 360 rows (shared memory)
[t+     12.4 ms] MASTER (RANK 0) | band 1: 2 OpenMP thread(s) start on rows 0..359 (360 rows)
[t+    120.2 ms] MASTER (RANK 0) | band 1 done in 107.8 ms
[t+    120.3 ms] MASTER (RANK 0) | band 1 shared 360 rows: T0: 0-179 T1: 180-359
[t+    121.2 ms] RANK 1 | band done in 102.7 ms | shared 360 rows: T0: 360-539 T1: 540-719
[t+    121.2 ms] RANK 1 | sending 360 result rows back to the master
[t+    121.9 ms] RANK 1 -> MASTER | band 2 stitched into rows 360..719 (360 rows)
[t+    122.6 ms] RANK 2 -> MASTER | band 3 stitched into rows 720..1079 (360 rows)

  WORK RECAP - who processed which rows (ranks=3, threads/rank=2)
  RANK 0 : rows     0 -   359 (360 rows)
          T0  [############............] rows     0 -   179 (180 rows)
          T1  [############............] rows   180 -   359 (180 rows)
  RANK 1 : rows   360 -   719 (360 rows)
          T0  [############............] rows   360 -   539 (180 rows)
          T1  [############............] rows   540 -   719 (180 rows)
  RANK 2 : rows   720 -  1079 (360 rows)
          T0  [############............] rows   720 -   899 (180 rows)
          T1  [############............] rows   900 -  1079 (180 rows)
  Legend: '#' bar length = share of the band's rows, Tn = OpenMP thread n.
```

How to read it:

* **Split map** — the master's horizontal banding: every band shows its row
  range, the rank that will own it, and the **halo rows** that travel with it
  so borders filter correctly across band seams.
* **Timeline lines `[t+ ... ms]`** — every MPI event with its timestamp:
  `MASTER -> RANK n` = work message sent, `RANK n |` = a worker's live report
  (shipped to the master over MPI tag 6 and printed the moment it arrives),
  `RANK n -> MASTER` = result stitched back. Results arrive in **any order**
  and the recap still puts every row in the right place.
* **Thread sharing** — the `T0: ... T1: ...` parts are the actual row chunks
  each OpenMP thread processed inside its band (measured, not assumed, via
  `omp_get_thread_num()` in the work-sharing loop). With `schedule(static)`
  each thread owns one contiguous chunk — exactly the "team shares the band"
  story. The recap's bar length is proportional to the share.
* **Idle ranks** — if the image has fewer rows than ranks, the extra bands are
  shown as `0 rows - rank stays idle` end to end (send, log, stitch, recap).

The view is printed only when `--visualize` is given; run benchmarks **without**
it so nothing interferes with the timings. The visualization rides along with
the normal protocol (workers return their per-row thread ids with MPI tag 5),
so the processed image stays bitwise identical in all four modes — verified by
MD5 comparison with and without `--visualize`. To keep a copy of the view for
the report, redirect the console output (e.g. `... --visualize > demo.txt`).

## 7. Benchmarking

`scripts\run_benchmarks.bat <image> <filter> [threads] [ranks]` runs all four modes,
records them into `times.csv` (one row per run:
`mode,filter,width,height,ranks,threads,load_ms,process_ms,save_ms,total_ms,checksum`)
and prints the comparison report. `himg --report times.csv` regenerates the table at
any time. Speedup is `T_sequential / T_mode` (process phase).

### Reference results (verification machine)

Measured with the code in this folder on a **2-core Linux VM, OpenMPI 5.0.7, g++ 14 `-O2`**,
Gaussian filter, 3840x2160 test image — your numbers on Windows will differ:

| Mode | Ranks x Threads | Process (ms) | Total (ms) | Speedup |
|---|---|---|---|---|
| sequential | 1 x 1 | 625.34 | 670.90 | 1.00x |
| openmp | 1 x 2 | 387.70 | 424.41 | 1.61x |
| mpi | 2 x 1 | 434.60 | 479.16 | 1.44x |
| hybrid | 2 x 2 | 447.21 | 464.15 | 1.40x |

On a machine with 4+ physical cores the MPI and Hybrid modes pull further ahead
(distribution across real cores instead of oversubscribed threads). With small images
the parallel overhead (thread spawn, message passing) can make parallel modes *slower*
than sequential — that expected behaviour is exactly what the proposal's evaluation
section anticipates.

### Correctness verification performed on this code

* All six filters: sequential vs OpenMP checksums match (HD image).
* 4K image, all four modes: **identical MD5 output files** and identical checksums —
  no tearing, no deadlocks, seamless stitching.
* 8x3 image distributed over 4 ranks (empty-band edge case): identical output.
* `mpiexec -n 1` degenerates to OpenMP-only with identical output.
* BMP and PPM loaders produce identical pixel data for the same image
  (Gaussian checksum of the HD test image: 731049793 through both paths).
* `--visualize` regression: HD image, all four modes with visualization enabled —
  output MD5s still all identical (68075c96… for the HD Gaussian run), proving the
  live view records its data without touching the pipeline. Workers report only
  through extra MPI messages (tags 5/6); the pixel math never changes.

## 8. Project structure

```
HybridImageEngine/
├── build.bat                    one-command cmd build (MinGW g++ or MSVC cl)
├── run_demo.bat                 one-command live demo (build + test image + mpiexec)
├── HybridImageEngine.sln        Visual Studio solution (x64)
├── HybridImageEngine.vcxproj    VS project (MS-MPI include/lib wired via $(MSMPI_*))
├── CMakeLists.txt               CMake build (VS Open Folder / Linux)
├── README.md
├── src/
│   ├── Image.h                  image struct + filter metadata
│   ├── ImageIO.h/.cpp           PPM/BMP I/O module
│   ├── Filters.h/.cpp           filter kernels + OpenMP loops + checksum
│   ├── DistributedEngine.h/.cpp MPI master-worker engine + mode logic
│   ├── PartitionMap.h/.cpp      --visualize live console visualizer (split map,
│   │                            message timeline, per-thread recap bars)
│   ├── Benchmark.h/.cpp         timing, times.csv, --report table
│   └── main.cpp                 CLI
└── scripts/
    ├── run_benchmarks.bat       4-mode benchmark runner (Windows)
    └── run_benchmarks.sh        4-mode benchmark runner (Linux)
```

## 9. Troubleshooting (Windows / MS-MPI)

* **`mpiexec` not found** — install the MS-MPI *runtime* (`msmpisetup.exe`), then
  reopen the terminal so PATH updates apply.
* **MinGW build: `undefined reference to MPI_*`** — the automatic `gendef`+`dlltool`
  fallback inside `build.bat` needs the full MSYS2 mingw64 toolchain (binutils).
  Alternatively create `libmsmpi.a` once: `gendef msmpi.dll` then
  `dlltool -d msmpi.def -D msmpi.dll -l libmsmpi.a`, and rebuild.
* **MinGW build: `libgomp-1.dll` not found when running** — run from the same cmd
  where MinGW's `bin` is on PATH (that folder ships the OpenMP runtime DLL), or copy
  `libgomp-1.dll` next to `himg.exe`.
* **MSVC build from cmd** — open the *x64 Native Tools Command Prompt* (ships with
  Build Tools/VS) so `cl.exe` is in PATH, then run `build.bat`; it switches to `cl`
  automatically.
* **Link error `msmpi.lib` not found** — the SDK is missing or you are building Win32;
  build x64 (`$(MSMPI_LIB64)`) or switch the lib directory to `$(MSMPI_LIB32)`.
* **`mpiexec` hangs or errors on first run** — start the MS-MPI service manager once:
  `smpd -d` in an administrator console, or reboot after installing the runtime.
* **Windows Firewall prompt** — allow `mpiexec`/`smpd` on private networks.
* **Hybrid slower than MPI with high thread counts** — normal on machines with few
  physical cores; benchmark with `--threads` close to the number of physical cores.
* **Console output ordering when multiple ranks print** — by design every worker
  ships its live log lines to the master (MPI tag 6) and only Rank 0 prints, so
  the `--visualize` timeline stays ordered even under real mpiexec.

## 10. Scope and limitations

Educational implementation for the PDC course, not a commercial graphics engine.
Only uncompressed images (PPM P6/P3, 24-bit BMP) are supported by design to avoid
external libraries (no OpenCV/libjpeg); filters target multi-core CPUs only
(no GPU/accelerators), so every PDC concept from the course — multithreading,
shared memory, distributed memory, master-worker, synchronization, and measured
performance evaluation — stays visible in the code.
