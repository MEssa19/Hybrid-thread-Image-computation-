# ============================================================================
#  Hybrid Distributed Image Processing Engine
#  CMakeLists.txt
#
#  Works with MS-MPI (Windows / Visual Studio) and OpenMPI / MPICH (Linux):
#
#    Windows (after installing MS-MPI SDK + runtime):
#      cmake -S . -B build -G "Visual Studio 17 2022" -A x64
#      cmake --build build --config Release
#
#    Linux:
#      cmake -S . -B build && cmake --build build
# ============================================================================

cmake_minimum_required(VERSION 3.12)
project(HybridImageEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(himg
    src/main.cpp
    src/ImageIO.cpp
    src/Filters.cpp
    src/DistributedEngine.cpp
    src/Benchmark.cpp
    src/PartitionMap.cpp
)

target_include_directories(himg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(himg PRIVATE MPI::MPI_CXX)

if(MSVC)
    target_compile_options(himg PRIVATE /openmp /W3)
    target_compile_definitions(himg PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(himg PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_compile_options(himg PRIVATE -fopenmp)
        target_link_options(himg PRIVATE -fopenmp)
    endif()
    target_compile_options(himg PRIVATE -O2 -Wall -Wextra)
endif()
