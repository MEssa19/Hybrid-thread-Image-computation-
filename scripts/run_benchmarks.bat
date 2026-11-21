@echo off
REM ============================================================
REM  Hybrid Distributed Image Processing Engine
REM  run_benchmarks.bat - 4-mode benchmark runner (Windows)
REM
REM  Usage:
REM    run_benchmarks.bat <image.ppm|bmp> <filter> [threads] [ranks]
REM    e.g.  run_benchmarks.bat test_4k.ppm gaussian 4 2
REM
REM  Produces times.csv and prints the comparison report.
REM ============================================================
setlocal

REM Auto-detect MinGW compiler on PATH for OpenMP runtime DLLs
where g++ >nul 2>nul
if errorlevel 1 (
    for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs*") do (
        if exist "%%D\mingw64\bin\g++.exe" (
            set "PATH=%%D\mingw64\bin;%PATH%"
        )
    )
)

REM Auto-detect mpiexec
where mpiexec >nul 2>nul
if errorlevel 1 (
    if exist "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" (
        set "PATH=C:\Program Files\Microsoft MPI\Bin;%PATH%"
    )
)

REM Auto-detect himg.exe in root or scripts directory
where himg.exe >nul 2>nul
if errorlevel 1 (
    if exist "%~dp0..\himg.exe" (
        set "PATH=%~dp0..;%PATH%"
    ) else if exist "%~dp0himg.exe" (
        set "PATH=%~dp0;%PATH%"
    )
)

if "%~1"=="" (
    echo Usage: %~nx0 ^<image.ppm^|bmp^> ^<filter^> [threads] [ranks]
    echo Filters: gaussian sobel sharpen box grayscale median
    exit /b 1
)

set "IMG=%~1"
set "FILTER=%~2"
if "%~3"=="" (set "T=4") else (set "T=%~3")
if "%~4"=="" (set "R=2") else (set "R=%~4")
set "CSV=times.csv"

where himg.exe >nul 2>nul
if errorlevel 1 (
    echo [ERROR] himg.exe not found on PATH. Build the solution first and add
    echo         bin\Release to PATH, or copy himg.exe next to this script.
    exit /b 1
)
where mpiexec >nul 2>nul
if errorlevel 1 (
    echo [ERROR] mpiexec not found. Install the MS-MPI runtime msmpisetup.exe.
    exit /b 1
)

if exist "%CSV%" del "%CSV%"

echo [1/4] Sequential (single core) ...
himg "%IMG%" %FILTER% --mode sequential --csv "%CSV%"
if errorlevel 1 goto failed

echo [2/4] OpenMP only (%T% threads, 1 process) ...
himg "%IMG%" %FILTER% --mode openmp --threads %T% --csv "%CSV%"
if errorlevel 1 goto failed

echo [3/4] MPI only (%R% ranks, 1 thread each) ...
mpiexec -n %R% himg "%IMG%" %FILTER% --mode mpi --csv "%CSV%"
if errorlevel 1 goto failed

echo [4/4] Hybrid MPI + OpenMP (%R% ranks x %T% threads) ...
mpiexec -n %R% himg "%IMG%" %FILTER% --mode hybrid --threads %T% --csv "%CSV%"
if errorlevel 1 goto failed

echo.
himg --report "%CSV%"
echo.
echo Done. Raw data in %CSV%. Checksums printed above must match across modes.
exit /b 0

:failed
echo.
echo [ERROR] A benchmark run failed. See the message above.
exit /b 1
