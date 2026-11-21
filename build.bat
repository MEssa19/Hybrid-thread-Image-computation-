@echo off
setlocal
cd /d "%~dp0"
echo ================================================================
echo  Hybrid Distributed Image Processing Engine - cmd build
echo ================================================================

if /i "%~1"=="clean" (
    if exist himg.exe del himg.exe
    if exist msmpi.def del msmpi.def
    if exist libmsmpi.a del libmsmpi.a
    echo Cleaned.
    exit /b 0
)

if "%MSMPI_INC%"=="" (
    echo [X] MS-MPI SDK not found. Install msmpisdk.msi ^(SDK^) and
    echo     msmpisetup.exe ^(runtime^), open a NEW cmd window, and retry.
    exit /b 1
)
if "%MSMPI_INC:~-1%"=="\" set "MSMPI_INC=%MSMPI_INC:~0,-1%"
if defined MSMPI_LIB64 if "%MSMPI_LIB64:~-1%"=="\" set "MSMPI_LIB64=%MSMPI_LIB64:~0,-1%"

set SRC=src\main.cpp src\ImageIO.cpp src\Filters.cpp src\DistributedEngine.cpp src\Benchmark.cpp src\PartitionMap.cpp

where g++ >nul 2>nul
if not errorlevel 1 goto mingw
where cl >nul 2>nul
if not errorlevel 1 goto msvc

REM Auto-detect WinGet WinLibs if installed
for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs*") do (
    if exist "%%D\mingw64\bin\g++.exe" (
        set "PATH=%%D\mingw64\bin;%PATH%"
        goto mingw
    )
)

echo [X] No compiler found in PATH.
echo     Option A: install MinGW-w64 (MSYS2) and add its bin folder to PATH
echo     Option B: open the "x64 Native Tools Command Prompt" and rerun build.bat
exit /b 1

:mingw
echo [+] Building with MinGW g++ ...
g++ -std=c++17 -O2 -fopenmp -I"%MSMPI_INC%" %SRC% -L"%MSMPI_LIB64%" -lmsmpi -o himg.exe
if not errorlevel 1 goto ok
echo [-] Direct link against msmpi.lib failed - converting msmpi.dll with gendef...
where gendef >nul 2>nul
if errorlevel 1 (echo [X] gendef.exe missing - use a full MSYS2 mingw64 toolchain. & exit /b 1)
set "DLL=%MSMPI_BIN%msmpi.dll"
if "%MSMPI_BIN%"=="" set "DLL=C:\Program Files\Microsoft MPI\Bin\msmpi.dll"
if not exist "%DLL%" set "DLL=%SystemRoot%\System32\msmpi.dll"
if not exist "%DLL%" (echo [X] msmpi.dll not found - install the MS-MPI runtime first. & exit /b 1)
gendef "%DLL%"
dlltool -d msmpi.def -D msmpi.dll -l libmsmpi.a
g++ -std=c++17 -O2 -fopenmp -I"%MSMPI_INC%" %SRC% -L. -L"%MSMPI_LIB64%" -lmsmpi -o himg.exe
if errorlevel 1 (echo [X] Build failed. & exit /b 1)
goto ok

:msvc
echo [+] Building with MSVC cl.exe ...
cl /nologo /EHsc /O2 /openmp /std:c++17 /D_CRT_SECURE_NO_WARNINGS /I"%MSMPI_INC%" %SRC% /Fe:himg.exe /link /LIBPATH:"%MSMPI_LIB64%" msmpi.lib
if errorlevel 1 (echo [X] Build failed. & exit /b 1)
goto ok

:ok
echo.
echo [OK] himg.exe built.
echo.
echo Quick start (mpi and hybrid modes need mpiexec):
echo   himg.exe --make-test test.ppm 1920 1080
echo   himg.exe test.ppm gaussian --mode sequential
echo   himg.exe test.ppm gaussian --mode openmp --threads 4 --visualize
echo   mpiexec -n 3 himg.exe test.ppm gaussian --mode hybrid --threads 2 --visualize
where mpiexec >nul 2>nul
if errorlevel 1 (
    if not exist "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" (
        echo [!] mpiexec not in PATH - install the MS-MPI runtime ^(msmpisetup.exe^) and reopen cmd.
    )
)
endlocal
