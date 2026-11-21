@echo off
setlocal
cd /d "%~dp0"

REM If g++ is not in PATH, check WinGet WinLibs installation
where g++ >nul 2>nul
if errorlevel 1 (
    for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs*") do (
        if exist "%%D\mingw64\bin\g++.exe" (
            set "PATH=%%D\mingw64\bin;%PATH%"
        )
    )
)

if not exist himg.exe (
    call build.bat
    if errorlevel 1 exit /b 1
)

set "IMG=%~1"
if "%IMG%"=="" set "IMG=test.ppm"
set "FLT=%~2"
if "%FLT%"=="" set "FLT=gaussian"
set "N=%~3"
if "%N%"=="" set "N=3"

where mpiexec >nul 2>nul
if errorlevel 1 (
    if exist "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" (
        set "PATH=C:\Program Files\Microsoft MPI\Bin;%PATH%"
    ) else (
        echo [X] mpiexec not found - install the MS-MPI runtime ^(msmpisetup.exe^)
        echo     and reopen cmd. Usage: run_demo.bat [image] [filter] [ranks]
        exit /b 1
    )
)

if not exist "%IMG%" (
    echo [+] No image "%IMG%" - generating a test image first...
    himg.exe --make-test "%IMG%" 1280 720
    if errorlevel 1 exit /b 1
)

echo Running live master/worker demo: mpiexec -n %N%  hybrid  %FLT%  on %IMG%
mpiexec -n %N% himg.exe "%IMG%" %FLT% --mode hybrid --threads 2 --visualize
endlocal
