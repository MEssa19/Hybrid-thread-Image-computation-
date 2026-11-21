@echo off
setlocal
cd /d "%~dp0"

REM Check for compiler and build himg.exe if needed
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
    if errorlevel 1 (
        echo [X] Could not build himg.exe. Check your environment and try again.
        pause
        exit /b 1
    )
)

where mpiexec >nul 2>nul
if errorlevel 1 (
    if exist "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" (
        set "PATH=C:\Program Files\Microsoft MPI\Bin;%PATH%"
    )
)

echo Starting Interactive Image Processing Studio...
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\interactive.ps1"
endlocal
