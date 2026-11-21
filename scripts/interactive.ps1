<#
============================================================================
 Hybrid Distributed Image Processing Engine
 interactive.ps1 - Interactive CLI Runner with Live Worker Partitioning
============================================================================
#>

[CmdletBinding()]
param()

# Ensure System.Drawing is available for image handling
Add-Type -AssemblyName System.Drawing -ErrorAction SilentlyContinue

$WorkspaceDir = (Split-Path -Parent $PSScriptRoot)
Set-Location $WorkspaceDir

# Auto-detect WinLibs MinGW
$WingetMinGW = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter "g++.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($WingetMinGW) {
    $env:PATH = "$($WingetMinGW.DirectoryName);$env:PATH"
}

# Auto-detect MS-MPI
if (Test-Path "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe") {
    $env:PATH = "C:\Program Files\Microsoft MPI\Bin;$env:PATH"
}

# Ensure himg.exe is built
if (-not (Test-Path "$WorkspaceDir\himg.exe")) {
    Write-Host "`n[+] himg.exe not found. Building now via build.bat..." -ForegroundColor Cyan
    cmd /c "$WorkspaceDir\build.bat"
    if (-not (Test-Path "$WorkspaceDir\himg.exe")) {
        Write-Host "[X] Build failed. Please check compiler and MS-MPI installations." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

function Show-Banner {
    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "             HYBRID DISTRIBUTED IMAGE PROCESSING ENGINE" -ForegroundColor White
    Write-Host "       Master-Worker MPI (Distributed) + OpenMP (Shared Memory) CLI" -ForegroundColor Yellow
    Write-Host "================================================================================`n" -ForegroundColor Cyan
}

function Convert-ToBmp24 {
    param(
        [string]$SourcePath,
        [string]$DestPath
    )
    try {
        $srcBmp = [System.Drawing.Bitmap]::FromFile((Resolve-Path $SourcePath).Path)
        $w = $srcBmp.Width
        $h = $srcBmp.Height
        $bmp24 = New-Object System.Drawing.Bitmap($w, $h, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
        $g = [System.Drawing.Graphics]::FromImage($bmp24)
        $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::NearestNeighbor
        $g.DrawImage($srcBmp, 0, 0, $w, $h)
        $g.Dispose()
        $srcBmp.Dispose()
        $bmp24.Save($DestPath, [System.Drawing.Imaging.ImageFormat]::Bmp)
        $bmp24.Dispose()
        return $true
    } catch {
        Write-Host "[!] Error converting image to BMP: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Convert-FromBmp {
    param(
        [string]$BmpPath,
        [string]$DestPath,
        [string]$Format
    )
    try {
        $bmp = [System.Drawing.Bitmap]::FromFile((Resolve-Path $BmpPath).Path)
        $imgFormat = [System.Drawing.Imaging.ImageFormat]::Bmp
        switch ($Format.ToLower()) {
            ".png"  { $imgFormat = [System.Drawing.Imaging.ImageFormat]::Png }
            ".jpg"  { $imgFormat = [System.Drawing.Imaging.ImageFormat]::Jpeg }
            ".jpeg" { $imgFormat = [System.Drawing.Imaging.ImageFormat]::Jpeg }
            ".gif"  { $imgFormat = [System.Drawing.Imaging.ImageFormat]::Gif }
        }
        $bmp.Save($DestPath, $imgFormat)
        $bmp.Dispose()
        return $true
    } catch {
        return $false
    }
}

function Display-WorkerSplitPlan {
    param(
        [int]$W,
        [int]$H,
        [string]$FilterName,
        [int]$Radius,
        [int]$Ranks,
        [int]$Threads
    )
    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host "                IMAGE PARTITION & WORKER TRANSFER PLAN" -ForegroundColor White
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  Image Dimensions : $W x $H pixels" -ForegroundColor Gray
    Write-Host "  Selected Filter  : $FilterName (Halo Overlap Radius = $Radius row(s))" -ForegroundColor Gray
    Write-Host "  Process Topology : $Ranks MPI Rank(s) x $Threads OpenMP Thread(s)/Rank" -ForegroundColor Gray
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  HOW THE IMAGE IS SPLIT & TRANSFERRED TO WORKERS:" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan

    $base = [math]::Floor($H / $Ranks)
    $rem = $H % $Ranks

    for ($r = 0; $r -lt $Ranks; $r++) {
        $rows = $base + ($(if ($r -lt $rem) { 1 } else { 0 }))
        $start = ($r * $base) + ($(if ($r -lt $rem) { $r } else { $rem }))
        $end = $start + $rows - 1

        $y0 = [math]::Max(0, $start - $Radius)
        $y1 = [math]::Min($H, $start + $rows + $Radius)
        $haloTop = $start - $y0
        $haloBot = $y1 - ($start + $rows)
        $windowRows = $y1 - $y0
        $windowBytes = $windowRows * $W * 3
        $windowKb = [math]::Round($windowBytes / 1024, 1)

        if ($r -eq 0) {
            Write-Host "  [RANK 0 : MASTER]" -ForegroundColor Green
            Write-Host "    * Assigned Band : Rows $start..$end ($rows rows) [Kept locally in Rank 0 memory]" -ForegroundColor White
            Write-Host "    * Halo Buffers  : Top: None (image edge) | Bottom: +$haloBot rows (rows $start..$($start+$rows+$haloBot-1))" -ForegroundColor Gray
            Write-Host "    * Processing    : Shared among $Threads local OpenMP thread(s)" -ForegroundColor Gray
        } else {
            Write-Host "  [RANK $r : WORKER]" -ForegroundColor Cyan
            Write-Host "    * Assigned Band : Rows $start..$end ($rows rows)" -ForegroundColor White
            Write-Host "    * Halo Overlap  : Top: +$haloTop rows | Bottom: +$haloBot rows" -ForegroundColor Gray
            Write-Host "    * MPI Transfer  : Master -> Rank $r sending window of $windowRows rows ($windowKb KB)" -ForegroundColor Magenta
            Write-Host "                      MPI_Send(Header, Tag 0) + MPI_Send(WindowPixels, Tag 1)" -ForegroundColor DarkGray
            Write-Host "    * Processing    : Worker Rank $r filters band with $Threads OpenMP thread(s)" -ForegroundColor Gray
            Write-Host "    * MPI Return    : Worker -> Master sends $rows filtered rows back" -ForegroundColor Magenta
            Write-Host "                      MPI_Send(ResultHeader, Tag 3) + MPI_Send(ResultPixels, Tag 4)" -ForegroundColor DarkGray
        }
        Write-Host ""
    }
    Write-Host "  Master stitches all returned bands into the final full-resolution image." -ForegroundColor Yellow
    Write-Host "================================================================================`n" -ForegroundColor Cyan
}

# Main Interactive Loop
do {
    Show-Banner

    # ------------------------------------------------------------------------
    # STEP 1: Image Selection
    # ------------------------------------------------------------------------
    Write-Host "STEP 1: SELECT IMAGE" -ForegroundColor Green
    Write-Host "  Provide your own image file (PNG, JPG, BMP, PPM) or press [Enter] to generate a test image." -ForegroundColor Gray
    Write-Host "  (Tip: You can drag & drop an image file directly into this terminal window!)`n" -ForegroundColor DarkGray

    $rawInput = Read-Host "Enter image path (or press Enter for test image)"
    $rawInput = $rawInput.Trim("`"").Trim("'").Trim()

    $origPath = ""
    $isTempGenerated = $false
    $inputToEngine = ""

    if ([string]::IsNullOrWhiteSpace($rawInput)) {
        $origPath = "$WorkspaceDir\test.ppm"
        Write-Host "`n[+] Generating test sample image (1280x720 PPM)..." -ForegroundColor Yellow
        cmd /c "himg.exe --make-test `"$origPath`" 1280 720"
        $inputToEngine = $origPath
    } else {
        if (-not (Test-Path $rawInput)) {
            Write-Host "[X] File not found: $rawInput" -ForegroundColor Red
            $retry = Read-Host "Press Enter to try again"
            continue
        }
        $origPath = (Resolve-Path $rawInput).Path
        $ext = [System.IO.Path]::GetExtension($origPath).ToLower()

        if ($ext -eq ".ppm" -or $ext -eq ".bmp") {
            $inputToEngine = $origPath
        } else {
            # Convert non-BMP/PPM to 24-bit BMP for the C++ engine
            Write-Host "[+] Converting $ext image to 24-bit BMP for processing engine..." -ForegroundColor Yellow
            $convertedBmp = "$WorkspaceDir\input_converted.bmp"
            $ok = Convert-ToBmp24 -SourcePath $origPath -DestPath $convertedBmp
            if (-not $ok) {
                Write-Host "[X] Image conversion failed. Please try a different image." -ForegroundColor Red
                continue
            }
            $inputToEngine = $convertedBmp
            $isTempGenerated = $true
        }
    }

    # Inspect image size
    $imgWidth = 0
    $imgHeight = 0
    try {
        $probe = [System.Drawing.Bitmap]::FromFile($inputToEngine)
        $imgWidth = $probe.Width
        $imgHeight = $probe.Height
        $probe.Dispose()
        Write-Host "[OK] Image Loaded: $origPath ($imgWidth x $imgHeight)" -ForegroundColor Green
    } catch {
        # Fallback if System.Drawing cannot open (e.g. PPM)
        $imgWidth = 1280
        $imgHeight = 720
        Write-Host "[OK] Image Loaded: $origPath" -ForegroundColor Green
    }

    # ------------------------------------------------------------------------
    # STEP 2: Filter Selection
    # ------------------------------------------------------------------------
    Write-Host "`nSTEP 2: SELECT FILTER" -ForegroundColor Green
    Write-Host "  [1] Gaussian Blur   (5x5 smooth blur, weighted Gaussian matrix)"
    Write-Host "  [2] Sobel Edge      (3x3 edge detection, gradient magnitude |Gx|+|Gy|)"
    Write-Host "  [3] Sharpen         (3x3 high-pass edge enhancement)"
    Write-Host "  [4] Box Blur        (3x3 simple mean averaging)"
    Write-Host "  [5] Grayscale       (Luminance conversion: (77R + 151G + 28B) >> 8)"
    Write-Host "  [6] Median Filter   (3x3 non-linear noise reduction)`n"

    $filterChoice = Read-Host "Choose filter (1-6) [default: 1 (gaussian)]"
    $filter = "gaussian"
    $filterRadius = 2
    $filterDesc = "Gaussian Blur (5x5)"

    switch ($filterChoice.Trim()) {
        "2" { $filter = "sobel";     $filterRadius = 1; $filterDesc = "Sobel Edge Detection (3x3)" }
        "3" { $filter = "sharpen";   $filterRadius = 1; $filterDesc = "Sharpen Filter (3x3)" }
        "4" { $filter = "box";       $filterRadius = 1; $filterDesc = "Box Blur (3x3)" }
        "5" { $filter = "grayscale"; $filterRadius = 0; $filterDesc = "Grayscale Conversion" }
        "6" { $filter = "median";    $filterRadius = 1; $filterDesc = "Median Filter (3x3)" }
        default { $filter = "gaussian"; $filterRadius = 2; $filterDesc = "Gaussian Blur (5x5)" }
    }
    Write-Host "Selected: $filterDesc" -ForegroundColor Cyan

    # ------------------------------------------------------------------------
    # STEP 3: Execution Mode & Parallelism
    # ------------------------------------------------------------------------
    Write-Host "`nSTEP 3: SELECT EXECUTION MODE" -ForegroundColor Green
    Write-Host "  [1] Sequential   (Single-core baseline, 1 process, 1 thread)"
    Write-Host "  [2] OpenMP       (Shared-memory multi-threaded, 1 process, N threads)"
    Write-Host "  [3] MPI          (Distributed-memory multi-process, N ranks, 1 thread each)"
    Write-Host "  [4] Hybrid       (Distributed MPI master-worker + OpenMP threads) [RECOMMENDED]`n"

    $modeChoice = Read-Host "Choose mode (1-4) [default: 4 (hybrid)]"
    $mode = "hybrid"
    switch ($modeChoice.Trim()) {
        "1" { $mode = "sequential" }
        "2" { $mode = "openmp" }
        "3" { $mode = "mpi" }
        default { $mode = "hybrid" }
    }
    Write-Host "Selected Mode: $mode" -ForegroundColor Cyan

    $ranks = 1
    $threads = 1

    if ($mode -eq "openmp") {
        $threadsInput = Read-Host "`nEnter OpenMP thread count [default: 4]"
        if ([int]::TryParse($threadsInput, [ref]$threads) -and $threads -gt 0) { } else { $threads = 4 }
    } elseif ($mode -eq "mpi") {
        $ranksInput = Read-Host "`nEnter number of MPI ranks [default: 3]"
        if ([int]::TryParse($ranksInput, [ref]$ranks) -and $ranks -gt 0) { } else { $ranks = 3 }
    } elseif ($mode -eq "hybrid") {
        $ranksInput = Read-Host "`nEnter number of MPI ranks [default: 3]"
        if ([int]::TryParse($ranksInput, [ref]$ranks) -and $ranks -gt 0) { } else { $ranks = 3 }
        $threadsInput = Read-Host "Enter OpenMP threads per rank [default: 2]"
        if ([int]::TryParse($threadsInput, [ref]$threads) -and $threads -gt 0) { } else { $threads = 2 }
    }

    # ------------------------------------------------------------------------
    # STEP 4: Worker Split & Transfer Visualization
    # ------------------------------------------------------------------------
    Display-WorkerSplitPlan -W $imgWidth -H $imgHeight -FilterName $filterDesc -Radius $filterRadius -Ranks $ranks -Threads $threads

    Write-Host "Starting execution with real-time transfer tracking...`n" -ForegroundColor Yellow

    # Output file path
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($origPath)
    $outBmp = "$WorkspaceDir\${baseName}_${filter}_${mode}.bmp"
    if (Test-Path $outBmp) { Remove-Item $outBmp -Force }

    # Launch engine
    $cmd = ""
    if ($mode -eq "sequential") {
        $cmd = ".\himg.exe `"$inputToEngine`" $filter --mode sequential --out `"$outBmp`" --visualize"
        cmd /c $cmd
    } elseif ($mode -eq "openmp") {
        $cmd = ".\himg.exe `"$inputToEngine`" $filter --mode openmp --threads $threads --out `"$outBmp`" --visualize"
        cmd /c $cmd
    } elseif ($mode -eq "mpi") {
        $cmd = "mpiexec -n $ranks himg.exe `"$inputToEngine`" $filter --mode mpi --out `"$outBmp`" --visualize"
        cmd /c $cmd
    } elseif ($mode -eq "hybrid") {
        $cmd = "mpiexec -n $ranks himg.exe `"$inputToEngine`" $filter --mode hybrid --threads $threads --out `"$outBmp`" --visualize"
        cmd /c $cmd
    }

    # ------------------------------------------------------------------------
    # STEP 5: Results & Export
    # ------------------------------------------------------------------------
    if (Test-Path $outBmp) {
        $finalOutPath = $outBmp
        $origExt = [System.IO.Path]::GetExtension($origPath).ToLower()
        if ($origExt -ne ".bmp" -and $origExt -ne ".ppm" -and -not [string]::IsNullOrWhiteSpace($origExt)) {
            $convertedFinal = "$WorkspaceDir\${baseName}_${filter}_${mode}$origExt"
            if (Convert-FromBmp -BmpPath $outBmp -DestPath $convertedFinal -Format $origExt) {
                $finalOutPath = $convertedFinal
            }
        }

        Write-Host "`n================================================================================" -ForegroundColor Green
        Write-Host "  SUCCESS: Filtered image saved to:" -ForegroundColor Green
        Write-Host "  $finalOutPath" -ForegroundColor White
        Write-Host "================================================================================" -ForegroundColor Green

        $openView = Read-Host "`nOpen the filtered image now? [Y/n]"
        if ($openView.Trim().ToLower() -ne "n") {
            try {
                Invoke-Item $finalOutPath
            } catch {
                Write-Host "Could not open image viewer automatically." -ForegroundColor DarkGray
            }
        }
    } else {
        Write-Host "`n[!] Output file was not generated." -ForegroundColor Yellow
        Write-Host "If Windows Smart App Control blocked himg.exe, disable SAC in Windows Security settings." -ForegroundColor DarkGray
    }

    # Clean up temporary conversion BMP if used
    if ($isTempGenerated -and (Test-Path "$WorkspaceDir\input_converted.bmp")) {
        Remove-Item "$WorkspaceDir\input_converted.bmp" -Force -ErrorAction SilentlyContinue
    }

    Write-Host ""
    $again = Read-Host "Would you like to process another image or filter? [Y/n]"
} while ($again.Trim().ToLower() -ne "n")

Write-Host "`nThank you for using Hybrid Distributed Image Processing Engine!" -ForegroundColor Cyan
