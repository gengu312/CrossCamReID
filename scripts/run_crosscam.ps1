param(
    [int]$CamA = 0,
    [int]$CamB = 2,
    [string]$Backend = "dshow",
    [string]$RoiA = "80,80,480,220",
    [string]$RoiB = "80,80,480,220",
    [int]$WarmupFrames = 45,
    [int]$MinArea = 5000,
    [double]$CrossThreshold = 0.72,
    [string]$LogDir = "runs",
    [switch]$Headless,
    [int]$Frames = 0,
    [switch]$Demo,
    [switch]$Probe,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Find-Python {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return $py.Source
    }

    throw "Python was not found. Install Python and try again."
}

$PythonExe = Find-Python

if (-not $SkipInstall) {
    & $PythonExe -c "import cv2, numpy" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing Python dependencies from requirements.txt..."
        & $PythonExe -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}

$AppArgs = @("src\crosscam_mvp.py")

if ($Probe) {
    $AppArgs += @("--probe", "--probe-max", "5", "--backend", $Backend)
} elseif ($Demo) {
    $AppArgs += @("--demo")
} else {
    $AppArgs += @(
        "--cam-a", "$CamA",
        "--cam-b", "$CamB",
        "--backend", $Backend,
        "--roi-a", $RoiA,
        "--roi-b", $RoiB,
        "--warmup-frames", "$WarmupFrames",
        "--min-area", "$MinArea",
        "--cross-threshold", "$CrossThreshold",
        "--log-dir", $LogDir
    )
}

if ($Headless) {
    $AppArgs += "--headless"
}

if ($Frames -gt 0) {
    $AppArgs += @("--frames", "$Frames")
}

Write-Host "Running CrossCamReID..."
Write-Host "$PythonExe $($AppArgs -join ' ')"
Write-Host ""

& $PythonExe @AppArgs
exit $LASTEXITCODE
