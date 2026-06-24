param(
    [int]$CamA = 0,
    [int]$CamB = 2,
    [string]$Backend = "dshow",
    [string]$RoiA = "80,80,480,220",
    [string]$RoiB = "80,80,480,220",
    [int]$WarmupFrames = 30,
    [int]$MinArea = 900,
    [double]$CrossThreshold = 0.65,
    [string]$TargetMode = "pencil",
    [bool]$SingleObject = $true,
    [double]$MaxAreaRatio = 0.45,
    [double]$MaxShapeRatio = 0.75,
    [int]$MinLongSide = 45,
    [int]$MaxShortSide = 180,
    [double]$TargetThreshold = 0.58,
    [double]$TargetUpdateAlpha = 0.04,
    [string]$LogDir = "runs",
    [switch]$Headless,
    [int]$Frames = 0,
    [switch]$Demo,
    [switch]$Probe,
    [switch]$AutoRegisterFirst,
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
    $AppArgs += @(
        "--demo",
        "--target-threshold", "$TargetThreshold",
        "--target-update-alpha", "$TargetUpdateAlpha",
        "--log-dir", $LogDir
    )
} else {
    $AppArgs += @(
        "--cam-a", "$CamA",
        "--cam-b", "$CamB",
        "--backend", $Backend,
        "--roi-a", $RoiA,
        "--roi-b", $RoiB,
        "--warmup-frames", "$WarmupFrames",
        "--min-area", "$MinArea",
        "--target-mode", $TargetMode,
        "--max-area-ratio", "$MaxAreaRatio",
        "--max-shape-ratio", "$MaxShapeRatio",
        "--min-long-side", "$MinLongSide",
        "--max-short-side", "$MaxShortSide",
        "--cross-threshold", "$CrossThreshold",
        "--target-threshold", "$TargetThreshold",
        "--target-update-alpha", "$TargetUpdateAlpha",
        "--log-dir", $LogDir
    )

    if ($SingleObject) {
        $AppArgs += "--single-object"
    }

}

if (-not $Probe -and $AutoRegisterFirst) {
    $AppArgs += "--auto-register-first"
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
