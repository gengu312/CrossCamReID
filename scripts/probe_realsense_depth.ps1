param(
    [switch]$ListOnly,
    [int]$Frames = 30,
    [int]$Width = 640,
    [int]$Height = 480,
    [int]$Fps = 30
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
$AppArgs = @(
    "src\realsense_depth_probe.py",
    "--frames", "$Frames",
    "--width", "$Width",
    "--height", "$Height",
    "--fps", "$Fps"
)

if ($ListOnly) {
    $AppArgs += "--list-only"
}

Write-Host "正在检查 RealSense 深度能力..."
Write-Host "$PythonExe $($AppArgs -join ' ')"
Write-Host ""

& $PythonExe @AppArgs
exit $LASTEXITCODE
