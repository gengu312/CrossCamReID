param(
    [int]$CamA = 0,
    [int]$CamB = 2,
    [string]$Backend = "dshow",
    [ValidateSet("single", "stack", "hand_move", "negative")]
    [string]$Scenario = "stack",
    [string]$OutputRoot = "dataset_raw",
    [string]$Prefix = "",
    [double]$AutoInterval = 0,
    [int]$Limit = 0
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
    "src\capture_dataset.py",
    "--cam-a", "$CamA",
    "--cam-b", "$CamB",
    "--backend", $Backend,
    "--scenario", $Scenario,
    "--output-root", $OutputRoot
)

if ($Prefix -ne "") {
    $AppArgs += @("--prefix", $Prefix)
}

if ($AutoInterval -gt 0) {
    $AppArgs += @("--auto-interval", "$AutoInterval")
}

if ($Limit -gt 0) {
    $AppArgs += @("--limit", "$Limit")
}

Write-Host "正在启动训练照片采集..."
Write-Host "$PythonExe $($AppArgs -join ' ')"
Write-Host ""

& $PythonExe @AppArgs
exit $LASTEXITCODE
