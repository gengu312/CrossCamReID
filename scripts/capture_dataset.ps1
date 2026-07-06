param(
    [string]$CamA = "auto",
    [string]$CamB = "auto",
    [string]$CameraScanOrder = "1,3,2,0,4,5",
    [int]$ProbeMax = 10,
    [string]$Backend = "dshow",
    [ValidateSet("single", "stack", "hand_move", "negative")]
    [string]$Scenario = "stack",
    [string]$OutputRoot = "dataset_raw",
    [string]$Prefix = "",
    [double]$AutoInterval = 0,
    [int]$Limit = 0,
    [ValidateSet("AB", "BA")]
    [string]$ViewOrder = "AB",
    [switch]$FlipA,
    [switch]$FlipB,
    [switch]$FlipBoth,
    [switch]$PrintOnly
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

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

function Write-Utf8Host {
    param([string]$Base64Text)
    & $PythonExe -c "import base64; print(base64.b64decode('$Base64Text').decode('utf-8'))"
}

$AppArgs = @(
    "src\capture_dataset.py",
    "--cam-a", "$CamA",
    "--cam-b", "$CamB",
    "--camera-scan-order", $CameraScanOrder,
    "--probe-max", "$ProbeMax",
    "--backend", $Backend,
    "--scenario", $Scenario,
    "--view-order", $ViewOrder,
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

if ($FlipA) {
    $AppArgs += "--flip-a"
}

if ($FlipB) {
    $AppArgs += "--flip-b"
}

if ($FlipBoth) {
    $AppArgs += "--flip-both"
}

if ($PrintOnly) {
    $AppArgs += "--print-only"
}

Write-Utf8Host "5q2j5Zyo5ZCv5Yqo6K6t57uD54Wn54mH6YeH6ZuGLi4u"
Write-Host "$PythonExe $($AppArgs -join ' ')"
Write-Host ""

& $PythonExe @AppArgs
exit $LASTEXITCODE
