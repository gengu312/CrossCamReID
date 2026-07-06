param(
    [string]$YoloRoot = "datasets/pipe_yolo",
    [string]$OutputRoot = "datasets/pipe_rfdetr",
    [string]$ClassNames = "pipe",
    [int]$CategoryIdOffset = 1,
    [switch]$Clean,
    [switch]$TestFromVal,
    [switch]$AllowMissingLabels
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
$AppArgs = @(
    "src\export_yolo_to_coco.py",
    "--yolo-root", $YoloRoot,
    "--output-root", $OutputRoot,
    "--class-names", $ClassNames,
    "--category-id-offset", "$CategoryIdOffset"
)

if ($Clean) {
    $AppArgs += "--clean"
}

if ($TestFromVal) {
    $AppArgs += "--test-from-val"
}

if ($AllowMissingLabels) {
    $AppArgs += "--allow-missing-labels"
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
