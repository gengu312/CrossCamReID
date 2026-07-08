param(
    [string]$SourceImages,
    [string]$SourceLabels,
    [string]$DatasetRoot = "datasets/pipe_yolo",
    [string]$ClassNames = "pipe",
    [double]$ValRatio = 0.2,
    [int]$Seed = 42,
    [switch]$Clean,
    [switch]$AllowNegative,
    [switch]$DropConfidenceColumn
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

if (-not $SourceImages -or -not $SourceLabels) {
    Write-Host "Usage: .\prepare_yolo_dataset.bat -SourceImages <images_dir> -SourceLabels <labels_dir> [-DatasetRoot <dir>] [-ClassNames pipe] [-ValRatio 0.2] [-Clean] [-AllowNegative] [-DropConfidenceColumn]"
    exit 2
}

$PythonExe = Find-Python
$AppArgs = @(
    "src\prepare_yolo_dataset.py",
    "--source-images", $SourceImages,
    "--source-labels", $SourceLabels,
    "--dataset-root", $DatasetRoot,
    "--class-names", $ClassNames,
    "--val-ratio", "$ValRatio",
    "--seed", "$Seed"
)

if ($Clean) {
    $AppArgs += "--clean"
}

if ($AllowNegative) {
    $AppArgs += "--allow-negative"
}

if ($DropConfidenceColumn) {
    $AppArgs += "--drop-confidence-column"
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
