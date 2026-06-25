param(
    [string]$SourceImages,
    [string]$SourceLabels,
    [string]$DatasetRoot = "datasets/pipe_yolo",
    [double]$ValRatio = 0.2,
    [int]$Seed = 42,
    [switch]$Clean,
    [switch]$AllowNegative
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

if (-not $SourceImages -or -not $SourceLabels) {
    Write-Host "Usage: .\prepare_yolo_dataset.bat -SourceImages <images_dir> -SourceLabels <labels_dir> [-Clean]"
    exit 2
}

$PythonExe = Find-Python
$AppArgs = @(
    "src\prepare_yolo_dataset.py",
    "--source-images", $SourceImages,
    "--source-labels", $SourceLabels,
    "--dataset-root", $DatasetRoot,
    "--val-ratio", "$ValRatio",
    "--seed", "$Seed"
)

if ($Clean) {
    $AppArgs += "--clean"
}

if ($AllowNegative) {
    $AppArgs += "--allow-negative"
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
