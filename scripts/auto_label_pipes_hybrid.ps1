param(
    [string]$Images = "dataset_raw\to_label_next\images",
    [string]$Labels = "dataset_raw\to_label_next\labels",
    [string]$Previews = "dataset_raw\to_label_next\previews",
    [string]$Report = "dataset_raw\to_label_next\hybrid_label_report.csv",
    [int]$ExpectedCount = 7,
    [switch]$Preview,
    [switch]$Overwrite
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

$AppArgs = @(
    "src\auto_label_pipes_hybrid.py",
    "--images", $Images,
    "--labels", $Labels,
    "--previews", $Previews,
    "--report", $Report,
    "--expected-count", "$ExpectedCount"
)
if ($Preview) {
    $AppArgs += "--preview"
}
if ($Overwrite) {
    $AppArgs += "--overwrite"
}

& (Find-Python) @AppArgs
exit $LASTEXITCODE
