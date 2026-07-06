param(
    [string]$SamplesCsv = "runs\targets\target_samples.csv",
    [string]$OutputDir = "runs\target_sample_review",
    [int]$MaxCount = 120,
    [ValidateSet("all", "register", "match")]
    [string]$Source = "all",
    [double]$MinSimilarity = -1,
    [int]$MinCount = -1,
    [switch]$Preview,
    [int]$PreviewCols = 4,
    [switch]$Clean,
    [switch]$Strict
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
    "src\collect_target_samples.py",
    "--samples-csv", $SamplesCsv,
    "--output-dir", $OutputDir,
    "--max-count", "$MaxCount",
    "--source", $Source,
    "--preview-cols", "$PreviewCols"
)

if ($MinSimilarity -ge 0) {
    $AppArgs += @("--min-similarity", "$MinSimilarity")
}
if ($MinCount -ge 0) {
    $AppArgs += @("--min-count", "$MinCount")
}
if ($Preview) {
    $AppArgs += "--preview"
}
if ($Clean) {
    $AppArgs += "--clean"
}
if ($Strict) {
    $AppArgs += "--strict"
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
