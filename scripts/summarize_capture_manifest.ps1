param(
    [string]$Manifest = "dataset_raw\capture_manifest.csv",
    [switch]$Strict,
    [switch]$FirstBatchPlan,
    [switch]$RequireFirstBatch
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
$AppArgs = @("src\summarize_capture_manifest.py", "--manifest", $Manifest)
if ($Strict) {
    $AppArgs += "--strict"
}
if ($FirstBatchPlan) {
    $AppArgs += "--first-batch-plan"
}
if ($RequireFirstBatch) {
    $AppArgs += "--require-first-batch"
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
