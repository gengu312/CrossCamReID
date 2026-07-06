param(
    [string]$Images = "dataset_raw\to_label_20260626\images",
    [string]$Labels = "dataset_raw\to_label_20260626\labels",
    [string]$Previews = "dataset_raw\to_label_20260626\previews",
    [double]$MinAreaRatio = 0.0008,
    [string]$NegativePrefix = "negative",
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

$PythonExe = Find-Python

$AppArgs = @(
    "src\auto_label_pipes.py",
    "--images", $Images,
    "--labels", $Labels,
    "--previews", $Previews,
    "--min-area-ratio", "$MinAreaRatio",
    "--negative-prefix", $NegativePrefix
)

if ($Preview) {
    $AppArgs += "--preview"
}

if ($Overwrite) {
    $AppArgs += "--overwrite"
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
