param(
    [string]$YoloModel = "",
    [ValidateSet("nano", "small", "base", "medium", "large", "xlarge", "2xlarge")]
    [string]$RfDetrSize = "nano",
    [string]$RfDetrWeights = "",
    [switch]$Json
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    $PythonExe = (Get-Command py -ErrorAction SilentlyContinue).Source
}
if (-not $PythonExe) {
    throw "Python was not found. Install Python and try again."
}

$AppArgs = @(
    "src\detector_backend_status.py",
    "--repo-root", $RepoRoot,
    "--rfdetr-size", $RfDetrSize
)
if ($YoloModel -ne "") {
    $AppArgs += @("--yolo-model", $YoloModel)
}
if ($RfDetrWeights -ne "") {
    $AppArgs += @("--rfdetr-weights", $RfDetrWeights)
}
if ($Json) {
    $AppArgs += "--json"
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
