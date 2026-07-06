param(
    [string]$YoloCsv = "runs_yolo_eval\pipe_yolov8n_eval_predict\analysis.csv",
    [string]$RfDetrCsv = "runs_rfdetr_eval\pipe_rfdetr_nano_eval\analysis.csv",
    [string]$OutputCsv = "runs_detector_compare\yolo_vs_rfdetr.csv",
    [string]$SummaryJson = "runs_detector_compare\yolo_vs_rfdetr_summary.json",
    [string]$SummaryMd = "runs_detector_compare\yolo_vs_rfdetr_summary.md",
    [int]$MaxExamples = 12,
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
$AppArgs = @(
    "src\compare_detector_eval.py",
    "--left-csv", $YoloCsv,
    "--right-csv", $RfDetrCsv,
    "--left-name", "YOLO",
    "--right-name", "RF-DETR",
    "--output-csv", $OutputCsv,
    "--max-examples", "$MaxExamples"
)
if ($SummaryJson) {
    $AppArgs += @("--summary-json", $SummaryJson)
}
if ($SummaryMd) {
    $AppArgs += @("--summary-md", $SummaryMd)
}

if ($PrintOnly) {
    Write-Host ($PythonExe + " " + ($AppArgs -join " "))
    exit 0
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
