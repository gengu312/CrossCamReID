param(
    [string]$AnalysisCsv = "runs_yolo_eval\pipe_yolov8n_eval_predict\analysis.csv",
    [string]$DatasetRoot = "datasets\pipe_yolo",
    [ValidateSet("train", "val")]
    [string]$Split = "val",
    [string]$ImagesDir = "",
    [string]$LabelsDir = "",
    [string]$PredLabels = "runs_yolo_eval\pipe_yolov8n_eval_predict\labels",
    [string]$OutputDir = "runs_yolo_eval\pipe_yolov8n_eval_predict\issue_samples",
    [int]$MaxCount = 80,
    [switch]$Preview,
    [switch]$Clean
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
    "src\collect_yolo_issues.py",
    "--analysis-csv", $AnalysisCsv,
    "--dataset-root", $DatasetRoot,
    "--split", $Split,
    "--pred-labels", $PredLabels,
    "--output-dir", $OutputDir,
    "--max-count", "$MaxCount"
)

if ($ImagesDir -ne "") {
    $AppArgs += @("--images-dir", $ImagesDir)
}
if ($LabelsDir -ne "") {
    $AppArgs += @("--labels-dir", $LabelsDir)
}
if ($Preview) {
    $AppArgs += "--preview"
}
if ($Clean) {
    $AppArgs += "--clean"
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
