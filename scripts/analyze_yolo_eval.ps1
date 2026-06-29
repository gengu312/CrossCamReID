param(
    [string]$DatasetRoot = "datasets/pipe_yolo",
    [ValidateSet("train", "val")]
    [string]$Split = "val",
    [string]$PredLabels = "runs_yolo_eval\pipe_yolov8n_eval_predict\labels",
    [string]$ReportCsv = "runs_yolo_eval\pipe_yolov8n_eval_predict\analysis.csv",
    [double]$IouThreshold = 0.50,
    [double]$ConfThreshold = 0.0,
    [double]$OversizeRatio = 1.8,
    [int]$MaxExamples = 12,
    [double]$MinPrecision = -1,
    [double]$MinRecall = -1,
    [int]$MaxFalsePositives = -1,
    [int]$MaxFalseNegatives = -1,
    [switch]$RequirePredictions
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

$PythonExe = Find-Python
$AppArgs = @(
    "src\analyze_yolo_eval.py",
    "--dataset-root", $DatasetRoot,
    "--split", $Split,
    "--pred-labels", $PredLabels,
    "--report-csv", $ReportCsv,
    "--iou-threshold", "$IouThreshold",
    "--conf-threshold", "$ConfThreshold",
    "--oversize-ratio", "$OversizeRatio",
    "--max-examples", "$MaxExamples"
)

if ($MinPrecision -ge 0) {
    $AppArgs += @("--min-precision", "$MinPrecision")
}
if ($MinRecall -ge 0) {
    $AppArgs += @("--min-recall", "$MinRecall")
}
if ($MaxFalsePositives -ge 0) {
    $AppArgs += @("--max-false-positives", "$MaxFalsePositives")
}
if ($MaxFalseNegatives -ge 0) {
    $AppArgs += @("--max-false-negatives", "$MaxFalseNegatives")
}
if ($RequirePredictions) {
    $AppArgs += "--require-predictions"
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
