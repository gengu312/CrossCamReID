param(
    [string]$Model = "runs_yolo\pipe_yolov8n\weights\best.pt",
    [string]$Data = "datasets/pipe_yolo/data.yaml",
    [string]$Source = "datasets/pipe_yolo/images/val",
    [double]$Conf = 0.25,
    [int]$Imgsz = 640,
    [string]$Device = "cpu",
    [string]$Project = "runs_yolo_eval",
    [string]$Name = "pipe_yolov8n_eval",
    [switch]$ValOnly,
    [switch]$PredictOnly,
    [switch]$SkipInstall,
    [switch]$PrintOnly
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

function Find-Yolo {
    param([string]$PythonExe)

    $pythonDir = Split-Path -Parent $PythonExe
    $candidate = Join-Path $pythonDir "Scripts\yolo.exe"
    if (Test-Path $candidate) {
        return $candidate
    }

    $command = Get-Command yolo -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    return $null
}

$PythonExe = Find-Python

if (-not $SkipInstall) {
    & $PythonExe -c "import ultralytics" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing YOLO dependencies..."
        & $PythonExe -m pip install -r requirements-yolo.txt
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}

$YoloExe = Find-Yolo -PythonExe $PythonExe
if (-not $YoloExe) {
    Write-Host "yolo.exe was not found. Install it with: python -m pip install -r requirements-yolo.txt"
    exit 2
}

$ValArgs = @(
    "detect",
    "val",
    "model=$Model",
    "data=$Data",
    "imgsz=$Imgsz",
    "device=$Device",
    "project=$Project",
    "name=$Name"
)

$PredictArgs = @(
    "detect",
    "predict",
    "model=$Model",
    "source=$Source",
    "conf=$Conf",
    "imgsz=$Imgsz",
    "device=$Device",
    "project=$Project",
    "name=$($Name)_predict",
    "save=True"
)

$RunVal = -not $PredictOnly
$RunPredict = -not $ValOnly

if ($PrintOnly) {
    if ($RunVal) {
        Write-Host ($YoloExe + " " + ($ValArgs -join " "))
    }
    if ($RunPredict) {
        Write-Host ($YoloExe + " " + ($PredictArgs -join " "))
    }
    exit 0
}

if (-not (Test-Path $Model)) {
    Write-Host "Model file was not found: $Model"
    Write-Host "Train first with: .\train_pipe_yolo.bat"
    exit 2
}

if ($RunPredict -and -not (Test-Path $Source)) {
    Write-Host "Prediction source was not found: $Source"
    exit 2
}

if ($RunVal) {
    Write-Host "Running YOLO validation..."
    Write-Host ($YoloExe + " " + ($ValArgs -join " "))
    & $YoloExe @ValArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($RunPredict) {
    Write-Host "Saving prediction previews..."
    Write-Host ($YoloExe + " " + ($PredictArgs -join " "))
    & $YoloExe @PredictArgs
    exit $LASTEXITCODE
}

exit 0
