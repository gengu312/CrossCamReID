param(
    [string]$Data = "datasets/pipe_yolo/data.yaml",
    [string]$DatasetRoot = "datasets/pipe_yolo",
    [string]$Model = "yolov8n.pt",
    [int]$Epochs = 80,
    [int]$Imgsz = 640,
    [int]$Batch = 8,
    [string]$Device = "cpu",
    [string]$Project = "runs_yolo",
    [string]$Name = "pipe_yolov8n",
    [switch]$AllowMissingLabels,
    [switch]$SkipInstall,
    [switch]$SkipValidate,
    [switch]$CheckOnly
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

function Test-TrainingArgs {
    if ($Epochs -le 0) {
        Write-Host "Epochs must be greater than 0."
        exit 2
    }
    if ($Imgsz -le 0) {
        Write-Host "Imgsz must be greater than 0."
        exit 2
    }
    if ($Batch -le 0) {
        Write-Host "Batch must be greater than 0."
        exit 2
    }
    if (-not (Test-Path $Data)) {
        Write-Host "YOLO data.yaml was not found: $Data"
        exit 2
    }
}

$PythonExe = Find-Python
Test-TrainingArgs

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

if (-not $SkipValidate) {
    $ValidateArgs = @(
        "src\validate_yolo_dataset.py",
        "--dataset-root", $DatasetRoot,
        "--class-count", "1"
    )
    if ($AllowMissingLabels) {
        $ValidateArgs += "--allow-missing-labels"
    }
    & $PythonExe @ValidateArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($CheckOnly) {
    if ($SkipValidate) {
        Write-Host "CheckOnly: parameters checked. Dataset validation was skipped. Training is not started."
    } else {
        Write-Host "CheckOnly: dataset validation passed. Training is not started."
    }
    exit 0
}

$YoloExe = Find-Yolo -PythonExe $PythonExe
if (-not $YoloExe) {
    Write-Host "yolo.exe was not found. Install it with: python -m pip install -r requirements-yolo.txt"
    exit 2
}

$TrainArgs = @(
    "detect",
    "train",
    "model=$Model",
    "data=$Data",
    "epochs=$Epochs",
    "imgsz=$Imgsz",
    "batch=$Batch",
    "device=$Device",
    "project=$Project",
    "name=$Name"
)

Write-Host "Starting YOLO training..."
$CommandLine = $YoloExe + " " + ($TrainArgs -join " ")
Write-Host $CommandLine
Write-Host ""

& $YoloExe @TrainArgs
exit $LASTEXITCODE
