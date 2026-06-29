param(
    [int]$DemoFrames = 260,
    [switch]$SkipYolo
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

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "== $Name =="
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name"
        exit $LASTEXITCODE
    }
    Write-Host "OK: $Name"
}

$PythonExe = Find-Python
$SmokeRoot = Join-Path $env:TEMP "crosscam_reid_smoke"
$ExportRoot = Join-Path $SmokeRoot "exported"
$DatasetRoot = Join-Path $SmokeRoot "dataset"
$RunLogDir = Join-Path $SmokeRoot "runs"

Remove-Item -Recurse -Force $SmokeRoot -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $ExportRoot "images"), (Join-Path $ExportRoot "labels") | Out-Null

Invoke-Step "Python compile" {
    & $PythonExe -m py_compile `
        src\crosscam_mvp.py `
        src\capture_dataset.py `
        src\prepare_yolo_dataset.py `
        src\validate_yolo_dataset.py `
        src\analyze_run_log.py `
        src\analyze_yolo_eval.py `
        src\realsense_depth_probe.py
}

Invoke-Step "Prepare sample YOLO export" {
    $env:CROSSCAM_SMOKE_EXPORT = $ExportRoot
    @'
import cv2
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["CROSSCAM_SMOKE_EXPORT"])
for i in range(1, 6):
    image = np.full((64, 128, 3), 20 + i * 30, dtype=np.uint8)
    cv2.rectangle(image, (24, 22), (104, 42), (0, 180, 220), -1)
    image_path = root / "images" / f"pipe_{i:03d}.jpg"
    label_path = root / "labels" / f"pipe_{i:03d}.txt"
    cv2.imwrite(str(image_path), image)
    label_path.write_text("0 0.500000 0.500000 0.625000 0.312500\n", encoding="utf-8")
'@ | & $PythonExe -
}

Invoke-Step "Prepare train/val dataset" {
    & $PythonExe src\prepare_yolo_dataset.py `
        --source-images (Join-Path $ExportRoot "images") `
        --source-labels (Join-Path $ExportRoot "labels") `
        --dataset-root $DatasetRoot `
        --clean
}

Invoke-Step "Validate prepared dataset" {
    & $PythonExe src\validate_yolo_dataset.py --dataset-root $DatasetRoot
}

Invoke-Step "Analyze sample YOLO predictions" {
    $PredLabelRoot = Join-Path $SmokeRoot "pred_labels"
    New-Item -ItemType Directory -Force -Path $PredLabelRoot | Out-Null
    Copy-Item -Path (Join-Path $DatasetRoot "labels\val\*.txt") -Destination $PredLabelRoot -Force
    & $PythonExe src\analyze_yolo_eval.py `
        --dataset-root $DatasetRoot `
        --split val `
        --pred-labels $PredLabelRoot `
        --report-csv (Join-Path $SmokeRoot "yolo_eval_analysis.csv") `
        --require-predictions `
        --min-precision 1.0 `
        --min-recall 1.0 `
        --max-false-positives 0 `
        --max-false-negatives 0
}

Invoke-Step "Synthetic cross-camera handoff demo" {
    & $PythonExe src\crosscam_mvp.py `
        --demo `
        --auto-register-first `
        --headless `
        --frames $DemoFrames `
        --require-match `
        --log-dir $RunLogDir
}

Invoke-Step "Verify target sample index" {
    $TargetSampleCsv = Join-Path $RunLogDir "targets\target_samples.csv"
    if (-not (Test-Path $TargetSampleCsv)) {
        Write-Host "Missing target sample index: $TargetSampleCsv"
        exit 2
    }
    $SampleLines = Get-Content $TargetSampleCsv
    if ($SampleLines.Count -lt 2) {
        Write-Host "Target sample index has no sample rows: $TargetSampleCsv"
        exit 2
    }
}

Invoke-Step "Analyze handoff log" {
    & $PythonExe src\analyze_run_log.py `
        --log-dir $RunLogDir `
        --require-handoff `
        --min-target-matches 2 `
        --min-cross-camera-ids 1 `
        --max-unique-ids 1
}

Invoke-Step "Training script check branch" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_pipe_yolo.ps1 -CheckOnly -SkipInstall -SkipValidate
}

Invoke-Step "Evaluation command generation" {
    & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\evaluate_pipe_yolo.ps1 -PrintOnly -SkipInstall
}

if (-not $SkipYolo) {
    Invoke-Step "YOLO detector smoke path" {
        & powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_crosscam.ps1 -Demo -PipeMode -Headless -Frames 1 -LogDir $RunLogDir -SkipInstall
    }
}

Write-Host ""
Write-Host "Smoke test passed."
Write-Host "Temporary files: $SmokeRoot"
