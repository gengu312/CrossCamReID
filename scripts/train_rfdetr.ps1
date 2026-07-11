param(
    [string]$DatasetDir = "datasets/pipe_rfdetr",
    [string]$OutputDir = "runs_rfdetr/pipe_rfdetr_nano",
    [ValidateSet("nano", "small", "base", "medium", "large", "xlarge", "2xlarge")]
    [string]$ModelSize = "nano",
    [int]$Epochs = 10,
    [int]$BatchSize = 4,
    [int]$GradAccumSteps = 4,
    [double]$Lr = 0.0001,
    [int]$NumClasses = 0,
    [string]$PretrainWeights = "",
    [string]$Resume = "",
    [switch]$CheckOnly,
    [switch]$PrintOnly,
    [switch]$SkipInstall
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

if ($OutputDir -eq "runs_rfdetr/pipe_rfdetr_nano" -and $ModelSize -ne "nano") {
    $OutputDir = "runs_rfdetr/pipe_rfdetr_$ModelSize"
}

if (-not $SkipInstall -and -not $CheckOnly -and -not $PrintOnly) {
    & $PythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('rfdetr') else 1)" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "正在安装 RF-DETR 依赖 rfdetr..."
        & $PythonExe -m pip install -r requirements-rfdetr.txt
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}

$AppArgs = @(
    "src\train_rfdetr.py",
    "--dataset-dir", $DatasetDir,
    "--output-dir", $OutputDir,
    "--model-size", $ModelSize,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--grad-accum-steps", "$GradAccumSteps",
    "--lr", "$Lr",
    "--num-classes", "$NumClasses"
)

if ($PretrainWeights -ne "") {
    $AppArgs += @("--pretrain-weights", $PretrainWeights)
}

if ($Resume -ne "") {
    $AppArgs += @("--resume", $Resume)
}

if ($CheckOnly) {
    $AppArgs += "--check-only"
}

if ($PrintOnly) {
    $AppArgs += "--print-only"
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
