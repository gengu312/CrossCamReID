param(
    [string]$Model = "runs_yolo\pipe_yolov8n\weights\best.pt",
    [string]$Data = "datasets/pipe_yolo/data.yaml",
    [string]$DatasetRoot = "datasets/pipe_yolo",
    [ValidateSet("train", "val")]
    [string]$Split = "val",
    [string]$Source = "datasets/pipe_yolo/images/val",
    [double]$Conf = 0.25,
    [int]$Imgsz = 640,
    [string]$Device = "cpu",
    [string]$Project = "runs_yolo_eval",
    [string]$Name = "pipe_yolov8n_eval",
    [switch]$ValOnly,
    [switch]$PredictOnly,
    [switch]$SkipAnalyze,
    [switch]$CollectIssues,
    [string]$IssueOutputDir = "",
    [int]$IssueMaxCount = 80,
    [switch]$IssuePreview,
    [switch]$IssueClean,
    [switch]$SkipInstall,
    [switch]$CheckOnly,
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

$ProjectForYolo = $Project
if (-not [System.IO.Path]::IsPathRooted($ProjectForYolo)) {
    $ProjectForYolo = Join-Path $RepoRoot $ProjectForYolo
}

$ValArgs = @(
    "detect",
    "val",
    "model=$Model",
    "data=$Data",
    "imgsz=$Imgsz",
    "device=$Device",
    "project=$ProjectForYolo",
    "name=$Name"
)

$PredictName = "$($Name)_predict"
$PredictArgs = @(
    "detect",
    "predict",
    "model=$Model",
    "source=$Source",
    "conf=$Conf",
    "imgsz=$Imgsz",
    "device=$Device",
    "project=$ProjectForYolo",
    "name=$PredictName",
    "exist_ok=True",
    "save=True",
    "save_txt=True",
    "save_conf=True"
)

$PredictRoot = Join-Path $Project $PredictName
$PredictionLabels = Join-Path $PredictRoot "labels"
$AnalysisCsv = Join-Path $PredictRoot "analysis.csv"
$AnalysisJson = Join-Path $PredictRoot "analysis_summary.json"
$AnalysisMd = Join-Path $PredictRoot "analysis_summary.md"
if ($IssueOutputDir -eq "") {
    $IssueOutputDir = Join-Path $PredictRoot "issue_samples"
}

$AnalyzeArgs = @(
    "src\analyze_yolo_eval.py",
    "--dataset-root", $DatasetRoot,
    "--split", $Split,
    "--pred-labels", $PredictionLabels,
    "--report-csv", $AnalysisCsv,
    "--summary-json", $AnalysisJson,
    "--summary-md", $AnalysisMd
)

$IssueArgs = @(
    "src\collect_yolo_issues.py",
    "--analysis-csv", $AnalysisCsv,
    "--dataset-root", $DatasetRoot,
    "--split", $Split,
    "--pred-labels", $PredictionLabels,
    "--output-dir", $IssueOutputDir,
    "--max-count", "$IssueMaxCount"
)
if ($IssuePreview) {
    $IssueArgs += "--preview"
}
if ($IssueClean) {
    $IssueArgs += "--clean"
}

$RunVal = -not $PredictOnly
$RunPredict = -not $ValOnly

if ($CollectIssues -and ($ValOnly -or $SkipAnalyze -or $CheckOnly)) {
    Write-Host "CollectIssues requires prediction labels and analysis.csv."
    Write-Host "Remove -ValOnly/-SkipAnalyze/-CheckOnly, or run .\collect_yolo_issues.bat after analysis.csv exists."
    exit 2
}

if ($PrintOnly) {
    if ($RunVal) {
        Write-Host ($YoloExe + " " + ($ValArgs -join " "))
    }
    if ($RunPredict) {
        Write-Host ($YoloExe + " " + ($PredictArgs -join " "))
        if (-not $SkipAnalyze) {
            Write-Host ($PythonExe + " " + ($AnalyzeArgs -join " "))
            if ($CollectIssues) {
                Write-Host ($PythonExe + " " + ($IssueArgs -join " "))
            }
        }
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

if ($CheckOnly) {
    $Failures = @()
    if ($RunVal -and -not (Test-Path -LiteralPath $Data)) {
        $Failures += "YOLO data.yaml does not exist: $Data"
    }
    if ($RunPredict -and -not $SkipAnalyze -and -not (Test-Path -LiteralPath $DatasetRoot)) {
        $Failures += "Dataset root does not exist: $DatasetRoot"
    }
    if ($Conf -lt 0.0 -or $Conf -gt 1.0) {
        $Failures += "Conf must be between 0 and 1: $Conf"
    }
    if ($Imgsz -le 0) {
        $Failures += "Imgsz must be greater than 0: $Imgsz"
    }
    if ($Failures.Count -gt 0) {
        Write-Host "YOLO evaluation check failed:"
        $Failures | ForEach-Object { Write-Host "  - $_" }
        exit 2
    }

    Write-Host "YOLO evaluation check:"
    Write-Host "  model=$Model"
    Write-Host "  data=$Data"
    Write-Host "  dataset_root=$DatasetRoot"
    Write-Host "  split=$Split"
    Write-Host "  source=$Source"
    Write-Host "  yolo=$YoloExe"
    Write-Host "  prediction_labels=$PredictionLabels"
    Write-Host "  analysis_csv=$AnalysisCsv"
    Write-Host "CheckOnly: YOLO evaluation inputs are ready. Inference was not started."
    exit 0
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
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    if (-not $SkipAnalyze) {
        Write-Host "Analyzing prediction labels..."
        Write-Host ($PythonExe + " " + ($AnalyzeArgs -join " "))
        & $PythonExe @AnalyzeArgs
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        if ($CollectIssues) {
            Write-Host "Collecting YOLO issue samples..."
            Write-Host ($PythonExe + " " + ($IssueArgs -join " "))
            & $PythonExe @IssueArgs
            exit $LASTEXITCODE
        }
    }
}

exit 0
