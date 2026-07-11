param(
    [string]$DatasetRoot = "datasets/pipe_yolo",
    [ValidateSet("train", "val")]
    [string]$Split = "val",
    [string]$Source = "",
    [ValidateSet("nano", "small", "base", "medium", "large", "xlarge", "2xlarge")]
    [string]$ModelSize = "nano",
    [string]$Weights = "",
    [int]$NumClasses = 0,
    [double]$Conf = 0.35,
    [string]$Classes = "",
    [ValidateSet("auto", "zero", "category")]
    [string]$ClassIdMode = "auto",
    [int]$CategoryIdOffset = 1,
    [string]$Project = "runs_rfdetr_eval",
    [string]$Name = "pipe_rfdetr_nano_eval",
    [switch]$Optimize,
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

function Resolve-RfDetrCheckpoint {
    param([string]$ModelSize)

    $OutputDir = "runs_rfdetr\pipe_rfdetr_$ModelSize"
    if (-not (Test-Path -LiteralPath $OutputDir)) {
        return ""
    }

    $CandidateNames = @(
        "checkpoint_best_ema.pth",
        "checkpoint_best_total.pth",
        "checkpoint_best_regular.pth",
        "checkpoint_best.pth",
        "checkpoint.pth"
    )
    foreach ($Name in $CandidateNames) {
        $Candidate = Join-Path $OutputDir $Name
        if (Test-Path -LiteralPath $Candidate) {
            return $Candidate
        }
    }

    $BestMatch = Get-ChildItem -LiteralPath $OutputDir -Filter "checkpoint_best*.pth" -File -ErrorAction SilentlyContinue |
        Sort-Object Name |
        Select-Object -First 1
    if ($BestMatch) {
        return $BestMatch.FullName
    }

    $LatestMatch = Get-ChildItem -LiteralPath $OutputDir -Filter "*.pth" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($LatestMatch) {
        return $LatestMatch.FullName
    }
    return ""
}

$PythonExe = Find-Python

if ($Name -eq "pipe_rfdetr_nano_eval" -and $ModelSize -ne "nano") {
    $Name = "pipe_rfdetr_${ModelSize}_eval"
}

if ($Weights -eq "") {
    $DefaultRfDetrWeights = Resolve-RfDetrCheckpoint $ModelSize
    if ($DefaultRfDetrWeights -ne "") {
        $Weights = $DefaultRfDetrWeights
    }
}

$OutputRoot = Join-Path $Project $Name
$OutputLabels = Join-Path $OutputRoot "labels"
$ReportCsv = Join-Path $OutputRoot "analysis.csv"
$ReportJson = Join-Path $OutputRoot "analysis_summary.json"
$ReportMd = Join-Path $OutputRoot "analysis_summary.md"
if ($IssueOutputDir -eq "") {
    $IssueOutputDir = Join-Path $OutputRoot "issue_samples"
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

$PredictArgs = @(
    "src\evaluate_rfdetr.py",
    "--dataset-root", $DatasetRoot,
    "--split", $Split,
    "--output-labels", $OutputLabels,
    "--model-size", $ModelSize,
    "--num-classes", "$NumClasses",
    "--conf", "$Conf",
    "--class-id-mode", $ClassIdMode,
    "--category-id-offset", "$CategoryIdOffset",
    "--clean"
)

if ($Source -ne "") {
    $PredictArgs += @("--source", $Source)
}

if ($Weights -ne "") {
    $PredictArgs += @("--weights", $Weights)
}

if ($Classes -ne "") {
    $PredictArgs += @("--classes", $Classes)
}

if ($Optimize) {
    $PredictArgs += "--optimize"
}

if ($CheckOnly) {
    $PredictArgs += "--check-only"
}

if ($PrintOnly) {
    $PredictArgs += "--print-only"
}

$AnalyzeArgs = @(
    "src\analyze_yolo_eval.py",
    "--dataset-root", $DatasetRoot,
    "--split", $Split,
    "--pred-labels", $OutputLabels,
    "--report-csv", $ReportCsv,
    "--summary-json", $ReportJson,
    "--summary-md", $ReportMd,
    "--require-predictions"
)

$IssueArgs = @(
    "src\collect_yolo_issues.py",
    "--analysis-csv", $ReportCsv,
    "--dataset-root", $DatasetRoot,
    "--split", $Split,
    "--pred-labels", $OutputLabels,
    "--output-dir", $IssueOutputDir,
    "--max-count", "$IssueMaxCount"
)
if ($IssuePreview) {
    $IssueArgs += "--preview"
}
if ($IssueClean) {
    $IssueArgs += "--clean"
}

if ($CollectIssues -and ($SkipAnalyze -or $CheckOnly)) {
    Write-Host "CollectIssues requires prediction labels and analysis.csv."
    Write-Host "Remove -SkipAnalyze/-CheckOnly, or run .\collect_yolo_issues.bat after analysis.csv exists."
    exit 2
}

if ($PrintOnly) {
    Write-Host ($PythonExe + " " + ($PredictArgs -join " "))
    if (-not $SkipAnalyze) {
        Write-Host ($PythonExe + " " + ($AnalyzeArgs -join " "))
        if ($CollectIssues) {
            Write-Host ($PythonExe + " " + ($IssueArgs -join " "))
        }
    }
    exit 0
}

& $PythonExe @PredictArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($CheckOnly -or $SkipAnalyze) {
    exit 0
}

Write-Host "Analyzing RF-DETR prediction labels..."
Write-Host ($PythonExe + " " + ($AnalyzeArgs -join " "))
& $PythonExe @AnalyzeArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($CollectIssues) {
    Write-Host "Collecting RF-DETR issue samples..."
    Write-Host ($PythonExe + " " + ($IssueArgs -join " "))
    & $PythonExe @IssueArgs
    exit $LASTEXITCODE
}

exit 0
