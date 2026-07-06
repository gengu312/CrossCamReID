param(
    [string]$DatasetRoot = "datasets/pipe_yolo",
    [ValidateSet("train", "val")]
    [string]$Split = "val",
    [string]$Source = "",
    [string]$YoloModel = "runs_yolo\pipe_yolov8n\weights\best.pt",
    [string]$YoloData = "datasets/pipe_yolo\data.yaml",
    [double]$YoloConf = 0.25,
    [int]$YoloImgsz = 640,
    [string]$YoloDevice = "cpu",
    [string]$YoloProject = "runs_yolo_eval",
    [string]$YoloName = "pipe_yolov8n_eval",
    [ValidateSet("nano", "small", "base", "medium", "large", "xlarge", "2xlarge")]
    [string]$RfDetrModelSize = "nano",
    [string]$RfDetrWeights = "",
    [int]$RfDetrNumClasses = 0,
    [double]$RfDetrConf = 0.35,
    [string]$RfDetrClasses = "",
    [ValidateSet("auto", "zero", "category")]
    [string]$RfDetrClassIdMode = "auto",
    [int]$RfDetrCategoryIdOffset = 1,
    [string]$RfDetrProject = "runs_rfdetr_eval",
    [string]$RfDetrName = "pipe_rfdetr_nano_eval",
    [string]$OutputCsv = "runs_detector_compare\yolo_vs_rfdetr.csv",
    [string]$SummaryJson = "runs_detector_compare\yolo_vs_rfdetr_summary.json",
    [string]$SummaryMd = "runs_detector_compare\yolo_vs_rfdetr_summary.md",
    [int]$MaxExamples = 12,
    [switch]$CollectIssues,
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

if ($Source -eq "") {
    $Source = Join-Path $DatasetRoot (Join-Path "images" $Split)
}

if ($CheckOnly -and $CollectIssues) {
    Write-Host "CheckOnly does not create prediction labels, so it cannot be used with -CollectIssues."
    exit 2
}

$YoloPredictRoot = Join-Path $YoloProject "$($YoloName)_predict"
$YoloCsv = Join-Path $YoloPredictRoot "analysis.csv"
$EffectiveRfDetrName = $RfDetrName
if ($RfDetrName -eq "pipe_rfdetr_nano_eval" -and $RfDetrModelSize -ne "nano") {
    $EffectiveRfDetrName = "pipe_rfdetr_${RfDetrModelSize}_eval"
}
$RfDetrCsv = Join-Path (Join-Path $RfDetrProject $EffectiveRfDetrName) "analysis.csv"

$YoloParams = [ordered]@{
    Model = $YoloModel
    Data = $YoloData
    DatasetRoot = $DatasetRoot
    Split = $Split
    Source = $Source
    Conf = $YoloConf
    Imgsz = $YoloImgsz
    Device = $YoloDevice
    Project = $YoloProject
    Name = $YoloName
    PredictOnly = $true
}

$RfDetrParams = [ordered]@{
    DatasetRoot = $DatasetRoot
    Split = $Split
    Source = $Source
    ModelSize = $RfDetrModelSize
    NumClasses = $RfDetrNumClasses
    Conf = $RfDetrConf
    ClassIdMode = $RfDetrClassIdMode
    CategoryIdOffset = $RfDetrCategoryIdOffset
    Project = $RfDetrProject
    Name = $RfDetrName
}
if ($RfDetrWeights -ne "") {
    $RfDetrParams["Weights"] = $RfDetrWeights
}
if ($RfDetrClasses -ne "") {
    $RfDetrParams["Classes"] = $RfDetrClasses
}

if ($CollectIssues) {
    $YoloParams["CollectIssues"] = $true
    $YoloParams["IssueMaxCount"] = $IssueMaxCount
    $RfDetrParams["CollectIssues"] = $true
    $RfDetrParams["IssueMaxCount"] = $IssueMaxCount
    if ($IssuePreview) {
        $YoloParams["IssuePreview"] = $true
        $RfDetrParams["IssuePreview"] = $true
    }
    if ($IssueClean) {
        $YoloParams["IssueClean"] = $true
        $RfDetrParams["IssueClean"] = $true
    }
}
if ($SkipInstall) {
    $YoloParams["SkipInstall"] = $true
    $RfDetrParams["SkipInstall"] = $true
}

$CompareParams = [ordered]@{
    YoloCsv = $YoloCsv
    RfDetrCsv = $RfDetrCsv
    OutputCsv = $OutputCsv
    SummaryJson = $SummaryJson
    SummaryMd = $SummaryMd
    MaxExamples = $MaxExamples
}

function Format-CommandLine {
    param(
        [string]$ScriptPath,
        [System.Collections.IDictionary]$Parameters
    )

    function Format-Argument {
        param([object]$Value)

        $Text = "$Value"
        if ($Text -eq "") {
            return "''"
        }
        if ($Text -match "[\s'`"]") {
            return "'" + ($Text -replace "'", "''") + "'"
        }
        return $Text
    }

    $Arguments = @()
    foreach ($Entry in $Parameters.GetEnumerator()) {
        if ($Entry.Value -is [bool]) {
            if ($Entry.Value) {
                $Arguments += "-$($Entry.Key)"
            }
            continue
        }
        $Arguments += "-$($Entry.Key)"
        $Arguments += (Format-Argument $Entry.Value)
    }
    return "powershell -NoProfile -ExecutionPolicy Bypass -File $ScriptPath " + ($Arguments -join " ")
}

if ($PrintOnly) {
    Write-Host (Format-CommandLine "scripts\evaluate_pipe_yolo.ps1" $YoloParams)
    Write-Host (Format-CommandLine "scripts\evaluate_rfdetr.ps1" $RfDetrParams)
    Write-Host (Format-CommandLine "scripts\compare_detector_eval.ps1" $CompareParams)
    exit 0
}

if ($CheckOnly) {
    $Failures = @()
    foreach ($Check in @(
        @{ Path = $DatasetRoot; Label = "Dataset root" },
        @{ Path = $Source; Label = "Prediction source" },
        @{ Path = $YoloData; Label = "YOLO data.yaml" },
        @{ Path = $YoloModel; Label = "YOLO model" }
    )) {
        if (-not (Test-Path -LiteralPath $Check.Path)) {
            $Failures += "$($Check.Label) does not exist: $($Check.Path)"
        }
    }
    if ($RfDetrWeights -ne "" -and -not (Test-Path -LiteralPath $RfDetrWeights)) {
        $Failures += "RF-DETR weights do not exist: $RfDetrWeights"
    }
    if ($Failures.Count -gt 0) {
        Write-Host "Detector evaluation check failed:"
        $Failures | ForEach-Object { Write-Host "  - $_" }
        exit 2
    }

    Write-Host "Detector evaluation check:"
    Write-Host "  dataset_root=$DatasetRoot"
    Write-Host "  split=$Split"
    Write-Host "  source=$Source"
    Write-Host "  yolo_model=$YoloModel"
    Write-Host "  yolo_analysis=$YoloCsv"
    Write-Host "  rfdetr_analysis=$RfDetrCsv"
    Write-Host "  compare_summary=$SummaryMd"

    $YoloCheckParams = [ordered]@{}
    foreach ($Entry in $YoloParams.GetEnumerator()) {
        $YoloCheckParams[$Entry.Key] = $Entry.Value
    }
    $YoloCheckParams["CheckOnly"] = $true
    & (Join-Path $PSScriptRoot "evaluate_pipe_yolo.ps1") @YoloCheckParams
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $RfDetrCheckParams = [ordered]@{}
    foreach ($Entry in $RfDetrParams.GetEnumerator()) {
        $RfDetrCheckParams[$Entry.Key] = $Entry.Value
    }
    $RfDetrCheckParams["CheckOnly"] = $true
    & (Join-Path $PSScriptRoot "evaluate_rfdetr.ps1") @RfDetrCheckParams
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    Write-Host "CheckOnly: detector evaluation inputs are ready. Inference was not started."
    exit 0
}

Write-Host "Running YOLO detector evaluation..."
& (Join-Path $PSScriptRoot "evaluate_pipe_yolo.ps1") @YoloParams
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Running RF-DETR detector evaluation..."
& (Join-Path $PSScriptRoot "evaluate_rfdetr.ps1") @RfDetrParams
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Comparing detector evaluation reports..."
& (Join-Path $PSScriptRoot "compare_detector_eval.ps1") @CompareParams
exit $LASTEXITCODE
