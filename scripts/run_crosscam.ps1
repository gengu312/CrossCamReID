param(
    [string]$CameraIndexes = "",
    [string]$CamA = "auto",
    [string]$CamB = "auto",
    [string]$CameraScanOrder = "1,3,2,0,4,5",
    [string]$PreferredCameraIndexes = "1,3",
    [int]$ProbeMax = 10,
    [string]$Backend = "dshow",
    [string]$RoiA = "80,80,480,220",
    [string]$RoiB = "80,80,480,220",
    [string]$RoiC = "80,80,480,220",
    [int]$WarmupFrames = 30,
    [ValidateSet("motion", "yolo", "rfdetr")]
    [string]$Detector = "motion",
    [int]$MinArea = 900,
    [double]$CrossThreshold = 0.65,
    [string]$TargetMode = "pencil",
    [bool]$SingleObject = $true,
    [int]$MaxDetections = 4,
    [double]$MaxAreaRatio = 0.45,
    [double]$MaxShapeRatio = 0.75,
    [int]$MinLongSide = 45,
    [int]$MaxShortSide = 180,
    [int]$MaxMissed = 14,
    [double]$LostTtl = 8.0,
    [double]$TargetThreshold = 0.58,
    [double]$TargetUpdateAlpha = 0.04,
    [int]$TargetTemplateLimit = 6,
    [double]$TargetStickDistance = 120.0,
    [double]$TargetSwitchMargin = 0.08,
    [int]$TargetSampleMaxCount = 12,
    [double]$TargetSampleMinSimilarity = 0.72,
    [double]$TargetSampleMinInterval = 0.8,
    [string]$YoloModel = "yolov8n.pt",
    [double]$YoloConf = 0.25,
    [double]$YoloIou = 0.45,
    [int]$YoloImgsz = 640,
    [string]$YoloDevice = "",
    [string]$YoloClasses = "",
    [ValidateSet("nano", "small", "base", "medium", "large", "xlarge", "2xlarge")]
    [string]$RfDetrSize = "nano",
    [string]$RfDetrWeights = "",
    [int]$RfDetrNumClasses = 0,
    [double]$RfDetrConf = 0.35,
    [string]$RfDetrClasses = "",
    [ValidateSet("auto", "zero", "category")]
    [string]$RfDetrClassIdMode = "auto",
    [int]$RfDetrCategoryIdOffset = 1,
    [switch]$RfDetrOptimize,
    [double]$PredictionHorizon = 0.35,
    [string]$LogDir = "runs",
    [ValidateSet("AB", "BA")]
    [string]$ViewOrder = "AB",
    [switch]$FlipA,
    [switch]$FlipB,
    [switch]$FlipC,
    [switch]$FlipBoth,
    [switch]$ShowTrails,
    [switch]$Headless,
    [int]$Frames = 0,
    [switch]$Demo,
    [switch]$FallbackDemo,
    [switch]$RequireMatch,
    [switch]$Probe,
    [switch]$AutoRegisterFirst,
    [switch]$PipeMode,
    [switch]$TrackAllAfterRegister,
    [switch]$SelectCameras,
    [switch]$AnalyzeAfterRun,
    [switch]$AnalyzeRequireHandoff,
    [switch]$AnalyzeTargetLockGate,
    [int]$AnalyzeMaxUniqueIds = -1,
    [int]$AnalyzeMaxNewAfterRegister = -1,
    [int]$AnalyzeMaxRegisteredLefts = -1,
    [int]$AnalyzeMinTargetMatches = -1,
    [double]$AnalyzeMinTargetSimilarity = -1,
    [int]$AnalyzeMaxTargetSwitches = -1,
    [double]$AnalyzeMaxTargetDistance = -1,
    [int]$AnalyzeMaxTargetJumps = -1,
    [int]$AnalyzeMaxBlockedTargetCandidates = -1,
    [int]$AnalyzeMinCrossCameraIds = -1,
    [int]$AnalyzeMinTargetSamples = -1,
    [int]$AnalyzeMinMatchSamples = -1,
    [int]$AnalyzeMinSampleCameras = -1,
    [string]$AnalyzeSummaryJson = "",
    [string]$AnalyzeSummaryMd = "",
    [switch]$CollectTargetSamplesAfterRun,
    [string]$TargetSampleReviewDir = "",
    [switch]$TargetSamplePreview,
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

$DetectorWasProvided = $PSBoundParameters.ContainsKey("Detector")
$RfDetrNumClassesWasProvided = $PSBoundParameters.ContainsKey("RfDetrNumClasses")
$RfDetrClassesWasProvided = $PSBoundParameters.ContainsKey("RfDetrClasses")
$SelectorBaseDetector = $Detector
$SelectorBaseMaxDetections = $MaxDetections
$SelectorBaseCrossThreshold = $CrossThreshold
$SelectorBaseTargetThreshold = $TargetThreshold
$SelectorBaseTargetUpdateAlpha = $TargetUpdateAlpha
$SelectorBaseTrackAllAfterRegister = $TrackAllAfterRegister

if ($PipeMode) {
    $DefaultPipeModel = "runs_yolo\pipe_yolov8n\weights\best.pt"
    if ((-not $DetectorWasProvided) -or $Detector -eq "motion") {
        $Detector = "yolo"
    }
    if ($Detector -eq "yolo") {
        if ($YoloModel -eq "yolov8n.pt" -and (Test-Path $DefaultPipeModel)) {
            $YoloModel = $DefaultPipeModel
        } elseif ($YoloModel -eq "yolov8n.pt") {
            Write-Host "PipeMode warning: trained pipe model was not found at runs_yolo\pipe_yolov8n\weights\best.pt. Using yolov8n.pt for smoke testing only."
        }
    } elseif ($Detector -eq "rfdetr") {
        $DefaultRfDetrWeights = Resolve-RfDetrCheckpoint $RfDetrSize
        if ($RfDetrWeights -eq "" -and $DefaultRfDetrWeights -ne "") {
            $RfDetrWeights = $DefaultRfDetrWeights
            if (-not $RfDetrNumClassesWasProvided) {
                $RfDetrNumClasses = 1
            }
            if (-not $RfDetrClassesWasProvided) {
                $RfDetrClasses = "0"
            }
        } elseif ($RfDetrWeights -eq "") {
            Write-Host "PipeMode warning: trained RF-DETR checkpoint was not found under runs_rfdetr\pipe_rfdetr_$RfDetrSize. Using the selected RF-DETR base weights for smoke testing only."
        }
    }
    $TargetMode = "general"
    $SingleObject = $false
    $MaxDetections = [Math]::Max($MaxDetections, 30)
    $CrossThreshold = 0.62
    $TargetThreshold = 0.50
    $TargetUpdateAlpha = 0.0
    $TrackAllAfterRegister = $true
}

$EffectiveAnalyzeSummaryJson = $AnalyzeSummaryJson
if ($AnalyzeAfterRun -and -not $Probe -and $EffectiveAnalyzeSummaryJson -eq "") {
    $EffectiveAnalyzeSummaryJson = Join-Path $LogDir "latest-summary.json"
}
$EffectiveAnalyzeSummaryMd = $AnalyzeSummaryMd
if ($AnalyzeAfterRun -and -not $Probe -and $EffectiveAnalyzeSummaryMd -eq "") {
    $EffectiveAnalyzeSummaryMd = Join-Path $LogDir "latest-summary.md"
}

if ($AnalyzeTargetLockGate -and -not $Probe) {
    $CollectTargetSamplesAfterRun = $true
    $TargetSamplePreview = $true
}

function Write-Utf8Host {
    param([string]$Base64Text)
    & $PythonExe -c "import base64; print(base64.b64decode('$Base64Text').decode('utf-8'))"
}

function Show-RunAnalysisSummary {
    param([string]$SummaryPath, [string]$ReportPath = "")

    if ($SummaryPath -eq "" -or -not (Test-Path -LiteralPath $SummaryPath)) {
        return
    }

    try {
        $Summary = Get-Content -LiteralPath $SummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Host "Run analysis summary read failed: $SummaryPath"
        return
    }

    $Status = $Summary.target_lock_status
    if (-not $Status) {
        return
    }

    $StatusLabel = $Summary.target_lock_status_label
    if (-not $StatusLabel) {
        $StatusLabel = $Status
    }

    Write-Host ""
    Write-Host "Run analysis quick summary:"
    Write-Host "  target_lock: $StatusLabel ($Status)"
    if ($Summary.registered_id) {
        Write-Host "  registered_target: $($Summary.registered_id), target_matches: $($Summary.target_match_count)"
    } else {
        Write-Host "  registered_target: none, target_matches: $($Summary.target_match_count)"
    }
    if ($Summary.handoff_success) {
        Write-Host "  handoff: success"
    } else {
        Write-Host "  handoff: not_confirmed"
    }

    $Actions = @($Summary.recommended_actions)
    if ($Actions.Count -gt 0) {
        Write-Host "  next_action: $($Actions[0])"
    }
    if ($ReportPath -ne "" -and (Test-Path -LiteralPath $ReportPath)) {
        Write-Host "  report: $ReportPath"
    }
}

function Add-TargetSampleReviewToReport {
    param([string]$ReportPath, [string]$ReviewDir)

    if ($ReportPath -eq "" -or -not (Test-Path -LiteralPath $ReportPath)) {
        return
    }

    $SamplesReport = Join-Path $ReviewDir "target_samples.csv"
    $PreviewPath = Join-Path $ReviewDir "target_samples_preview.jpg"
    $Lines = @(
        "",
        "## Target Sample Review",
        "",
        ('- review_dir: `{0}`' -f $ReviewDir)
    )
    if (Test-Path -LiteralPath $SamplesReport) {
        $Lines += ('- samples_csv: `{0}`' -f $SamplesReport)
    }
    if (Test-Path -LiteralPath $PreviewPath) {
        $Lines += ('- preview: `{0}`' -f $PreviewPath)
    }
    $Lines | Add-Content -LiteralPath $ReportPath -Encoding UTF8
}

if (-not $SkipInstall -and -not $PrintOnly) {
    & $PythonExe -c "import cv2, numpy, PIL" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Utf8Host "5q2j5Zyo5LuOIHJlcXVpcmVtZW50cy50eHQg5a6J6KOFIFB5dGhvbiDkvp3otZYuLi4="
        & $PythonExe -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
    if ($Detector -eq "yolo") {
        & $PythonExe -c "import ultralytics" *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "正在安装 YOLO 依赖 ultralytics..."
            & $PythonExe -m pip install ultralytics
            if ($LASTEXITCODE -ne 0) {
                exit $LASTEXITCODE
            }
        }
    }
    if ($Detector -eq "rfdetr") {
        & $PythonExe -c "import rfdetr" *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "正在安装 RF-DETR 依赖 rfdetr..."
            & $PythonExe -m pip install -r requirements-rfdetr.txt
            if ($LASTEXITCODE -ne 0) {
                exit $LASTEXITCODE
            }
        }
    }
}

if ($SelectCameras) {
    $SelectorArgs = @(
        "src\camera_selector.py",
        "--backend", $Backend,
        "--probe-max", "$ProbeMax",
        "--preferred-indexes", $PreferredCameraIndexes,
        "--view-order", $ViewOrder,
        "--script", "scripts\run_crosscam.ps1"
    )
    function Add-SelectorExtraArg {
        param([string]$Value)
        $script:SelectorArgs += "--extra-arg=$Value"
    }
    function Add-SelectorExtraPair {
        param([string]$Name, [string]$Value)
        Add-SelectorExtraArg $Name
        Add-SelectorExtraArg $Value
    }

    Add-SelectorExtraPair "-RoiA" $RoiA
    Add-SelectorExtraPair "-RoiB" $RoiB
    Add-SelectorExtraPair "-RoiC" $RoiC
    Add-SelectorExtraPair "-Detector" $SelectorBaseDetector
    Add-SelectorExtraPair "-MaxDetections" "$SelectorBaseMaxDetections"
    Add-SelectorExtraPair "-CrossThreshold" "$SelectorBaseCrossThreshold"
    Add-SelectorExtraPair "-MaxMissed" "$MaxMissed"
    Add-SelectorExtraPair "-LostTtl" "$LostTtl"
    Add-SelectorExtraPair "-TargetThreshold" "$SelectorBaseTargetThreshold"
    Add-SelectorExtraPair "-TargetUpdateAlpha" "$SelectorBaseTargetUpdateAlpha"
    Add-SelectorExtraPair "-TargetStickDistance" "$TargetStickDistance"
    Add-SelectorExtraPair "-TargetSwitchMargin" "$TargetSwitchMargin"
    Add-SelectorExtraPair "-YoloModel" $YoloModel
    Add-SelectorExtraPair "-YoloConf" "$YoloConf"
    Add-SelectorExtraPair "-YoloIou" "$YoloIou"
    Add-SelectorExtraPair "-YoloImgsz" "$YoloImgsz"
    Add-SelectorExtraPair "-RfDetrSize" $RfDetrSize
    Add-SelectorExtraPair "-RfDetrNumClasses" "$RfDetrNumClasses"
    Add-SelectorExtraPair "-RfDetrConf" "$RfDetrConf"
    Add-SelectorExtraPair "-RfDetrClassIdMode" $RfDetrClassIdMode
    Add-SelectorExtraPair "-RfDetrCategoryIdOffset" "$RfDetrCategoryIdOffset"
    Add-SelectorExtraPair "-PredictionHorizon" "$PredictionHorizon"
    Add-SelectorExtraPair "-LogDir" $LogDir
    if ($YoloDevice -ne "") {
        Add-SelectorExtraPair "-YoloDevice" $YoloDevice
    }
    if ($YoloClasses -ne "") {
        Add-SelectorExtraPair "-YoloClasses" $YoloClasses
    }
    if ($RfDetrWeights -ne "") {
        Add-SelectorExtraPair "-RfDetrWeights" $RfDetrWeights
    }
    if ($RfDetrClasses -ne "") {
        Add-SelectorExtraPair "-RfDetrClasses" $RfDetrClasses
    }
    if ($PipeMode) {
        $SelectorArgs += "--pipe-mode"
    }
    if ($FlipBoth) {
        $SelectorArgs += "--flip-both"
    }
    if ($ShowTrails) {
        $SelectorArgs += "--show-trails"
    }
    if ($RfDetrOptimize) {
        Add-SelectorExtraArg "-RfDetrOptimize"
    }
    if ($SelectorBaseTrackAllAfterRegister) {
        Add-SelectorExtraArg "-TrackAllAfterRegister"
    }
    if ($FallbackDemo) {
        Add-SelectorExtraArg "-FallbackDemo"
    }
    if ($RequireMatch) {
        Add-SelectorExtraArg "-RequireMatch"
    }
    if ($AnalyzeAfterRun) {
        Add-SelectorExtraArg "-AnalyzeAfterRun"
    }
    if ($AnalyzeRequireHandoff) {
        Add-SelectorExtraArg "-AnalyzeRequireHandoff"
    }
    if ($AnalyzeTargetLockGate) {
        Add-SelectorExtraArg "-AnalyzeTargetLockGate"
    }
    if ($AnalyzeMaxUniqueIds -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMaxUniqueIds" "$AnalyzeMaxUniqueIds"
    }
    if ($AnalyzeMaxNewAfterRegister -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMaxNewAfterRegister" "$AnalyzeMaxNewAfterRegister"
    }
    if ($AnalyzeMaxRegisteredLefts -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMaxRegisteredLefts" "$AnalyzeMaxRegisteredLefts"
    }
    if ($AnalyzeMinTargetMatches -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMinTargetMatches" "$AnalyzeMinTargetMatches"
    }
    if ($AnalyzeMinTargetSimilarity -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMinTargetSimilarity" "$AnalyzeMinTargetSimilarity"
    }
    if ($AnalyzeMaxTargetSwitches -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMaxTargetSwitches" "$AnalyzeMaxTargetSwitches"
    }
    if ($AnalyzeMaxTargetDistance -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMaxTargetDistance" "$AnalyzeMaxTargetDistance"
    }
    if ($AnalyzeMaxTargetJumps -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMaxTargetJumps" "$AnalyzeMaxTargetJumps"
    }
    if ($AnalyzeMaxBlockedTargetCandidates -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMaxBlockedTargetCandidates" "$AnalyzeMaxBlockedTargetCandidates"
    }
    if ($AnalyzeMinCrossCameraIds -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMinCrossCameraIds" "$AnalyzeMinCrossCameraIds"
    }
    if ($AnalyzeMinTargetSamples -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMinTargetSamples" "$AnalyzeMinTargetSamples"
    }
    if ($AnalyzeMinMatchSamples -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMinMatchSamples" "$AnalyzeMinMatchSamples"
    }
    if ($AnalyzeMinSampleCameras -ge 0) {
        Add-SelectorExtraPair "-AnalyzeMinSampleCameras" "$AnalyzeMinSampleCameras"
    }
    if ($EffectiveAnalyzeSummaryJson -ne "") {
        Add-SelectorExtraPair "-AnalyzeSummaryJson" $EffectiveAnalyzeSummaryJson
    }
    if ($EffectiveAnalyzeSummaryMd -ne "") {
        Add-SelectorExtraPair "-AnalyzeSummaryMd" $EffectiveAnalyzeSummaryMd
    }
    if ($CollectTargetSamplesAfterRun) {
        Add-SelectorExtraArg "-CollectTargetSamplesAfterRun"
    }
    if ($TargetSampleReviewDir -ne "") {
        Add-SelectorExtraPair "-TargetSampleReviewDir" $TargetSampleReviewDir
    }
    if ($TargetSamplePreview) {
        Add-SelectorExtraArg "-TargetSamplePreview"
    }
    if ($SkipInstall) {
        Add-SelectorExtraArg "-SkipInstall"
    }
    if ($PrintOnly) {
        Write-Host "Camera selector command:"
        Write-Host "$PythonExe $($SelectorArgs -join ' ')"
        Write-Host "PrintOnly: selector was not started."
        exit 0
    }
    & $PythonExe @SelectorArgs
    exit $LASTEXITCODE
}

$AppArgs = @("src\crosscam_mvp.py")

if ($Probe) {
    $AppArgs += @("--probe", "--probe-max", "$ProbeMax", "--backend", $Backend)
} elseif ($Demo) {
    $AppArgs += @(
        "--demo",
        "--detector", $Detector,
        "--target-threshold", "$TargetThreshold",
        "--target-update-alpha", "$TargetUpdateAlpha",
        "--target-template-limit", "$TargetTemplateLimit",
        "--target-stick-distance", "$TargetStickDistance",
        "--target-switch-margin", "$TargetSwitchMargin",
        "--target-sample-max-count", "$TargetSampleMaxCount",
        "--target-sample-min-similarity", "$TargetSampleMinSimilarity",
        "--target-sample-min-interval", "$TargetSampleMinInterval",
        "--max-detections", "$MaxDetections",
        "--max-missed", "$MaxMissed",
        "--lost-ttl", "$LostTtl",
        "--yolo-model", $YoloModel,
        "--yolo-conf", "$YoloConf",
        "--yolo-iou", "$YoloIou",
        "--yolo-imgsz", "$YoloImgsz",
        "--rfdetr-size", $RfDetrSize,
        "--rfdetr-num-classes", "$RfDetrNumClasses",
        "--rfdetr-conf", "$RfDetrConf",
        "--rfdetr-class-id-mode", $RfDetrClassIdMode,
        "--rfdetr-category-id-offset", "$RfDetrCategoryIdOffset",
        "--prediction-horizon", "$PredictionHorizon",
        "--view-order", $ViewOrder,
        "--log-dir", $LogDir
    )
    if ($CameraIndexes -ne "") {
        $AppArgs += @("--camera-indexes", $CameraIndexes)
    }
    if ($YoloDevice -ne "") {
        $AppArgs += @("--yolo-device", $YoloDevice)
    }
    if ($YoloClasses -ne "") {
        $AppArgs += @("--yolo-classes", $YoloClasses)
    }
    if ($RfDetrClasses -ne "") {
        $AppArgs += @("--rfdetr-classes", $RfDetrClasses)
    }
    if ($RfDetrWeights -ne "") {
        $AppArgs += @("--rfdetr-weights", $RfDetrWeights)
    }
    if ($RfDetrOptimize) {
        $AppArgs += "--rfdetr-optimize"
    }
} else {
    if ($CameraIndexes -ne "") {
        $AppArgs += @("--camera-indexes", $CameraIndexes)
    } else {
        $AppArgs += @("--cam-a", "$CamA", "--cam-b", "$CamB")
    }
    $AppArgs += @(
        "--camera-scan-order", $CameraScanOrder,
        "--probe-max", "$ProbeMax",
        "--backend", $Backend,
        "--roi-a", $RoiA,
        "--roi-b", $RoiB,
        "--roi-c", $RoiC,
        "--detector", $Detector,
        "--warmup-frames", "$WarmupFrames",
        "--min-area", "$MinArea",
        "--target-mode", $TargetMode,
        "--max-detections", "$MaxDetections",
        "--max-area-ratio", "$MaxAreaRatio",
        "--max-shape-ratio", "$MaxShapeRatio",
        "--min-long-side", "$MinLongSide",
        "--max-short-side", "$MaxShortSide",
        "--cross-threshold", "$CrossThreshold",
        "--max-missed", "$MaxMissed",
        "--lost-ttl", "$LostTtl",
        "--target-threshold", "$TargetThreshold",
        "--target-update-alpha", "$TargetUpdateAlpha",
        "--target-template-limit", "$TargetTemplateLimit",
        "--target-stick-distance", "$TargetStickDistance",
        "--target-switch-margin", "$TargetSwitchMargin",
        "--target-sample-max-count", "$TargetSampleMaxCount",
        "--target-sample-min-similarity", "$TargetSampleMinSimilarity",
        "--target-sample-min-interval", "$TargetSampleMinInterval",
        "--yolo-model", $YoloModel,
        "--yolo-conf", "$YoloConf",
        "--yolo-iou", "$YoloIou",
        "--yolo-imgsz", "$YoloImgsz",
        "--rfdetr-size", $RfDetrSize,
        "--rfdetr-num-classes", "$RfDetrNumClasses",
        "--rfdetr-conf", "$RfDetrConf",
        "--rfdetr-class-id-mode", $RfDetrClassIdMode,
        "--rfdetr-category-id-offset", "$RfDetrCategoryIdOffset",
        "--prediction-horizon", "$PredictionHorizon",
        "--view-order", $ViewOrder,
        "--log-dir", $LogDir
    )
    if ($YoloDevice -ne "") {
        $AppArgs += @("--yolo-device", $YoloDevice)
    }
    if ($YoloClasses -ne "") {
        $AppArgs += @("--yolo-classes", $YoloClasses)
    }
    if ($RfDetrClasses -ne "") {
        $AppArgs += @("--rfdetr-classes", $RfDetrClasses)
    }
    if ($RfDetrWeights -ne "") {
        $AppArgs += @("--rfdetr-weights", $RfDetrWeights)
    }
    if ($RfDetrOptimize) {
        $AppArgs += "--rfdetr-optimize"
    }

    if ($SingleObject) {
        $AppArgs += "--single-object"
    }

}

if (-not $Probe) {
    if ($FlipA -or $FlipBoth) {
        $AppArgs += "--flip-a"
    }
    if ($FlipB -or $FlipBoth) {
        $AppArgs += "--flip-b"
    }
    if ($FlipC) {
        $AppArgs += "--flip-c"
    }
    if ($ShowTrails) {
        $AppArgs += "--show-trails"
    }
    if ($FallbackDemo) {
        $AppArgs += "--fallback-demo"
    }
}

if (-not $Probe -and $AutoRegisterFirst) {
    $AppArgs += "--auto-register-first"
}

if (-not $Probe -and $TrackAllAfterRegister) {
    $AppArgs += "--track-all-after-register"
}

if ($Headless) {
    $AppArgs += "--headless"
}

if ($Frames -gt 0) {
    $AppArgs += @("--frames", "$Frames")
}

if ($RequireMatch) {
    $AppArgs += "--require-match"
}

function Format-ExistingPathStatus {
    param([string]$PathValue)

    if ($PathValue -eq "") {
        return "not_set"
    }
    if (Test-Path -LiteralPath $PathValue) {
        return "found"
    }
    return "missing"
}

function Show-LaunchSummary {
    Write-Host ""
    Write-Host "Launch summary:"
    if ($Probe) {
        Write-Host "  mode: probe"
        Write-Host "  cameras: backend=$Backend, probe_max=$ProbeMax"
        return
    }

    if ($Demo) {
        $ModeLabel = "demo"
    } else {
        $ModeLabel = "camera"
    }
    if ($PipeMode) {
        $ModeLabel = "$ModeLabel + PipeMode"
    }
    Write-Host "  mode: $ModeLabel"

    if ($CameraIndexes -ne "") {
        Write-Host "  cameras: indexes=$CameraIndexes, backend=$Backend"
    } elseif (-not $Demo) {
        Write-Host "  cameras: A=$CamA, B=$CamB, scan_order=$CameraScanOrder, backend=$Backend"
    } else {
        Write-Host "  cameras: demo_source"
    }

    Write-Host "  detector: $Detector, max_detections=$MaxDetections"
    if ($Detector -eq "yolo") {
        $YoloModelStatus = Format-ExistingPathStatus $YoloModel
        $YoloDeviceText = if ($YoloDevice -ne "") { $YoloDevice } else { "auto" }
        Write-Host "  yolo: model=$YoloModel ($YoloModelStatus), conf=$YoloConf, iou=$YoloIou, imgsz=$YoloImgsz, device=$YoloDeviceText"
    } elseif ($Detector -eq "rfdetr") {
        $RfDetrWeightsStatus = Format-ExistingPathStatus $RfDetrWeights
        Write-Host "  rfdetr: size=$RfDetrSize, weights=$RfDetrWeights ($RfDetrWeightsStatus), classes=$RfDetrNumClasses, conf=$RfDetrConf"
    }
    Write-Host "  target: threshold=$TargetThreshold, update_alpha=$TargetUpdateAlpha, track_all_after_register=$TrackAllAfterRegister"
    Write-Host "  view: order=$ViewOrder, flip_a=$($FlipA -or $FlipBoth), flip_b=$($FlipB -or $FlipBoth), flip_c=$FlipC"
    Write-Host "  output: log_dir=$LogDir"
}

$RunExitCode = 0

if ($PrintOnly) {
    Write-Host "Preparing CrossCamReID launch..."
} else {
    Write-Utf8Host "5q2j5Zyo6L+Q6KGMIENyb3NzQ2FtUmVJRC4uLg=="
}
Show-LaunchSummary
Write-Host "$PythonExe $($AppArgs -join ' ')"
Write-Host ""

if ($PrintOnly) {
    Write-Host "PrintOnly: CrossCamReID was not started."
    exit 0
}

& $PythonExe @AppArgs
$AppExitCode = $LASTEXITCODE
if ($AppExitCode -ne 0) {
    exit $AppExitCode
}

if ($AnalyzeAfterRun -and -not $Probe) {
    $AnalyzeArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\analyze_run.ps1",
        "-LogDir", $LogDir
    )
    if ($AnalyzeRequireHandoff) {
        $AnalyzeArgs += "-RequireHandoff"
    }
    if ($AnalyzeTargetLockGate) {
        $AnalyzeArgs += "-TargetLockGate"
    }
    if ($AnalyzeMaxUniqueIds -ge 0) {
        $AnalyzeArgs += @("-MaxUniqueIds", "$AnalyzeMaxUniqueIds")
    }
    if ($AnalyzeMaxNewAfterRegister -ge 0) {
        $AnalyzeArgs += @("-MaxNewAfterRegister", "$AnalyzeMaxNewAfterRegister")
    }
    if ($AnalyzeMaxRegisteredLefts -ge 0) {
        $AnalyzeArgs += @("-MaxRegisteredLefts", "$AnalyzeMaxRegisteredLefts")
    }
    if ($AnalyzeMinTargetMatches -ge 0) {
        $AnalyzeArgs += @("-MinTargetMatches", "$AnalyzeMinTargetMatches")
    }
    if ($AnalyzeMinTargetSimilarity -ge 0) {
        $AnalyzeArgs += @("-MinTargetSimilarity", "$AnalyzeMinTargetSimilarity")
    }
    if ($AnalyzeMaxTargetSwitches -ge 0) {
        $AnalyzeArgs += @("-MaxTargetSwitches", "$AnalyzeMaxTargetSwitches")
    }
    if ($AnalyzeMaxTargetDistance -ge 0) {
        $AnalyzeArgs += @("-MaxTargetDistance", "$AnalyzeMaxTargetDistance")
    }
    if ($AnalyzeMaxTargetJumps -ge 0) {
        $AnalyzeArgs += @("-MaxTargetJumps", "$AnalyzeMaxTargetJumps")
    }
    if ($AnalyzeMaxBlockedTargetCandidates -ge 0) {
        $AnalyzeArgs += @("-MaxBlockedTargetCandidates", "$AnalyzeMaxBlockedTargetCandidates")
    }
    if ($AnalyzeMinCrossCameraIds -ge 0) {
        $AnalyzeArgs += @("-MinCrossCameraIds", "$AnalyzeMinCrossCameraIds")
    }
    if ($AnalyzeMinTargetSamples -ge 0) {
        $AnalyzeArgs += @("-MinTargetSamples", "$AnalyzeMinTargetSamples")
    }
    if ($AnalyzeMinMatchSamples -ge 0) {
        $AnalyzeArgs += @("-MinMatchSamples", "$AnalyzeMinMatchSamples")
    }
    if ($AnalyzeMinSampleCameras -ge 0) {
        $AnalyzeArgs += @("-MinSampleCameras", "$AnalyzeMinSampleCameras")
    }
    if ($EffectiveAnalyzeSummaryJson -ne "") {
        $AnalyzeArgs += @("-SummaryJson", $EffectiveAnalyzeSummaryJson)
    }
    if ($EffectiveAnalyzeSummaryMd -ne "") {
        $AnalyzeArgs += @("-SummaryMd", $EffectiveAnalyzeSummaryMd)
    }

    Write-Host ""
    Write-Utf8Host "5q2j5Zyo5YiG5p6Q5pys5qyh6L+Q6KGM5pel5b+XLi4u"
    & powershell @AnalyzeArgs
    $AnalyzeExitCode = $LASTEXITCODE
    Show-RunAnalysisSummary $EffectiveAnalyzeSummaryJson $EffectiveAnalyzeSummaryMd
    if ($AnalyzeExitCode -ne 0) {
        $RunExitCode = $AnalyzeExitCode
    }
}

if ($CollectTargetSamplesAfterRun -and -not $Probe) {
    $SamplesCsv = Join-Path $LogDir "targets\target_samples.csv"
    $EffectiveTargetSampleReviewDir = $TargetSampleReviewDir
    if ($EffectiveTargetSampleReviewDir -eq "") {
        $EffectiveTargetSampleReviewDir = Join-Path $LogDir "target_sample_review"
    }
    $CollectArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\collect_target_samples.ps1",
        "-SamplesCsv", $SamplesCsv,
        "-OutputDir", $EffectiveTargetSampleReviewDir,
        "-Clean",
        "-Strict"
    )
    if ($TargetSamplePreview) {
        $CollectArgs += "-Preview"
    }

    Write-Host ""
    Write-Host "Collecting target samples for this run..."
    & powershell @CollectArgs
    $CollectExitCode = $LASTEXITCODE
    if ($CollectExitCode -ne 0) {
        exit $CollectExitCode
    }
    Add-TargetSampleReviewToReport $EffectiveAnalyzeSummaryMd $EffectiveTargetSampleReviewDir
    exit $RunExitCode
}

exit $RunExitCode
