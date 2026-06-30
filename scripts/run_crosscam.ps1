param(
    [string]$CamA = "auto",
    [string]$CamB = "auto",
    [string]$Backend = "dshow",
    [string]$RoiA = "80,80,480,220",
    [string]$RoiB = "80,80,480,220",
    [int]$WarmupFrames = 30,
    [ValidateSet("motion", "yolo")]
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
    [double]$TargetThreshold = 0.58,
    [double]$TargetUpdateAlpha = 0.04,
    [int]$TargetTemplateLimit = 6,
    [int]$TargetSampleMaxCount = 12,
    [double]$TargetSampleMinSimilarity = 0.72,
    [double]$TargetSampleMinInterval = 0.8,
    [string]$YoloModel = "yolov8n.pt",
    [double]$YoloConf = 0.25,
    [double]$YoloIou = 0.45,
    [int]$YoloImgsz = 640,
    [string]$YoloDevice = "",
    [string]$YoloClasses = "",
    [double]$PredictionHorizon = 0.35,
    [string]$LogDir = "runs",
    [ValidateSet("AB", "BA")]
    [string]$ViewOrder = "AB",
    [switch]$Headless,
    [int]$Frames = 0,
    [switch]$Demo,
    [switch]$Probe,
    [switch]$AutoRegisterFirst,
    [switch]$PipeMode,
    [switch]$TrackAllAfterRegister,
    [switch]$SkipInstall
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

if ($PipeMode) {
    $DefaultPipeModel = "runs_yolo\pipe_yolov8n\weights\best.pt"
    if ($YoloModel -eq "yolov8n.pt" -and (Test-Path $DefaultPipeModel)) {
        $YoloModel = $DefaultPipeModel
    } elseif ($YoloModel -eq "yolov8n.pt") {
        Write-Host "PipeMode warning: trained pipe model was not found at runs_yolo\pipe_yolov8n\weights\best.pt. Using yolov8n.pt for smoke testing only."
    }
    $Detector = "yolo"
    $TargetMode = "general"
    $SingleObject = $false
    $MaxDetections = [Math]::Max($MaxDetections, 30)
    $CrossThreshold = 0.62
    $TargetThreshold = 0.50
    $TargetUpdateAlpha = 0.0
    $TrackAllAfterRegister = $true
}

function Write-Utf8Host {
    param([string]$Base64Text)
    & $PythonExe -c "import base64; print(base64.b64decode('$Base64Text').decode('utf-8'))"
}

if (-not $SkipInstall) {
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
}

$AppArgs = @("src\crosscam_mvp.py")

if ($Probe) {
    $AppArgs += @("--probe", "--probe-max", "5", "--backend", $Backend)
} elseif ($Demo) {
    $AppArgs += @(
        "--demo",
        "--detector", $Detector,
        "--target-threshold", "$TargetThreshold",
        "--target-update-alpha", "$TargetUpdateAlpha",
        "--target-template-limit", "$TargetTemplateLimit",
        "--target-sample-max-count", "$TargetSampleMaxCount",
        "--target-sample-min-similarity", "$TargetSampleMinSimilarity",
        "--target-sample-min-interval", "$TargetSampleMinInterval",
        "--max-detections", "$MaxDetections",
        "--yolo-model", $YoloModel,
        "--yolo-conf", "$YoloConf",
        "--yolo-iou", "$YoloIou",
        "--yolo-imgsz", "$YoloImgsz",
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
} else {
    $AppArgs += @(
        "--cam-a", "$CamA",
        "--cam-b", "$CamB",
        "--backend", $Backend,
        "--roi-a", $RoiA,
        "--roi-b", $RoiB,
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
        "--target-threshold", "$TargetThreshold",
        "--target-update-alpha", "$TargetUpdateAlpha",
        "--target-template-limit", "$TargetTemplateLimit",
        "--target-sample-max-count", "$TargetSampleMaxCount",
        "--target-sample-min-similarity", "$TargetSampleMinSimilarity",
        "--target-sample-min-interval", "$TargetSampleMinInterval",
        "--yolo-model", $YoloModel,
        "--yolo-conf", "$YoloConf",
        "--yolo-iou", "$YoloIou",
        "--yolo-imgsz", "$YoloImgsz",
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

    if ($SingleObject) {
        $AppArgs += "--single-object"
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

Write-Utf8Host "5q2j5Zyo6L+Q6KGMIENyb3NzQ2FtUmVJRC4uLg=="
Write-Host "$PythonExe $($AppArgs -join ' ')"
Write-Host ""

& $PythonExe @AppArgs
exit $LASTEXITCODE
