param(
    [string]$Log = "",
    [string]$LogDir = "runs",
    [switch]$RequireHandoff,
    [switch]$TargetLockGate,
    [int]$MaxNewIds = -1,
    [int]$MaxNewAfterRegister = -1,
    [int]$MaxUniqueIds = -1,
    [int]$MinTargetMatches = -1,
    [double]$MinTargetSimilarity = -1,
    [int]$MaxRegisteredLefts = -1,
    [int]$MaxTargetSwitches = -1,
    [double]$MaxTargetDistance = -1,
    [int]$MaxBlockedTargetCandidates = -1,
    [int]$MinCrossCameraIds = -1,
    [string]$TargetSamplesCsv = "",
    [int]$MinTargetSamples = -1,
    [int]$MinMatchSamples = -1,
    [int]$MinSampleCameras = -1,
    [string]$SummaryJson = "",
    [string]$SummaryMd = ""
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

if ($TargetLockGate -and $SummaryJson -eq "") {
    if ($Log -ne "") {
        $SummaryRoot = Split-Path -Parent $Log
        if ($SummaryRoot -eq "") {
            $SummaryRoot = "."
        }
    } else {
        $SummaryRoot = $LogDir
    }
    $SummaryJson = Join-Path $SummaryRoot "latest-target-lock-summary.json"
}

if ($TargetLockGate -and $SummaryMd -eq "") {
    if ($Log -ne "") {
        $SummaryRoot = Split-Path -Parent $Log
        if ($SummaryRoot -eq "") {
            $SummaryRoot = "."
        }
    } else {
        $SummaryRoot = $LogDir
    }
    $SummaryMd = Join-Path $SummaryRoot "latest-target-lock-summary.md"
}

$AppArgs = @("src\analyze_run_log.py", "--log-dir", $LogDir)
if ($Log -ne "") {
    $AppArgs += @("--log", $Log)
}
if ($RequireHandoff) {
    $AppArgs += "--require-handoff"
}
if ($TargetLockGate) {
    $AppArgs += "--target-lock-gate"
}
if ($MaxNewIds -ge 0) {
    $AppArgs += @("--max-new-ids", "$MaxNewIds")
}
if ($MaxNewAfterRegister -ge 0) {
    $AppArgs += @("--max-new-after-register", "$MaxNewAfterRegister")
}
if ($MaxUniqueIds -ge 0) {
    $AppArgs += @("--max-unique-ids", "$MaxUniqueIds")
}
if ($MinTargetMatches -ge 0) {
    $AppArgs += @("--min-target-matches", "$MinTargetMatches")
}
if ($MinTargetSimilarity -ge 0) {
    $AppArgs += @("--min-target-similarity", "$MinTargetSimilarity")
}
if ($MaxRegisteredLefts -ge 0) {
    $AppArgs += @("--max-registered-lefts", "$MaxRegisteredLefts")
}
if ($MaxTargetSwitches -ge 0) {
    $AppArgs += @("--max-target-switches", "$MaxTargetSwitches")
}
if ($MaxTargetDistance -ge 0) {
    $AppArgs += @("--max-target-distance", "$MaxTargetDistance")
}
if ($MaxBlockedTargetCandidates -ge 0) {
    $AppArgs += @("--max-blocked-target-candidates", "$MaxBlockedTargetCandidates")
}
if ($MinCrossCameraIds -ge 0) {
    $AppArgs += @("--min-cross-camera-ids", "$MinCrossCameraIds")
}
if ($TargetSamplesCsv -ne "") {
    $AppArgs += @("--target-samples-csv", $TargetSamplesCsv)
}
if ($MinTargetSamples -ge 0) {
    $AppArgs += @("--min-target-samples", "$MinTargetSamples")
}
if ($MinMatchSamples -ge 0) {
    $AppArgs += @("--min-match-samples", "$MinMatchSamples")
}
if ($MinSampleCameras -ge 0) {
    $AppArgs += @("--min-sample-cameras", "$MinSampleCameras")
}
if ($SummaryJson -ne "") {
    $AppArgs += @("--summary-json", $SummaryJson)
}
if ($SummaryMd -ne "") {
    $AppArgs += @("--summary-md", $SummaryMd)
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
