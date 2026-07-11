param(
    [Parameter(Mandatory = $true)]
    [string]$Baseline,
    [Parameter(Mandatory = $true)]
    [string]$Candidate,
    [string]$OutputJson = "",
    [string]$OutputMd = "",
    [double]$MinMatchRatio = 0.98,
    [int]$MaxSwitchIncrease = 0,
    [double]$MaxAverageDistanceRatio = 1.10,
    [double]$MaxMaximumDistanceRatio = 1.10,
    [int]$MaxNewIdIncrease = 0,
    [int]$MaxRegisteredLeftIncrease = 0
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    $PythonExe = (Get-Command py -ErrorAction SilentlyContinue).Source
}
if (-not $PythonExe) {
    throw "Python was not found. Install Python and try again."
}

$AppArgs = @(
    "src\compare_target_lock_runs.py",
    "--baseline", $Baseline,
    "--candidate", $Candidate,
    "--min-match-ratio", "$MinMatchRatio",
    "--max-switch-increase", "$MaxSwitchIncrease",
    "--max-average-distance-ratio", "$MaxAverageDistanceRatio",
    "--max-maximum-distance-ratio", "$MaxMaximumDistanceRatio",
    "--max-new-id-increase", "$MaxNewIdIncrease",
    "--max-registered-left-increase", "$MaxRegisteredLeftIncrease"
)
if ($OutputJson -ne "") {
    $AppArgs += @("--output-json", $OutputJson)
}
if ($OutputMd -ne "") {
    $AppArgs += @("--output-md", $OutputMd)
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
