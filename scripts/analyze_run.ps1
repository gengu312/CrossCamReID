param(
    [string]$Log = "",
    [string]$LogDir = "runs",
    [switch]$RequireHandoff,
    [int]$MaxNewIds = -1,
    [int]$MaxUniqueIds = -1,
    [int]$MinTargetMatches = -1,
    [int]$MinCrossCameraIds = -1
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

$AppArgs = @("src\analyze_run_log.py", "--log-dir", $LogDir)
if ($Log -ne "") {
    $AppArgs += @("--log", $Log)
}
if ($RequireHandoff) {
    $AppArgs += "--require-handoff"
}
if ($MaxNewIds -ge 0) {
    $AppArgs += @("--max-new-ids", "$MaxNewIds")
}
if ($MaxUniqueIds -ge 0) {
    $AppArgs += @("--max-unique-ids", "$MaxUniqueIds")
}
if ($MinTargetMatches -ge 0) {
    $AppArgs += @("--min-target-matches", "$MinTargetMatches")
}
if ($MinCrossCameraIds -ge 0) {
    $AppArgs += @("--min-cross-camera-ids", "$MinCrossCameraIds")
}

& $PythonExe @AppArgs
exit $LASTEXITCODE
