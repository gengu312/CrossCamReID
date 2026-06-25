param(
    [string]$Log = "",
    [string]$LogDir = "runs",
    [switch]$RequireHandoff
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

& $PythonExe @AppArgs
exit $LASTEXITCODE
