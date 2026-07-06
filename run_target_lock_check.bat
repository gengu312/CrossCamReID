@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

echo Starting CrossCamReID target-lock check.
echo Select two cameras, click one detected pipe/pencil, then move it through the test scene.
echo The run will analyze target lock quality and save review samples after exit.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_crosscam.ps1" -SelectCameras -PipeMode -AnalyzeAfterRun -AnalyzeTargetLockGate -AnalyzeRequireHandoff %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo Target-lock check exited with code: %EXIT_CODE%
)
echo Press any key to close this window.
pause >nul

exit /b %EXIT_CODE%
