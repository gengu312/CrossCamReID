@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

echo Starting CrossCamReID with RF-DETR pipe backend.
echo Select two cameras, click one detected pipe, then move it through the test scene.
echo If a trained RF-DETR checkpoint exists, PipeMode will use it automatically.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_crosscam.ps1" -SelectCameras -PipeMode -Detector rfdetr -RfDetrNumClasses 1 -RfDetrClasses 0 -RfDetrConf 0.35 -AnalyzeAfterRun -AnalyzeTargetLockGate -AnalyzeRequireHandoff %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo RF-DETR pipe run exited with code: %EXIT_CODE%
)
echo Press any key to close this window.
pause >nul

exit /b %EXIT_CODE%
