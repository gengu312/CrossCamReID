@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

echo Analyzing the latest CrossCamReID event log.
echo Default log directory: %~dp0runs
echo To use another log, append -Log path or -LogDir path.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\analyze_run.ps1" -TargetLockGate %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo Target lock analysis failed, exit code: %EXIT_CODE%
)
echo Press any key to close this window.
pause >nul

exit /b %EXIT_CODE%
