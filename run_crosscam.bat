@echo off
setlocal

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_crosscam.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo CrossCamReID exited with code %EXIT_CODE%.
)
echo Press any key to close this window.
pause >nul

exit /b %EXIT_CODE%
