@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\summarize_capture_manifest.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo Capture manifest summary failed, exit code: %EXIT_CODE%
)
echo Press any key to close this window.
pause >nul

exit /b %EXIT_CODE%
