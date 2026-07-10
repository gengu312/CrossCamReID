@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

if "%~1"=="" (
  echo Usage:
  echo   run_video_replay.bat -VideoA "D:\videos\camera_a.mp4" -VideoB "D:\videos\camera_b.mp4"
  echo.
  echo Optional: -VideoC, -VideoPlaybackRate 0.5, -LoopVideos, -AnalyzeRequireHandoff
  echo.
  echo Press any key to close this window.
  pause >nul
  exit /b 2
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_crosscam.ps1" -PipeMode -AnalyzeAfterRun -AnalyzeTargetLockGate %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo Video replay or target-lock analysis exited with code: %EXIT_CODE%
)
echo Press any key to close this window.
pause >nul

exit /b %EXIT_CODE%
