@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_crosscam.ps1" -SelectCameras
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_crosscam.ps1" %*
)
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  powershell -NoProfile -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Write-Host (([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('Q3Jvc3NDYW1SZUlEIOW3sumAgOWHuu+8jOmAgOWHuuegge+8mg=='))) + '%EXIT_CODE%')"
)
powershell -NoProfile -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Write-Host ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('5oyJ5Lu75oSP6ZSu5YWz6Zet5q2k56qX5Y+j44CC')))"
pause >nul

exit /b %EXIT_CODE%
