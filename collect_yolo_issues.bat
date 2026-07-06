@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\collect_yolo_issues.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
  powershell -NoProfile -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Write-Host (([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('6Zeu6aKY5qC35pys5pS26ZuG56iL5bqP5bey6YCA5Ye677yM6YCA5Ye65Luj56CB77ya'))) + '%EXIT_CODE%')"
)
powershell -NoProfile -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Write-Host ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('5oyJ5Lu75oSP6ZSu5YWz6Zet5q2k56qX5Y+j44CC')))"
pause >nul

exit /b %EXIT_CODE%
