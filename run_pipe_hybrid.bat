@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

set "HYBRID_PYTHON=%CROSSCAM_HYBRID_PYTHON%"
if not defined HYBRID_PYTHON set "HYBRID_PYTHON=D:\SoftWare\PythonEnvs\CrossCamReID-rfdetr-cpu\Scripts\python.exe"
set "RFDETR_WEIGHTS=runs_rfdetr\pipe_rfdetr_nano_cpu_smoke\checkpoint_best_regular.pth"

if not exist "%HYBRID_PYTHON%" (
  echo Hybrid Python environment was not found: %HYBRID_PYTHON%
  echo Set CROSSCAM_HYBRID_PYTHON to a Python executable with YOLO and RF-DETR installed.
  exit /b 2
)

if not exist "%RFDETR_WEIGHTS%" (
  echo RF-DETR experiment weights were not found: %RFDETR_WEIGHTS%
  echo Run the RF-DETR training workflow first or pass another weight path through scripts\run_crosscam.ps1.
  exit /b 2
)

for %%I in ("%HYBRID_PYTHON%") do set "PATH=%%~dpI;%PATH%"
if not defined RF_HOME set "RF_HOME=D:\SoftWare\ModelCache\Roboflow"

"%HYBRID_PYTHON%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('ultralytics') and importlib.util.find_spec('rfdetr') else 1)" >nul 2>nul
if errorlevel 1 (
  echo Installing hybrid detector dependencies...
  "%HYBRID_PYTHON%" -m pip install -r requirements-hybrid.txt
  if errorlevel 1 exit /b %ERRORLEVEL%
)

echo Starting CrossCamReID with YOLO primary detection and RF-DETR fallback.
echo RF-DETR runs only after a registered target is not matched by YOLO.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_crosscam.ps1" -SelectCameras -PipeMode -Detector hybrid -RfDetrWeights "%RFDETR_WEIGHTS%" -RfDetrNumClasses 1 -RfDetrClasses 0 -RfDetrConf 0.25 -HybridFallbackInterval 15 -AnalyzeAfterRun -AnalyzeTargetLockGate %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" echo Hybrid detector run exited with code: %EXIT_CODE%
echo Press any key to close this window.
pause >nul

exit /b %EXIT_CODE%
