@echo off
setlocal EnableDelayedExpansion
title FahadPrimeX
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul && set "PY=py -3"

where !PY! >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ from https://python.org first.
  pause
  exit /b 1
)

echo Checking dependencies (torch, transformers)...
!PY! -c "import torch, transformers" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies — this runs only once...
  !PY! -m pip install --quiet -r requirements.txt
  if errorlevel 1 (
    echo Failed to install dependencies. Check your connection and run this file again.
    pause
    exit /b 1
  )
)

if not exist "models\FahadPrimeX\model.safetensors" (
  echo.
  echo First run: downloading model weights ~2.2 GB. This happens only once.
  echo.
  !PY! setup_models.py
  if errorlevel 1 (
    echo Download failed. Check your connection and run this file again.
    pause
    exit /b 1
  )
)

echo.
echo Starting FahadPrimeX on http://localhost:7777
echo Keep this window open. Press Ctrl+C to stop.
echo.
start "" http://localhost:7777
!PY! app\serve.py --port 7777
pause
