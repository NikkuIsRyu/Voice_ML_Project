@echo off
chcp 65001 > nul
title Antigravity Chinese Pronunciation Tutor Server

echo =======================================================
echo   Antigravity Chinese Pronunciation Tutor
echo   Starting FastAPI Server on http://0.0.0.0:8000
echo =======================================================

set "CONDA_ENV=D:\miniconda3\envs\GPTSoVits"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PATH=%CONDA_ENV%\Lib\site-packages\torch\lib;%CONDA_ENV%;%CONDA_ENV%\Scripts;%PATH%"

cd /d "%~dp0"

:loop
echo [%date% %time%] Launching tutor server...
"%CONDA_ENV%\python.exe" -m uvicorn tutor.backend.main:app --host 0.0.0.0 --port 8000 --app-dir .
echo [%date% %time%] Server process exited with code %errorlevel%.
echo Restarting in 5 seconds (Press Ctrl+C to stop)...
timeout /t 5 > nul
goto loop
