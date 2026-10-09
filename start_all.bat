@echo off
chcp 65001 > nul
title Antigravity Chinese Tutor & AI Teacher — All Services

echo ======================================================================
echo   Antigravity Chinese Pronunciation Tutor & Live AI Teacher
echo   Launching GPT-SoVITS TTS Server (9880) and Tutor Web Server (8000)
echo ======================================================================

set "PROJECT_ROOT=%~dp0"
set "CONDA_ENV=D:\miniconda3\envs\GPTSoVits"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PATH=%CONDA_ENV%\Lib\site-packages\torch\lib;%CONDA_ENV%;%CONDA_ENV%\Scripts;%PATH%"

cd /d "%PROJECT_ROOT%"

echo.
echo [1/2] Starting GPT-SoVITS v2ProPlus TTS Server on port 9880 in background...
start "GPT-SoVITS v2ProPlus Server (Port 9880)" cmd /k "cd /d "%PROJECT_ROOT%GPT-SoVITS" && "%CONDA_ENV%\python.exe" api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml"

echo Waiting 3 seconds for TTS engine initialization...
timeout /t 3 > nul

echo.
echo [2/2] Starting Chinese Pronunciation Tutor & AI Teacher on http://localhost:8000 ...
start "" "http://localhost:8000"

:loop
echo [%date% %time%] Running Tutor Web Server...
"%CONDA_ENV%\python.exe" -m uvicorn tutor.backend.main:app --host 0.0.0.0 --port 8000 --app-dir "%PROJECT_ROOT%"
echo [%date% %time%] Server process exited with code %errorlevel%.
echo Restarting in 5 seconds (Press Ctrl+C to stop)...
timeout /t 5 > nul
goto loop
