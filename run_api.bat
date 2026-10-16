@echo off
chcp 65001 > nul
set "CONDA_ENV=D:\miniconda3\envs\GPTSoVits"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PATH=%CONDA_ENV%\Lib\site-packages\torch\lib;%CONDA_ENV%;%CONDA_ENV%\Scripts;%PATH%"

cd /d "%~dp0\GPT-SoVITS"
"%CONDA_ENV%\python.exe" api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
