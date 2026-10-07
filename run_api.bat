@echo off
cd /d "%~dp0\GPT-SoVITS"
"D:\miniconda3\envs\GPTSoVits\python.exe" api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
pause
