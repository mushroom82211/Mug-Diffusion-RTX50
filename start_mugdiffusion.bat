@echo off
rem MuG Diffusion launcher (GPU: PyTorch 2.7.1 + CUDA 12.8, supports RTX 5060 sm_120)
cd /d "%~dp0"
set "PATH=%~dp0..;%PATH%"
".venv\Scripts\python.exe" webui.py
pause
