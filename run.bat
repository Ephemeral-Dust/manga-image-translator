@echo off

pushd "%~dp0"
git pull --quiet
rem GPU is auto-detected; pass --no-gpu to force CPU (flag overrides this)
set "MIMG_USE_GPU=1"
venv\Scripts\python.exe -m standalone %*
popd
