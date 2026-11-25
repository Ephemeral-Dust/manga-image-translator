@echo off

pushd "%~dp0"
rem No automatic git pull here: this fork's main has diverged from upstream,
rem and pulling in place caused an unattended merge conflict + a missing-module
rem crash (upstream swapped arabic-reshaper for khatt, venv not updated).
rem Update manually: git fetch origin; git merge origin/main  (then resolve
rem conflicts and run: pip install -r requirements.txt)
rem GPU is auto-detected; pass --no-gpu to force CPU (flag overrides this)
set "MIMG_USE_GPU=0"
venv\Scripts\python.exe -m standalone %*
popd
