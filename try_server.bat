@echo off
rem Drag an image file onto this .bat (or pass it as an argument) to test the server.
pushd "%~dp0"
venv\Scripts\python.exe try_server.py %*
popd
pause
