@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    py -3 -m venv .venv
    if errorlevel 1 goto :error
)
.venv\Scripts\python.exe -c "import flowdesk, PySide6, scipy, pyqtgraph, numba, pyamg, threadpoolctl" >nul 2>&1
if errorlevel 1 (
    .venv\Scripts\python.exe -m pip install -r requirements-lock.txt
    if errorlevel 1 goto :error
    .venv\Scripts\python.exe -m pip install --no-deps -e .
    if errorlevel 1 goto :error
)
.venv\Scripts\python.exe -m flowdesk
if errorlevel 1 goto :error
exit /b 0
:error
echo Installation or startup failed. See the error above.
pause
exit /b 1
