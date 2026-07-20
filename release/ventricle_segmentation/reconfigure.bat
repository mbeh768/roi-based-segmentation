@echo off
REM ===================================================================
REM  Switch between the GPU and CPU version of PyTorch.
REM
REM  Double-click this if you installed the CPU version and later want
REM  GPU speed (or the other way round). It reinstalls PyTorch only --
REM  the other packages are left alone, so it is much quicker than a
REM  full reinstall.
REM ===================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No environment found yet. Run run_windows.bat first.
    pause
    exit /b 1
)

echo.
echo Which version of PyTorch should be installed?
echo    1^) GPU  - needs an NVIDIA graphics card, about 3 GB
echo    2^) CPU  - works anywhere, about 300 MB
echo.
set /p CHOICE="Enter 1 or 2: "

if "%CHOICE%"=="1" (
    .venv\Scripts\python.exe install_deps.py --gpu --yes
) else if "%CHOICE%"=="2" (
    .venv\Scripts\python.exe install_deps.py --cpu --yes
) else (
    echo Not a valid choice. Nothing was changed.
    pause
    exit /b 1
)

echo.
pause
