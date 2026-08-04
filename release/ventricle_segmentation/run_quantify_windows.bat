@echo off
REM ===================================================================
REM  Epicardium quantification - Windows launcher
REM
REM  Double-click this file. The first run creates a private Python
REM  environment in a .venv folder next to this script and installs
REM  everything needed (shared with run_windows.bat -- if you already
REM  ran that once, this reuses the same environment).
REM
REM  Nothing is installed system-wide and nothing outside this folder
REM  is modified.
REM ===================================================================
setlocal
cd /d "%~dp0"

set VENV_DIR=.venv
set STAMP=%VENV_DIR%\.deps_installed

REM ---- locate a Python interpreter -----------------------------------
set PY_CMD=
where py >nul 2>nul && set PY_CMD=py -3
if "%PY_CMD%"=="" (
    where python >nul 2>nul && set PY_CMD=python
)
if "%PY_CMD%"=="" (
    echo.
    echo =====================================================
    echo  Python was not found on this computer.
    echo.
    echo  Install Python 3.10 or newer from:
    echo      https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: on the first installer screen, tick
    echo  "Add python.exe to PATH" before clicking Install.
    echo.
    echo  Then double-click this file again.
    echo =====================================================
    echo.
    pause
    exit /b 1
)

REM ---- create the virtual environment on first run -------------------
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating Python environment in %VENV_DIR% ...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo ERROR: could not create the Python environment.
        pause
        exit /b 1
    )
)

set VPY=%VENV_DIR%\Scripts\python.exe

REM ---- install dependencies once -------------------------------------
if not exist "%STAMP%" (
    echo.
    echo Installing dependencies. This happens only once.
    echo Please leave this window open.
    echo.
    "%VPY%" install_deps.py --auto
    if errorlevel 1 goto installfail
    echo done > "%STAMP%"
)

REM ---- run -----------------------------------------------------------
"%VPY%" -u quantify_epicardium.py %*
set EXITCODE=%errorlevel%
echo.
if not "%EXITCODE%"=="0" (
    echo The tool exited with an error ^(code %EXITCODE%^).
    echo Please send the text above to Max.
)
pause
exit /b %EXITCODE%

:installfail
echo.
echo =====================================================
echo  Dependency installation failed.
echo  Check your internet connection and try again.
echo  If it keeps failing, send this window's text to Max.
echo =====================================================
echo.
pause
exit /b 1
