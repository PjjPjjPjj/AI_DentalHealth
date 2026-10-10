@echo off
REM Dental AI Assistant - Windows build script (run from anywhere).
REM Needs Python 3.10 installed with "Add to PATH" (or the py launcher).

setlocal EnableDelayedExpansion
cd /d "%~dp0.."
echo [1/5] Repo root: %CD%

set "PY="
py -3.10 --version >nul 2>nul && set "PY=py -3.10"
if not defined PY (
    python --version >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python 3.10 not found. Install from python.org and tick "Add python.exe to PATH".
    goto :err
)
echo [2/5] Python found:
!PY! --version

if exist venv\Scripts\python.exe (
    echo [3/5] Reusing existing venv\
) else (
    echo [3/5] Creating venv ...
    !PY! -m venv venv || goto :err
)

echo [4/5] Installing dependencies - first run downloads ~300 MB, please wait ...
venv\Scripts\python.exe -m pip install --upgrade pip || goto :err
venv\Scripts\python.exe -m pip install -r app\requirements_app.txt pyinstaller || goto :err

echo [5/5] Building exe (2-5 minutes) ...
venv\Scripts\python.exe -m PyInstaller app\DentalAI.spec --noconfirm || goto :err
echo.
echo ============================================================
echo  Build OK:  dist\DentalAI\DentalAI.exe
echo ============================================================
pause
exit /b 0

:err
echo.
echo Build FAILED - see the last error above.
pause
exit /b 1
