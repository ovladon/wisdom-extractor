@echo off
REM ============================================================
REM  Wisdom Extractor - one-click launcher for Windows
REM  First run: installs everything (needs internet, ~5 minutes).
REM  Later runs: starts in seconds. A browser tab opens by itself.
REM  Close this black window to stop the app.
REM ============================================================
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python is not installed yet. Trying automatic install via winget...
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  if errorlevel 1 (
    echo.
    echo Automatic install failed. Please install Python manually:
    echo   1. Open  https://www.python.org/downloads/windows/
    echo   2. Download "Python 3.12", run it, and TICK "Add python.exe to PATH"
    echo   3. Double-click this file again.
    pause
    exit /b 1
  )
  echo Python installed. Please close this window and double-click run_windows.bat again.
  pause
  exit /b 0
)
if not exist ".venv\Scripts\python.exe" (
  echo First-time setup: creating environment and installing packages...
  py -3 -m venv .venv || (echo venv creation failed & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip -q
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo package install failed & pause & exit /b 1)
)
echo Starting the Wisdom Extractor... a browser tab will open shortly.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
