@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Missing repository virtual environment. See docs\research-desktop.md.
  pause
  exit /b 1
)
start "Research Control Center" ".venv\Scripts\pythonw.exe" -m iqoption_research.desktop.app
