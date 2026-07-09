@echo off
REM GeoSense one-click launcher (Windows). First run creates a venv and installs deps.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ============================================================
  echo  First run: creating environment and installing dependencies
  echo  ^(this downloads PyTorch CPU + deps, a few minutes^)
  echo ============================================================
  python -m venv .venv
  call ".venv\Scripts\activate.bat"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call ".venv\Scripts\activate.bat"
)

python run.py
pause
