@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the Python environment first. See CHAT.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m transformerx.chat %*
if errorlevel 1 (
  echo.
  echo Chat did not start or encountered an error. See the message above and CHAT.md.
  pause
  exit /b 1
)
