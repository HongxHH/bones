@echo off
setlocal
cd /d "%~dp0"

echo.
echo [Prism] Full restart: control plane + Lab UI + players...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0prism\scripts\restart_ui.ps1"
set ERR=%ERRORLEVEL%

echo.
if not "%ERR%"=="0" (
  echo Failed, exit code %ERR%
) else (
  echo Done.
)
echo.
pause
exit /b %ERR%
