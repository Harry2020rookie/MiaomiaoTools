@echo off
setlocal
cd /d "%~dp0"

set "PYTHONNOUSERSITE=1"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

"%PYTHON_EXE%" -m PyInstaller --noconfirm --clean build_windows.spec
if errorlevel 1 exit /b 1

echo.
echo Built: dist\妙妙工具.exe
endlocal
