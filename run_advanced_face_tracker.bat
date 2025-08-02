@echo off
echo Starting Advanced Face Tracker Application...
echo Controls:
echo   'q' - Quit the application
echo   's' - Take a screenshot
echo   'h' - Toggle help text
echo   'e' - Toggle eye detection

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b
)

:: Check if requirements are installed
echo Checking dependencies...
pip install -r requirements.txt

:: Create screenshots directory if it doesn't exist
if not exist screenshots mkdir screenshots

:: Run the advanced face tracker
echo Starting advanced face tracker...
python advanced_face_tracker.py

pause