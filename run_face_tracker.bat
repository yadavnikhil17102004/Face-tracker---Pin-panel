@echo off
echo Starting Face Tracker Application...
echo Press 'q' to quit when the application is running

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

:: Run the face tracker
echo Starting face tracker...
python face_tracker.py

pause