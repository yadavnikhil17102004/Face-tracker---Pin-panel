@echo off
echo Starting ATM Security Prototype...
echo.
echo 🏧 ATM SECURITY SYSTEM FEATURES:
echo   - Real-time face detection and monitoring
echo   - Virtual keypad window for PIN entry simulation  
echo   - Security warning when multiple people detected
echo   - Automatic screenshot capture during breaches
echo   - Audio alerts for security incidents
echo.
echo Controls:
echo   'q' - Quit the application
echo   's' - Take manual screenshot
echo   'k' - Open/Close keypad window
echo   'h' - Toggle help display
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b
)

:: Check if requirements are installed
echo Checking and installing dependencies...
pip install -r requirements.txt

:: Create directories if they don't exist
if not exist screenshots mkdir screenshots
if not exist security_screenshots mkdir security_screenshots

:: Run the ATM security prototype
echo.
echo 🚀 Starting ATM Security Prototype...
echo The keypad window will open automatically.
echo Position windows side by side for best experience.
echo.
python atm_security_prototype.py

pause
