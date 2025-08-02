@echo off
echo ===============================================
echo    Advanced ATM Security System Launcher
echo ===============================================
echo.

REM Check if models directory exists
if not exist "models\" (
    echo Creating models directory...
    mkdir models
)

echo Downloading DNN face detection models...
python models\download_models.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Failed to download DNN models.
    echo The ATM Security System will run using the Haar cascade face detector instead.
    echo.
    pause
)

REM Check if logs directory exists
if not exist "logs\" (
    echo Creating logs directory...
    mkdir logs
)

echo.
echo Starting Advanced ATM Security System...
echo.
echo Control Keys:
echo - Press 'q' to quit
echo - Press 's' to take a screenshot
echo - Press 'h' to toggle help overlay
echo - Press 'k' to toggle keypad
echo - Press 'a' to display analytics report
echo.

python atm_security_prototype.py

echo.
echo ATM Security System has stopped.
pause
