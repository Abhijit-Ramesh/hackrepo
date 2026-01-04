@echo off
REM Quick start script for Delhi Flood Risk System (Windows)

echo ========================================
echo Delhi Flood Risk Prediction System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo [2/3] Training machine learning model...
python train_model.py
if %errorlevel% neq 0 (
    echo ERROR: Model training failed
    pause
    exit /b 1
)
echo.

echo [3/3] Opening interactive map...
echo.
echo Starting local web server on http://localhost:8000
echo Open your browser and navigate to:
echo.
echo     http://localhost:8000/index.html
echo.
echo Press Ctrl+C to stop the server when done.
echo.

python -m http.server 8000

pause
