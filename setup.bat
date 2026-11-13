@echo off
echo ============================================
echo  NIDS - Network Intrusion Detection System
echo  Auto Setup Script for Windows
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Install from https://python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/4] Installing dependencies...
pip install flask flask-socketio python-socketio werkzeug scikit-learn pandas numpy joblib geoip2 eventlet shap --quiet

echo [3/4] Creating required directories...
if not exist "data"   mkdir data
if not exist "models" mkdir models

echo [4/4] Setup complete!
echo.
echo ============================================
echo  NEXT STEPS:
echo  1. Place CICIDS2017_subset.csv in data\
echo  2. Place GeoLite2-City.mmdb in this folder
echo  3. Run: python train.py
echo  4. Run: python app.py
echo  5. Open: http://localhost:5000
echo ============================================
echo.
pause
