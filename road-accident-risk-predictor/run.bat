@echo off
title Road Accident Risk Prediction System
echo ========================================================
echo   Starting Road Accident Risk Prediction System
echo ========================================================
echo.

cd /d "%~dp0"

if exist .venv\Scripts\activate.bat (
    echo [1/2] Activating Python environment...
    call .venv\Scripts\activate.bat
) else (
    echo [1/2] Setting up Python environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing dependencies (this happens only once)...
    pip install -r requirements.txt
)

echo [2/2] Launching Dashboard in your browser...
echo.
streamlit run app\streamlit_app.py
pause
