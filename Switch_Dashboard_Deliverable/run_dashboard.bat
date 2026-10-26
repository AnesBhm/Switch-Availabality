@echo off
title Switch Availability Dashboard
echo ============================================================
echo   Switch Availability Dashboard
echo   Starting server at http://127.0.0.1:8050
echo ============================================================
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No .venv found — using system Python.
)

REM Install / update dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

echo.
echo Starting dashboard...
python app.py

pause
