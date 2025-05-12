@echo off
echo ================================
echo Creating virtual environment...
echo ================================
python -m venv gb_env

echo.
echo ================================
echo Activating environment...
echo ================================
call gb_env\Scripts\activate

echo.
echo ================================
echo Installing required packages...
echo ================================
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ================================
echo Launching Grain Misorientation App...
echo ================================
python Grain_misorientation.py

pause
