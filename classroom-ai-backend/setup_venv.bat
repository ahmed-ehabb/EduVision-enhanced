@echo off
REM ================================
REM Virtual Environment Setup Script
REM Creates venv and installs Phase 1 requirements
REM ================================

echo ============================================================
echo  Classroom AI Backend - Virtual Environment Setup
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.11+
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Create virtual environment
echo [STEP 1/5] Creating virtual environment...
if exist venv (
    echo [WARNING] venv folder already exists
    set /p "choice=Delete existing venv? (y/n): "
    if /i "%choice%"=="y" (
        echo Deleting existing venv...
        rmdir /s /q venv
        python -m venv venv
        echo [OK] New venv created
    ) else (
        echo [OK] Using existing venv
    )
) else (
    python -m venv venv
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
echo [STEP 2/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [STEP 3/5] Upgrading pip...
python -m pip install --upgrade pip
echo [OK] Pip upgraded
echo.

REM Install Phase 1 requirements
echo [STEP 4/5] Installing Phase 1 requirements...
echo This may take 10-20 minutes (downloads ~5-10 GB)
echo.
pip install -r requirements/phase1.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)
echo [OK] Phase 1 requirements installed
echo.

REM Verify installation
echo [STEP 5/5] Verifying installation...
echo.
echo Checking NumPy...
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
echo.
echo Checking PyTorch...
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
echo.
echo Checking CUDA...
python -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not Available\"}')"
echo.
echo Checking Transformers...
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
echo.

echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Add test audio file to: testing\model_tests\test_audio.wav
echo 2. Run tests: cd testing\model_tests ^&^& python test_asr.py
echo.
echo To activate this environment in the future, run:
echo   venv\Scripts\activate
echo.
pause
