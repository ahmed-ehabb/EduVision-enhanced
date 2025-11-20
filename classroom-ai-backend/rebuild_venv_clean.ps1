#!/usr/bin/env pwsh
# ================================
# CLEAN VIRTUAL ENVIRONMENT REBUILD
# ================================

Write-Host "CLASSROOM AI - CLEAN VENV REBUILD" -ForegroundColor Yellow
Write-Host "====================================" -ForegroundColor Yellow

# Step 1: Deactivate current environment
Write-Host "`n1. Deactivating current virtual environment..." -ForegroundColor Green
try {
    deactivate 2>$null
} catch {
    Write-Host "   No active environment to deactivate" -ForegroundColor Gray
}

# Step 2: Remove old virtual environments
Write-Host "`n2. Removing old virtual environments..." -ForegroundColor Green

$venvDirs = @("venv", "venv_new", ".venv")
foreach ($dir in $venvDirs) {
    if (Test-Path $dir) {
        Write-Host "   Removing: $dir" -ForegroundColor Red
        Remove-Item -Recurse -Force $dir
    } else {
        Write-Host "   $dir not found (already clean)" -ForegroundColor Gray
    }
}

# Step 3: Create fresh virtual environment
Write-Host "`n3. Creating fresh virtual environment..." -ForegroundColor Green
python -m venv venv_clean
if (-not $?) {
    Write-Host "Failed to create virtual environment!" -ForegroundColor Red
    exit 1
}
Write-Host "   Virtual environment created: venv_clean" -ForegroundColor Gray

# Step 4: Activate new environment
Write-Host "`n4. Activating new virtual environment..." -ForegroundColor Green
& ".\venv_clean\Scripts\Activate.ps1"
if (-not $?) {
    Write-Host "Failed to activate virtual environment!" -ForegroundColor Red
    exit 1
}
Write-Host "   Virtual environment activated" -ForegroundColor Gray

# Step 5: Upgrade pip
Write-Host "`n5. Upgrading pip to latest version..." -ForegroundColor Green
python -m pip install --upgrade pip
Write-Host "   Pip upgraded" -ForegroundColor Gray

# Step 6: Install wheel and setuptools first
Write-Host "`n6. Installing build tools..." -ForegroundColor Green
pip install wheel setuptools
Write-Host "   Build tools installed" -ForegroundColor Gray

# Step 7: Install requirements in batches to avoid conflicts
Write-Host "`n7. Installing dependencies in optimized order..." -ForegroundColor Green

Write-Host "   Installing Core API Framework..." -ForegroundColor Cyan
pip install fastapi==0.115.12 uvicorn==0.34.2 python-multipart==0.0.20 pydantic==2.10.5

Write-Host "   Installing Scientific Computing..." -ForegroundColor Cyan
pip install numpy==1.24.3 pandas==2.3.0 scipy==1.15.2 scikit-learn==1.4.1.post1

Write-Host "   Installing PyTorch (GPU support)..." -ForegroundColor Cyan
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu121

Write-Host "   Installing Transformers & HuggingFace..." -ForegroundColor Cyan
pip install transformers==4.38.2 sentence-transformers==2.5.1 huggingface-hub==0.31.1 datasets==3.1.0

Write-Host "   Installing Audio Processing..." -ForegroundColor Cyan
pip install soundfile==0.13.1 librosa==0.10.1 noisereduce==3.0.2

Write-Host "   Installing Computer Vision..." -ForegroundColor Cyan
pip install opencv-python==4.9.0.80 mediapipe==0.10.11 face-recognition==1.3.0

Write-Host "   Installing NLP Tools..." -ForegroundColor Cyan
pip install nltk==3.9.1 spacy==3.7.4 deepmultilingualpunctuation==1.0.1

Write-Host "   Installing remaining dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# Step 8: Verify installation
Write-Host "`n8. Verifying critical dependencies..." -ForegroundColor Green

$criticalPackages = @(
    "torch", "transformers", "soundfile", "opencv-python", 
    "fastapi", "deepmultilingualpunctuation"
)

foreach ($package in $criticalPackages) {
    try {
        $version = pip show $package | Select-String "Version:" | ForEach-Object { $_.ToString().Split(":")[1].Trim() }
        Write-Host "   $package : $version" -ForegroundColor Green
    } catch {
        Write-Host "   $package : NOT INSTALLED" -ForegroundColor Red
    }
}

# Step 9: Test ASR module
Write-Host "`n9. Testing ASR module loading..." -ForegroundColor Green
python -c "
try:
    from backend.asr_module import load_asr
    asr = load_asr()
    if asr and asr.is_available():
        print('   ASR module loads successfully')
    else:
        print('   ASR module loaded but not available')
except Exception as e:
    print(f'   ASR module failed: {e}')
"

Write-Host "`nVIRTUAL ENVIRONMENT REBUILD COMPLETE!" -ForegroundColor Yellow
Write-Host "=================================" -ForegroundColor Yellow
Write-Host "To activate this environment in the future, run:" -ForegroundColor White
Write-Host ".\venv_clean\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "`nTo start the backend:" -ForegroundColor White
Write-Host "python run_backend_complete.py" -ForegroundColor Cyan 