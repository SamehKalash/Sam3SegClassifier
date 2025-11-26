# Quick Setup for SAM3 on Raspberry Pi 5 (PowerShell)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "SAM3 Setup for Raspberry Pi 5" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Note: This is for testing on Windows. 
# For actual Pi 5 deployment, use setup.sh on the Pi.

Write-Host ""
Write-Host "Checking Python version..." -ForegroundColor Yellow
python --version

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyTorch
Write-Host ""
Write-Host "Installing PyTorch..." -ForegroundColor Yellow
pip install torch torchvision torchaudio

# Install other requirements
Write-Host ""
Write-Host "Installing other dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "To get started:" -ForegroundColor Yellow
Write-Host "  1. Activate the environment: .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run a test: python test_image_basic.py"
Write-Host ""
Write-Host "Available tests:" -ForegroundColor Yellow
Write-Host "  - test_image_basic.py      : Image segmentation"
Write-Host "  - test_tracker.py          : Interactive segmentation"
Write-Host "  - test_video_streaming.py  : Video (streaming mode)"
Write-Host "  - test_video_full.py       : Video (batch mode)"
