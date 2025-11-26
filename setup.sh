#!/bin/bash
# Quick setup script for SAM3 on Raspberry Pi 5

echo "======================================"
echo "SAM3 Setup for Raspberry Pi 5"
echo "======================================"

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for Raspberry Pi (CPU version)
echo ""
echo "Installing PyTorch (CPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "======================================"
echo "✅ Setup complete!"
echo "======================================"
echo ""
echo "To get started:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run a test: python test_image_basic.py"
echo ""
echo "Available tests:"
echo "  - test_image_basic.py      : Image segmentation"
echo "  - test_tracker.py          : Interactive segmentation"
echo "  - test_video_streaming.py  : Video (streaming mode)"
echo "  - test_video_full.py       : Video (batch mode)"
