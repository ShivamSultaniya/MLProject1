#!/bin/bash

# Installation script for macOS using conda
echo "🚀 Installing Multi-Modal Concentration Analysis System with Conda"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Activate conda base environment
echo "📦 Initializing conda..."
eval "$(conda shell.bash hook)"

# Check if mlproj environment already exists
if conda env list | grep -q "mlproj"; then
    echo "⚠️  Environment 'mlproj' already exists. Removing it..."
    conda env remove -n mlproj -y
fi

# Create environment from yml file
echo "🔧 Creating conda environment 'mlproj'..."
conda env create -f environment.yml

# Activate the environment
echo "🔄 Activating conda environment..."
conda activate mlproj

# Verify installation
echo "✅ Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')

# Test core packages
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'PyTorch import error: {e}')

try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'OpenCV import error: {e}')

try:
    import mediapipe
    print(f'MediaPipe: {mediapipe.__version__}')
except ImportError as e:
    print(f'MediaPipe import error: {e}')

try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except ImportError as e:
    print(f'NumPy import error: {e}')
"

echo ""
echo "🎉 Installation completed!"
echo ""
echo "To activate the environment, run:"
echo "conda activate mlproj"
echo ""
echo "To test the system, run:"
echo "python demo_realtime.py --modules blink pose --fps 15"
echo ""
echo "For more information, see INSTALLATION.md"
