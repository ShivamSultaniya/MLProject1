# Installation Guide

This guide will help you set up the Multi-Modal Concentration Analysis System on your machine.

## Prerequisites

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (16GB recommended)
- **Storage**: At least 5GB free space
- **Camera**: Webcam or external USB camera
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster inference

### Hardware Recommendations

- **CPU**: Intel i5/AMD Ryzen 5 or better
- **GPU**: NVIDIA GTX 1060 or better (optional but recommended)
- **Camera**: HD webcam (720p minimum, 1080p recommended)

## Installation Methods

### Method 1: Quick Start (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/concentration-analysis.git
   cd concentration-analysis
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**:
   ```bash
   pip install -e .
   ```

### Method 2: Development Installation

For developers who want to contribute or modify the code:

1. **Clone and setup**:
   ```bash
   git clone https://github.com/username/concentration-analysis.git
   cd concentration-analysis
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Method 3: Docker Installation

1. **Build Docker image**:
   ```bash
   docker build -t concentration-analysis .
   ```

2. **Run with camera access**:
   ```bash
   # Linux
   docker run --rm -it --device=/dev/video0 concentration-analysis
   
   # Windows (requires additional setup)
   docker run --rm -it concentration-analysis
   ```

## GPU Support (Optional)

### CUDA Installation

1. **Install NVIDIA drivers** (latest version recommended)

2. **Install CUDA Toolkit** (11.6 or compatible):
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
   - Follow installation instructions for your OS

3. **Install PyTorch with CUDA support**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
   ```

4. **Verify GPU support**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")
   ```

## Additional Dependencies

### Face Detection Models

The system uses dlib for face detection and landmark extraction. You may need to download additional model files:

1. **Download dlib face predictor**:
   ```bash
   # Create models directory
   mkdir -p models/dlib
   
   # Download shape predictor (68 landmarks)
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   mv shape_predictor_68_face_landmarks.dat models/dlib/
   ```

2. **Alternative: Use MediaPipe** (no additional downloads needed):
   - The system can use MediaPipe instead of dlib
   - Set `face_detector_backend: mediapipe` in your config

### Pre-trained Models (Optional)

For better performance, you can download pre-trained models:

```bash
# Create models directory structure
mkdir -p models/{gaze,blink,pose,engagement}

# Download pre-trained models (if available)
# python scripts/download_models.py
```

## Verification

### Test Installation

1. **Run system test**:
   ```bash
   python -c "
   from src.integration import ConcentrationAnalyzer
   analyzer = ConcentrationAnalyzer()
   print('Installation successful!')
   "
   ```

2. **Test camera access**:
   ```bash
   python -c "
   import cv2
   cap = cv2.VideoCapture(0)
   if cap.isOpened():
       print('Camera access successful!')
       cap.release()
   else:
       print('Camera access failed!')
   "
   ```

3. **Run quick demo**:
   ```bash
   python demo_realtime.py --camera 0 --modules blink pose --fps 15
   ```

## Troubleshooting

### Common Issues

#### 1. OpenCV Installation Issues

**Problem**: OpenCV fails to install or import
**Solution**:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.6.0.66
```

#### 2. dlib Compilation Errors

**Problem**: dlib fails to compile
**Solutions**:
- **Windows**: Install Visual Studio Build Tools
- **macOS**: Install Xcode command line tools: `xcode-select --install`
- **Linux**: Install build essentials: `sudo apt-get install build-essential cmake`
- **Alternative**: Use conda: `conda install -c conda-forge dlib`

#### 3. Camera Not Detected

**Problem**: Camera not accessible
**Solutions**:
- Check camera permissions in system settings
- Try different camera IDs (0, 1, 2, etc.)
- On Linux, add user to video group: `sudo usermod -a -G video $USER`

#### 4. CUDA/GPU Issues

**Problem**: GPU not detected or CUDA errors
**Solutions**:
- Verify NVIDIA drivers: `nvidia-smi`
- Check CUDA installation: `nvcc --version`
- Reinstall PyTorch with correct CUDA version
- Use CPU-only version if GPU is not available

#### 5. Memory Issues

**Problem**: Out of memory errors
**Solutions**:
- Reduce batch size in config
- Lower input resolution
- Disable some modules
- Use model quantization

#### 6. Performance Issues

**Problem**: Low FPS or slow processing
**Solutions**:
- Use GPU acceleration
- Reduce input resolution
- Lower target FPS
- Enable model optimization
- Use lighter models (e.g., MobileNet instead of ResNet)

### Platform-Specific Issues

#### Windows
- Install Microsoft Visual C++ Redistributable
- Use Anaconda/Miniconda for easier dependency management
- May need Windows SDK for some packages

#### macOS
- Install Homebrew for system dependencies
- May need to install additional codecs for video processing
- Use conda for packages with C++ dependencies

#### Linux
- Install system dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip
  sudo apt-get install libopencv-dev python3-opencv
  sudo apt-get install cmake build-essential
  ```

## Configuration

### Basic Configuration

Create a configuration file or modify the default:

```yaml
# config.yaml
global:
  device: auto  # or 'cuda' for GPU
  max_fps: 15.0  # Adjust based on your hardware
  
# Enable only modules you need
gaze:
  enabled: false  # Disable if no pre-trained model
blink:
  enabled: true
  method: ear  # Works without trained models
pose:
  enabled: true
engagement:
  enabled: false  # Disable if no pre-trained model
```

### Performance Tuning

For optimal performance:

1. **Hardware-based settings**:
   ```yaml
   # For low-end hardware
   global:
     max_fps: 10.0
     input_resolution: [320, 240]
   
   # For high-end hardware with GPU
   global:
     max_fps: 30.0
     device: cuda
     input_resolution: [640, 480]
   ```

2. **Module-specific tuning**:
   ```yaml
   blink:
     history_length: 15  # Reduce for less memory usage
   
   pose:
     smoothing_window: 3  # Reduce for faster response
   
   engagement:
     temporal_window: 15  # Reduce for less memory usage
   ```

## Next Steps

After successful installation:

1. **Run the demo**: `python demo_realtime.py`
2. **Read the User Guide**: Check `docs/USER_GUIDE.md`
3. **Explore examples**: Look in the `examples/` directory
4. **Train custom models**: See `docs/TRAINING.md`
5. **API documentation**: Check `docs/API.md`

## Getting Help

If you encounter issues:

1. **Check the FAQ**: `docs/FAQ.md`
2. **Search existing issues**: GitHub Issues
3. **Create new issue**: Provide system info, error logs, and steps to reproduce
4. **Join discussions**: GitHub Discussions for questions and feature requests

## Contributing

See `CONTRIBUTING.md` for guidelines on contributing to the project.
