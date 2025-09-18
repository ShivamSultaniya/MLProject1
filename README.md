# Multi-Modal Concentration Analysis System

A real-time system that integrates four core modules to provide comprehensive concentration assessment through visual analysis.

## 🎯 Overview

This project combines multiple computer vision techniques to analyze user concentration through:

1. **Eye Gaze Estimation** - Determines where attention is directed
2. **Blink & Drowsiness Detection** - Measures fatigue and disengagement  
3. **Head Pose Estimation** - Analyzes attentional shifts and distractions
4. **Engagement Recognition** - Infers cognitive involvement from facial expressions

## 🏗️ System Architecture

```
├── src/
│   ├── modules/
│   │   ├── eye_gaze/          # Eye gaze estimation (MPIIGaze/GazeCapture)
│   │   ├── blink_detection/   # Blink and drowsiness detection (ZJU/NTHU-DDD)
│   │   ├── head_pose/         # Head pose estimation (BIWI Kinect)
│   │   └── engagement/        # Engagement recognition (DAiSEE)
│   ├── integration/           # Multi-modal fusion and scoring
│   ├── real_time/            # Webcam interface and GUI
│   └── main.py               # Main entry point
├── configs/                   # Configuration files
├── models/                   # Trained model weights
├── data/                     # Dataset storage
├── scripts/                  # Utility scripts
└── requirements.txt          # Dependencies
```

## 📊 Datasets

- **Eye Gaze**: MPIIGaze, GazeCapture
- **Blink/Drowsiness**: ZJU Eyeblink, NTHU-DDD  
- **Head Pose**: BIWI Kinect Head Pose Dataset
- **Engagement**: DAiSEE

## ✨ Features

- **Real-time Processing**: Live webcam analysis with 30 FPS
- **Multi-modal Fusion**: Combines all four modalities for robust assessment
- **Comprehensive Metrics**: Overall concentration score, component scores, attention levels
- **Visual Feedback**: Real-time overlay with scores and recommendations
- **Flexible Configuration**: YAML-based configuration system
- **Session Recording**: Export analysis data and session summaries
- **Cross-platform**: Windows, Linux, macOS support

## 🚀 Quick Start

### Prerequisites
- Python 3.10 (recommended for best compatibility)
- Webcam or video input device
- Windows/Linux/macOS

### Installation

1. **Clone or download the project**
```bash
git clone <repository-url>
cd MLproj
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup environment** (optional but recommended)
```bash
python scripts/download_models.py
```

4. **Run the system**
```bash
# Quick start demo
python quick_start.py

# Or run directly
python src/main.py
```

## 📱 Usage

### Real-time Analysis
```bash
# Default webcam (camera 0)
python src/main.py --mode realtime --input 0

# Specific camera
python src/main.py --mode realtime --input 1

# Custom configuration
python src/main.py --config configs/custom_config.yaml
```

### Batch Processing
```bash
# Process video file
python src/main.py --mode batch --input video.mp4 --output results.json
```

### Interactive Controls
- **'q'**: Quit application
- **'s'**: Save current session data
- **'r'**: Reset analysis
- **'h'**: Show help

## 📈 Output Metrics

The system provides comprehensive concentration assessment:

- **Overall Score**: 0-1 concentration level
- **Attention Level**: High/Medium/Low/Distracted
- **Component Scores**:
  - Gaze Focus (0-1)
  - Alertness (0-1) 
  - Head Stability (0-1)
  - Engagement (0-1)
- **Recommendations**: Actionable feedback
- **Confidence**: Assessment reliability

## ⚙️ Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
# Example configuration
gaze:
  device: "cpu"  # or "cuda"
  method: "gazenet"

blink:
  method: "ear"
  ear_threshold: 0.25

head_pose:
  method: "pnp"
  smoothing_window: 5

display:
  show_visualization: true
  show_fps: true
```

## 🔧 Python 3.10 Compatibility

This project is optimized for **Python 3.10** with carefully selected package versions:

- **MediaPipe**: Uses version compatible with Python 3.10
- **OpenCV**: Latest stable version with Python 3.10 support
- **PyTorch**: Compatible versions for both CPU and GPU
- **All dependencies**: Tested and verified for Python 3.10

### Migration from Python 3.13

If you're moving from Python 3.13 (Azure VM) to Python 3.10 (Windows laptop):

1. **Create new Python 3.10 environment**:
```bash
conda create -n concentration python=3.10
conda activate concentration
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import cv2, numpy, torch, mediapipe; print('All dependencies OK')"
```

## 🎛️ Advanced Features

### Custom Model Training
- Dataset loaders for all supported datasets
- Training scripts for each modality
- Model evaluation and benchmarking tools

### Multi-modal Fusion
- Weighted average fusion
- Temporal smoothing
- Confidence-based weighting

### Real-time Performance
- Multi-threaded processing
- Frame buffering
- Optimized inference pipelines

## 📝 Project Status

- ✅ **Core Modules**: All four modules implemented
- ✅ **Integration System**: Multi-modal fusion complete
- ✅ **Real-time Interface**: Webcam and GUI ready
- ✅ **Configuration System**: Flexible YAML configuration
- ✅ **Python 3.10 Compatibility**: Fully tested
- ⚠️ **Model Training**: Placeholder models (training scripts available)
- ⚠️ **Datasets**: Setup scripts provided (download separately)

## 🔮 Future Extensions

- **Physiological Sensors**: Heart rate, EEG integration
- **Audio Analysis**: Voice stress and engagement detection  
- **Multi-camera Setup**: 360-degree attention tracking
- **Cloud Integration**: Remote processing and analytics
- **Mobile App**: Smartphone-based concentration tracking

## 🛠️ Development

### Project Structure
- **Modular Design**: Each component is independent
- **Clean Interfaces**: Well-defined APIs between modules
- **Extensible**: Easy to add new modalities or algorithms
- **Configurable**: Runtime configuration without code changes

### Testing
```bash
# Run basic functionality test
python quick_start.py

# Test individual modules
python -m src.modules.eye_gaze.gaze_estimator
python -m src.modules.blink_detection.blink_detector
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

---

**Ready to analyze concentration in real-time!** 🎯

Start with `python quick_start.py` for an immediate demonstration.

