# Multi-Modal Concentration Analysis System

A comprehensive system for real-time concentration analysis using computer vision techniques. The system integrates four core modules to provide multi-modal representation of user concentration and engagement.

## Overview

This project combines multiple visual modalities to assess user concentration:
- **Eye Gaze Estimation**: Tracks where user attention is directed
- **Blink & Drowsiness Detection**: Measures fatigue and disengagement through blink patterns
- **Head Pose Estimation**: Analyzes attentional shifts through head orientation
- **Engagement Recognition**: Infers cognitive involvement from facial expressions

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Webcam Input  │───▶│  Frame Processor │───▶│ Multi-Modal     │
└─────────────────┘    └──────────────────┘    │ Analysis Engine │
                                               └─────────────────┘
                                                        │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 │                                 │
                       ▼                                 ▼                                 ▼
              ┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
              │   Eye Gaze      │                │   Head Pose     │                │   Engagement    │
              │   Estimation    │                │   Estimation    │                │   Recognition   │
              └─────────────────┘                └─────────────────┘                └─────────────────┘
                       │                                 │                                 │
                       └─────────────────────────────────┼─────────────────────────────────┘
                                                         │
                                               ┌─────────────────┐
                                               │ Blink/Drowsiness│
                                               │   Detection     │
                                               └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │ Concentration   │
                                               │ Score & Feedback│
                                               └─────────────────┘
```

## Datasets

The system is trained and evaluated on established benchmark datasets:

- **Eye Gaze**: MPIIGaze, GazeCapture
- **Blink/Drowsiness**: ZJU Eyeblink Dataset, NTHU-DDD
- **Head Pose**: BIWI Kinect Head Pose Dataset
- **Engagement**: DAiSEE Dataset

## Features

- Real-time webcam processing
- Multi-modal concentration analysis
- Robust model training on benchmark datasets
- Cross-domain evaluation capabilities
- Visual feedback system
- Modular architecture for easy extension

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Project

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (optional)
python scripts/download_models.py
```

## Quick Start

```python
from concentration_analyzer import ConcentrationAnalyzer

# Initialize the analyzer
analyzer = ConcentrationAnalyzer()

# Start real-time analysis
analyzer.start_realtime_analysis()
```

## Project Structure

```
Project/
├── src/
│   ├── modules/
│   │   ├── eye_gaze/          # Eye gaze estimation
│   │   ├── blink_drowsiness/  # Blink and drowsiness detection
│   │   ├── head_pose/         # Head pose estimation
│   │   └── engagement/        # Engagement recognition
│   ├── data/
│   │   ├── loaders/           # Dataset loading utilities
│   │   └── preprocessing/     # Data preprocessing
│   ├── training/              # Training scripts
│   ├── evaluation/            # Evaluation framework
│   └── integration/           # System integration
├── models/                    # Pre-trained models
├── data/                      # Dataset storage
├── configs/                   # Configuration files
├── scripts/                   # Utility scripts
└── notebooks/                 # Jupyter notebooks for analysis
```

## Training

Each module can be trained independently:

```bash
# Train eye gaze model
python src/training/train_eye_gaze.py --config configs/eye_gaze_config.yaml

# Train blink detection model
python src/training/train_blink_detection.py --config configs/blink_config.yaml

# Train head pose model
python src/training/train_head_pose.py --config configs/head_pose_config.yaml

# Train engagement model
python src/training/train_engagement.py --config configs/engagement_config.yaml
```

## Evaluation

```bash
# Run comprehensive evaluation
python src/evaluation/evaluate_system.py

# Individual module evaluation
python src/evaluation/evaluate_module.py --module eye_gaze
```

## Real-time Usage

```bash
# Start real-time concentration analysis
python src/integration/realtime_analyzer.py

# With custom configuration
python src/integration/realtime_analyzer.py --config configs/realtime_config.yaml
```

## Configuration

All modules are configurable through YAML files in the `configs/` directory. Key parameters include:

- Model architectures and hyperparameters
- Dataset paths and preprocessing options
- Real-time processing settings
- Visualization preferences

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{multimodal_concentration_2024,
  title={Multi-Modal Concentration Analysis System},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/username/concentration-analysis}}
}
```

## Acknowledgments

- MPIIGaze and GazeCapture datasets for eye gaze estimation
- ZJU Eyeblink and NTHU-DDD datasets for drowsiness detection
- BIWI Kinect dataset for head pose estimation
- DAiSEE dataset for engagement recognition
