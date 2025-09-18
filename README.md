# Concentration Tracker Using Webcam

A real-time concentration monitoring system that uses computer vision and machine learning to track user attention and focus levels through webcam analysis.

## Features

- **Real-time Face Detection**: Uses MediaPipe for accurate face detection and landmark tracking
- **Eye Tracking**: Monitors eye aspect ratio (EAR) for blink detection and drowsiness analysis
- **Concentration Scoring**: Advanced algorithm that calculates concentration levels based on multiple factors
- **Visual Feedback**: Real-time display with concentration metrics and alerts
- **Data Logging**: Comprehensive logging of concentration data and session statistics
- **Configurable Settings**: YAML-based configuration for easy customization
- **Session Management**: Track concentration over time with trend analysis

## How It Works

The system analyzes several key factors to determine concentration levels:

1. **Face Detection Consistency**: Tracks whether the user is looking at the camera
2. **Eye Aspect Ratio (EAR)**: Monitors eye openness to detect blinks and drowsiness
3. **Blink Rate Analysis**: Normal blink rates indicate good concentration
4. **Eye Movement Stability**: Less eye movement variation suggests better focus
5. **Temporal Analysis**: Analyzes patterns over configurable time windows

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam (built-in or external)
- Operating System: Windows, macOS, or Linux (including ARM64/aarch64)

### Architecture Support

- **x86_64**: Full support with MediaPipe for advanced face detection
- **ARM64/aarch64**: Compatible with OpenCV-based face detection (fallback mode)

### Setup

1. **Clone or download the project**:
   ```bash
   cd concentration_tracker
   ```

2. **Run the setup script** (recommended):
   ```bash
   python setup.py
   ```

   Or **install dependencies manually**:
   ```bash
   # For x86_64 systems
   pip install -r requirements.txt
   
   # For ARM64 systems (if setup.py fails)
   sudo apt-get install cmake libopenblas-dev liblapack-dev
   pip install -r requirements-arm64.txt
   ```

3. **Verify installation** (optional):
   ```bash
   python -c "import cv2, numpy; print('Core dependencies installed successfully!')"
   ```

## Usage

### Basic Usage

Run the concentration tracker with default settings:

```bash
python main.py
```

### Advanced Usage

Run with custom configuration:

```bash
python main.py --config config/custom_settings.yaml
```

Run without video display (headless mode):

```bash
python main.py --no-display
```

### Keyboard Controls

While the application is running:

- **`q`**: Quit the application
- **`r`**: Reset tracking data and restart session
- **`s`**: Save screenshot with current metrics

## Configuration

The system uses a YAML configuration file (`config/settings.yaml`) for customization:

### Camera Settings
```yaml
camera:
  device_id: 0        # Camera device ID
  width: 640          # Frame width
  height: 480         # Frame height
  fps: 30            # Frames per second
```

### Detection Thresholds
```yaml
face_detection:
  confidence_threshold: 0.7

eye_tracking:
  blink_threshold: 0.25
  drowsiness_threshold: 0.3
  eye_aspect_ratio_threshold: 0.25
```

### Concentration Analysis
```yaml
concentration:
  time_window: 30                    # Analysis window in seconds
  alert_threshold: 0.4               # Low concentration alert
  good_concentration_threshold: 0.7  # Good concentration threshold
```

## Project Structure

```
concentration_tracker/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── config/
│   └── settings.yaml      # Configuration file
├── src/
│   ├── __init__.py
│   └── concentration_tracker.py  # Main application logic
├── utils/
│   ├── __init__.py
│   ├── face_detector.py   # Face detection utilities
│   └── concentration_analyzer.py  # Concentration analysis
├── models/                # Directory for ML models (future use)
└── data/                  # Logs and saved data
    ├── concentration_log.txt
    └── frames/            # Saved screenshots
```

## Understanding the Metrics

### Concentration Score (0.0 - 1.0)

- **0.9 - 1.0**: Excellent concentration
- **0.7 - 0.9**: Good concentration  
- **0.4 - 0.7**: Fair concentration
- **0.0 - 0.4**: Poor concentration (alert triggered)

### Concentration Levels

- **Excellent**: Optimal focus and attention
- **Good**: Maintaining good concentration
- **Fair**: Acceptable but could improve
- **Poor**: Low concentration detected

### Eye Aspect Ratio (EAR)

- Higher values indicate more open eyes
- Sudden drops indicate blinks
- Consistently low values may indicate drowsiness

### Blink Rate

- Normal: 15-20 blinks per minute
- Too high: May indicate stress or fatigue
- Too low: May indicate intense focus or drowsiness

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check if camera is connected and not used by other applications
   - Try different `device_id` values (0, 1, 2, etc.) in config

2. **Poor face detection**:
   - Ensure good lighting conditions
   - Position face clearly in front of camera
   - Adjust `confidence_threshold` in config

3. **Inaccurate concentration scores**:
   - Allow 30-60 seconds for calibration
   - Adjust thresholds in configuration file
   - Ensure stable camera position

4. **Performance issues**:
   - Reduce camera resolution in config
   - Lower FPS settings
   - Close other resource-intensive applications

### Error Messages

- **"Could not open camera"**: Camera device not available
- **"Config file not found"**: Using default settings
- **"No face detected"**: User not visible to camera

## Technical Details

### Dependencies

- **OpenCV**: Computer vision and camera handling
- **MediaPipe**: Face detection and landmark tracking
- **NumPy**: Numerical computations
- **PyYAML**: Configuration file parsing
- **Pandas**: Data analysis (optional)
- **Matplotlib**: Plotting (optional)

### Algorithms

1. **Face Detection**: MediaPipe Face Mesh for 468 facial landmarks
2. **Eye Aspect Ratio**: Geometric calculation based on eye landmarks
3. **Blink Detection**: Threshold-based EAR analysis with temporal filtering
4. **Concentration Scoring**: Weighted combination of multiple factors:
   - EAR stability (30%)
   - Face detection consistency (30%)
   - Blink rate normality (20%)
   - Drowsiness indicators (20%)

## Future Enhancements

- [ ] Machine learning model for personalized concentration detection
- [ ] Head pose estimation for attention direction
- [ ] Integration with productivity applications
- [ ] Mobile app version
- [ ] Multi-user support
- [ ] Advanced analytics and reporting
- [ ] Biometric integration (heart rate, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or contributions, please:

1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information

## Acknowledgments

- MediaPipe team for face detection technology
- OpenCV community for computer vision tools
- Contributors to the open-source libraries used

---

**Note**: This system is designed for productivity and focus monitoring. It does not store or transmit personal video data - all processing is done locally on your device.
