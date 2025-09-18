"""
Quick Start Script for Multi-Modal Concentration Analysis System

This script sets up the environment and runs a basic demonstration
of the concentration analysis system using webcam input.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['cv2', 'numpy', 'torch', 'yaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logging.error(f"Missing required packages: {missing_packages}")
        logging.info("Please install dependencies: pip install -r requirements.txt")
        return False
    
    return True

def setup_basic_environment():
    """Setup basic environment for quick start."""
    # Create necessary directories
    directories = ['models', 'data', 'logs', 'outputs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Create basic config if it doesn't exist
    config_path = Path('configs/default_config.yaml')
    if not config_path.exists():
        config_path.parent.mkdir(exist_ok=True)
        
        basic_config = """
# Basic Configuration for Quick Start
gaze:
  device: "cpu"
  method: "gazenet"

blink:
  method: "ear"
  ear_threshold: 0.25

drowsiness:
  perclos_threshold: 0.8
  analysis_window: 60.0

head_pose:
  method: "pnp"
  smoothing_window: 5

fusion:
  method: "weighted_average"
  weights:
    gaze_focus: 0.3
    alertness: 0.3
    head_stability: 0.2
    engagement: 0.2

display:
  show_visualization: true
  show_fps: true
"""
        
        with open(config_path, 'w') as f:
            f.write(basic_config)
        
        logging.info(f"Created basic config at {config_path}")

def run_demo():
    """Run the concentration analysis demo."""
    try:
        # Import and run the main system
        from main import run_real_time_analysis, get_default_config
        
        logging.info("Starting Multi-Modal Concentration Analysis Demo")
        logging.info("Press 'q' to quit, 's' to save session data")
        logging.info("Make sure your webcam is connected and working")
        
        # Use default config for demo
        config = get_default_config()
        
        # Run real-time analysis
        run_real_time_analysis(config, video_source=0, duration=None)
        
    except ImportError as e:
        logging.error(f"Import error: {e}")
        logging.error("Make sure all source files are present")
        return False
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        return False
    
    return True

def main():
    """Main function for quick start."""
    print("=" * 60)
    print("Multi-Modal Concentration Analysis System - Quick Start")
    print("=" * 60)
    
    setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        logging.error("Dependency check failed. Please install required packages.")
        return 1
    
    # Setup environment
    setup_basic_environment()
    
    # Ask user if they want to proceed
    try:
        response = input("\nReady to start concentration analysis demo? (y/n): ").lower().strip()
        if response != 'y':
            print("Demo cancelled.")
            return 0
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return 0
    
    # Run demo
    success = run_demo()
    
    if success:
        print("\nDemo completed successfully!")
        print("Check the logs/ directory for detailed logs.")
        print("Session data has been saved automatically.")
    else:
        print("\nDemo encountered issues. Check the logs for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


