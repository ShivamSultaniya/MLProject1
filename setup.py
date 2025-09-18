#!/usr/bin/env python3
"""
Setup script for Concentration Tracker
"""

import os
import subprocess
import sys
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Detect architecture
    arch = platform.machine().lower()
    is_arm64 = arch in ['aarch64', 'arm64']
    
    if is_arm64:
        print("ARM64 architecture detected - using compatible requirements...")
        requirements_file = "requirements-arm64.txt"
        
        # Install system dependencies for ARM64
        print("Note: For ARM64 systems, you may need to install system dependencies:")
        print("sudo apt-get update")
        print("sudo apt-get install cmake libopenblas-dev liblapack-dev libatlas-base-dev gfortran")
        print("")
    else:
        requirements_file = "requirements.txt"
    
    try:
        # Try to install MediaPipe first (will fail on ARM64)
        if not is_arm64:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe>=0.10.0"], 
                                    capture_output=True)
                print("✓ MediaPipe installed successfully")
            except subprocess.CalledProcessError:
                print("⚠ MediaPipe installation failed, will use OpenCV fallback")
        
        # Install main requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install packages: {e}")
        print("You may need to install system dependencies manually.")
        if is_arm64:
            print("For ARM64 systems, try:")
            print("sudo apt-get install cmake libopenblas-dev liblapack-dev")
            print("pip install dlib --no-cache-dir")
        return False

def create_data_directory():
    """Create data directory for logs and screenshots"""
    data_dir = "data"
    frames_dir = os.path.join(data_dir, "frames")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    print("✓ Data directories created")

def test_camera():
    """Test if camera is available"""
    print("Testing camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ Camera test successful")
                return True
            else:
                print("⚠ Camera detected but failed to read frame")
                return False
        else:
            print("⚠ No camera detected (you can still run in no-display mode)")
            return False
    except ImportError:
        print("⚠ OpenCV not installed, skipping camera test")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    required_modules = ['cv2', 'numpy', 'yaml']
    optional_modules = ['mediapipe', 'dlib']
    
    success = True
    
    # Test required modules
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {module}: {e}")
            success = False
    
    # Test optional modules
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError:
            print(f"⚠ {module} not available (will use fallback)")
    
    return success

def main():
    """Main setup function"""
    print("Concentration Tracker Setup")
    print("=" * 30)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install requirements
    if success and not install_requirements():
        success = False
    
    # Create directories
    if success:
        create_data_directory()
    
    # Test imports
    if success and not test_imports():
        success = False
    
    # Test camera
    if success:
        test_camera()
    
    print("\n" + "=" * 30)
    if success:
        print("✓ Setup completed successfully!")
        print("\nYou can now run the concentration tracker:")
        print("  python main.py")
        print("\nFor help:")
        print("  python main.py --help")
    else:
        print("✗ Setup failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
