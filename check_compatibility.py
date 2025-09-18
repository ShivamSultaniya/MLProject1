"""
Compatibility Check Script for Multi-Modal Concentration Analysis System

This script verifies that all dependencies are compatible with your Python version
and system, especially for Python 3.10 on Windows.
"""

import sys
import importlib
import platform
from pathlib import Path


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ ERROR: Python 3.x required")
        return False
    
    if version.minor < 8:
        print("❌ ERROR: Python 3.8+ required")
        return False
    
    if version.minor == 13:
        print("⚠️  WARNING: Python 3.13 may have MediaPipe compatibility issues")
        print("   Recommendation: Use Python 3.10 for best compatibility")
    elif version.minor == 10:
        print("✅ GOOD: Python 3.10 is the recommended version")
    
    return True


def check_system_info():
    """Display system information."""
    print(f"\nSystem Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")


def check_dependency(package_name, import_name=None, optional=False):
    """Check if a dependency is available."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        status = "✅ OK" if not optional else "✅ OK (optional)"
        print(f"{status}: {package_name} ({version})")
        return True
    except ImportError as e:
        status = "❌ MISSING" if not optional else "⚠️  MISSING (optional)"
        print(f"{status}: {package_name} - {e}")
        return not optional  # Return False only for required dependencies


def check_core_dependencies():
    """Check core dependencies."""
    print(f"\nCore Dependencies:")
    
    required_deps = [
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('PyYAML', 'yaml'),
        ('Pillow', 'PIL'),
        ('matplotlib', 'matplotlib'),
        ('scikit-learn', 'sklearn')
    ]
    
    all_ok = True
    for package, import_name in required_deps:
        if not check_dependency(package, import_name):
            all_ok = False
    
    return all_ok


def check_optional_dependencies():
    """Check optional dependencies."""
    print(f"\nOptional Dependencies:")
    
    optional_deps = [
        ('mediapipe', 'mediapipe'),
        ('dlib', 'dlib'),
        ('torchvision', 'torchvision'),
        ('h5py', 'h5py'),
        ('scipy', 'scipy')
    ]
    
    for package, import_name in optional_deps:
        check_dependency(package, import_name, optional=True)


def check_project_structure():
    """Check if project structure is complete."""
    print(f"\nProject Structure:")
    
    required_paths = [
        'src/main.py',
        'src/modules/eye_gaze/__init__.py',
        'src/modules/blink_detection/__init__.py', 
        'src/modules/head_pose/__init__.py',
        'src/modules/engagement/__init__.py',
        'src/integration/__init__.py',
        'src/real_time/__init__.py',
        'configs/default_config.yaml',
        'requirements.txt'
    ]
    
    all_ok = True
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            print(f"✅ OK: {path_str}")
        else:
            print(f"❌ MISSING: {path_str}")
            all_ok = False
    
    return all_ok


def check_webcam():
    """Check webcam availability."""
    print(f"\nWebcam Check:")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ OK: Webcam detected and working")
                print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            else:
                print("⚠️  WARNING: Webcam detected but cannot read frames")
            cap.release()
        else:
            print("❌ ERROR: No webcam detected")
        
    except Exception as e:
        print(f"❌ ERROR: Webcam check failed - {e}")


def run_basic_test():
    """Run basic functionality test."""
    print(f"\nBasic Functionality Test:")
    
    try:
        # Test basic imports
        import numpy as np
        import cv2
        
        # Test array operations
        test_array = np.random.rand(100, 100, 3) * 255
        test_array = test_array.astype(np.uint8)
        
        # Test OpenCV operations
        gray = cv2.cvtColor(test_array, cv2.COLOR_BGR2GRAY)
        
        print("✅ OK: Basic NumPy and OpenCV operations working")
        
        # Test MediaPipe if available
        try:
            import mediapipe as mp
            print("✅ OK: MediaPipe import successful")
        except ImportError:
            print("⚠️  WARNING: MediaPipe not available")
        
        # Test PyTorch if available
        try:
            import torch
            x = torch.randn(2, 3)
            y = torch.mm(x, x.t())
            print("✅ OK: PyTorch operations working")
        except ImportError:
            print("⚠️  WARNING: PyTorch not available")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Basic functionality test failed - {e}")
        return False


def provide_recommendations():
    """Provide recommendations based on checks."""
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    version = sys.version_info
    
    if version.minor == 13:
        print("🔄 PYTHON VERSION:")
        print("   • Switch to Python 3.10 for best compatibility")
        print("   • MediaPipe has known issues with Python 3.13")
        print("   • Create new conda environment: conda create -n concentration python=3.10")
    
    print("\n📦 INSTALLATION:")
    print("   • Run: pip install -r requirements.txt")
    print("   • If MediaPipe fails, try: pip install mediapipe==0.9.3.0")
    print("   • For dlib issues on Windows: pip install cmake, then pip install dlib")
    
    print("\n🚀 QUICK START:")
    print("   • Run setup: python scripts/download_models.py")
    print("   • Test system: python quick_start.py")
    print("   • Full system: python src/main.py")
    
    print("\n💡 TROUBLESHOOTING:")
    print("   • Webcam issues: Check camera permissions and drivers")
    print("   • Import errors: Ensure all dependencies installed correctly")
    print("   • Performance: Use GPU-enabled PyTorch for better performance")


def main():
    """Main compatibility check function."""
    print("="*60)
    print("MULTI-MODAL CONCENTRATION ANALYSIS - COMPATIBILITY CHECK")
    print("="*60)
    
    # System checks
    python_ok = check_python_version()
    check_system_info()
    
    # Dependency checks
    core_ok = check_core_dependencies()
    check_optional_dependencies()
    
    # Project structure check
    structure_ok = check_project_structure()
    
    # Hardware checks
    check_webcam()
    
    # Functionality test
    basic_ok = run_basic_test()
    
    # Overall assessment
    print(f"\n" + "="*60)
    print("OVERALL ASSESSMENT:")
    print("="*60)
    
    if python_ok and core_ok and structure_ok and basic_ok:
        print("🎉 EXCELLENT: System appears fully compatible!")
        print("   You can proceed with confidence.")
    elif python_ok and core_ok and structure_ok:
        print("✅ GOOD: Core system is compatible!")
        print("   Some optional features may not work.")
    elif python_ok and core_ok:
        print("⚠️  PARTIAL: Basic functionality should work.")
        print("   Some project files may be missing.")
    else:
        print("❌ ISSUES: Several compatibility problems detected.")
        print("   Please address the issues above before proceeding.")
    
    # Provide recommendations
    provide_recommendations()


if __name__ == "__main__":
    main()

