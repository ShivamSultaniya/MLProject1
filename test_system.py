#!/usr/bin/env python3
"""
Simple system test for the concentration analysis project.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Test face detection
        from modules.utils.face_detector import FaceDetector
        print("✅ Face detector import: OK")
        
        # Test individual modules
        from modules.eye_gaze.gaze_estimator import GazeEstimator
        print("✅ Gaze estimator import: OK")
        
        from modules.blink_drowsiness.blink_detector import BlinkDetector
        print("✅ Blink detector import: OK")
        
        from modules.head_pose.pose_estimator import HeadPoseEstimator
        print("✅ Head pose estimator import: OK")
        
        from modules.engagement.engagement_detector import EngagementDetector
        print("✅ Engagement detector import: OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_camera():
    """Test camera access."""
    print("\n📷 Testing camera access...")
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera access: OK (Resolution: {frame.shape[1]}x{frame.shape[0]})")
            cap.release()
            return True
        else:
            print("❌ Camera access: Failed to read frame")
            cap.release()
            return False
    else:
        print("❌ Camera access: Failed to open camera")
        return False

def test_basic_detection():
    """Test basic face detection."""
    print("\n👤 Testing face detection...")
    
    try:
        # Import face detector
        from modules.utils.face_detector import FaceDetector
        
        # Create detector
        detector = FaceDetector(backend='mediapipe')
        
        # Create a test image (simple colored rectangle)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Try detection (won't find faces in solid color image, but shouldn't crash)
        faces = detector.detect_faces(test_image)
        print(f"✅ Face detection: OK (Found {len(faces)} faces in test image)")
        
        return True
        
    except Exception as e:
        print(f"❌ Face detection error: {e}")
        return False

def test_blink_detection():
    """Test basic blink detection functionality."""
    print("\n😴 Testing blink detection...")
    
    try:
        from modules.blink_drowsiness.blink_detector import BlinkDetector
        
        # Create detector with EAR method (doesn't require trained models)
        detector = BlinkDetector(method='ear')
        
        # Create test image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Try detection
        result = detector.detect_blink(test_image)
        print(f"✅ Blink detection: OK (EAR method working)")
        
        return True
        
    except Exception as e:
        print(f"❌ Blink detection error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Multi-Modal Concentration Analysis System Test\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Camera Access", test_camera),
        ("Face Detection", test_basic_detection),
        ("Blink Detection", test_blink_detection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: Exception - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("-"*50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: conda activate mlproj")
        print("2. Try: python demo_realtime.py --modules blink pose --fps 15")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the errors above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
