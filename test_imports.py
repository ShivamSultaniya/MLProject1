"""
Test script to verify all imports work correctly.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def test_imports():
    """Test all critical imports."""
    
    print("Testing imports...")
    
    try:
        print("✅ Basic imports...")
        import cv2
        import numpy as np
        import yaml
        print(f"   OpenCV: {cv2.__version__}")
        print(f"   NumPy: {np.__version__}")
        print(f"   PyYAML: {yaml.__version__}")
        
        print("✅ Core modules...")
        from modules.eye_gaze import GazeEstimator
        from modules.blink_detection import BlinkDetector, DrowsinessAnalyzer  
        from modules.head_pose import HeadPoseEstimator
        from modules.engagement import EngagementRecognizer
        print("   All module imports successful")
        
        print("✅ Integration system...")
        from integration.concentration_analyzer import ConcentrationAnalyzer
        from integration.fusion_engine import MultiModalFusion
        from integration.scoring import ConcentrationScorer
        print("   Integration imports successful")
        
        print("✅ Real-time interface...")
        from real_time.webcam_interface import WebcamInterface
        from real_time.gui_interface import GUIInterface
        print("   Real-time imports successful")
        
        print("\n🎉 ALL IMPORTS SUCCESSFUL!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality."""
    
    print("\nTesting basic functionality...")
    
    try:
        print("✅ Creating test analyzer...")
        from integration.concentration_analyzer import ConcentrationAnalyzer
        
        config = {
            'gaze': {'device': 'cpu'},
            'blink': {'method': 'ear'},
            'head_pose': {'method': 'pnp'},
            'fusion': {'method': 'weighted_average'},
            'scoring': {'temporal_smoothing': True}
        }
        
        analyzer = ConcentrationAnalyzer(config)
        print("   Analyzer created successfully")
        
        print("✅ Testing with dummy frame...")
        import numpy as np
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        metrics = analyzer.analyze_frame(dummy_frame)
        print(f"   Got metrics: score={metrics.overall_score:.2f}, level={metrics.attention_level}")
        
        print("\n🎉 BASIC FUNCTIONALITY TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("IMPORT AND FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n✅ ALL TESTS PASSED!")
            print("You can now run: python run_analysis.py")
            return 0
        else:
            print("\n⚠️ Imports OK, but functionality test failed")
            return 1
    else:
        print("\n❌ Import test failed")
        print("Please check your Python environment and dependencies")
        return 1


if __name__ == "__main__":
    exit(main())
