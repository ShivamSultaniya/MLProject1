#!/usr/bin/env python3
"""
Simple demo script to test concentration tracker without camera
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from concentration_tracker import ConcentrationTracker
    from utils.opencv_face_detector import OpenCVFaceDetector
    from utils.concentration_analyzer import ConcentrationAnalyzer
    
    print("✓ Concentration Tracker Demo")
    print("=" * 30)
    
    # Test face detector initialization
    print("Testing face detector...")
    face_detector = OpenCVFaceDetector()
    print("✓ Face detector initialized")
    
    # Test concentration analyzer
    print("Testing concentration analyzer...")
    analyzer = ConcentrationAnalyzer()
    print("✓ Concentration analyzer initialized")
    
    # Test a sample analysis
    print("Testing sample analysis...")
    results = analyzer.update(0.3, 0.3, True)  # Sample EAR values
    print(f"✓ Sample concentration score: {results['concentration_score']:.2f}")
    print(f"✓ Concentration level: {results['concentration_level']}")
    print(f"✓ Status: {results['attention_status']}")
    
    print("\n" + "=" * 30)
    print("✓ All components working correctly!")
    print("\nTo run the full tracker:")
    print("  python main.py                    # With camera display")
    print("  python main.py --no-display       # Without display")
    print("  python main.py --config custom.yaml  # With custom config")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
