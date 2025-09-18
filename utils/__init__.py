"""
Utilities package for concentration tracker
"""

try:
    from .face_detector import FaceDetector
except ImportError:
    from .opencv_face_detector import OpenCVFaceDetector as FaceDetector

from .concentration_analyzer import ConcentrationAnalyzer
from .data_analyzer import ConcentrationDataAnalyzer

__all__ = ['FaceDetector', 'ConcentrationAnalyzer', 'ConcentrationDataAnalyzer']
