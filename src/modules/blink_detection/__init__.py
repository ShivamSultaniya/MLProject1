"""
Blink and Drowsiness Detection Module

This module provides real-time blink detection and drowsiness analysis
using computer vision techniques and machine learning models.
"""

from .blink_detector import BlinkDetector
from .drowsiness_analyzer import DrowsinessAnalyzer
from .ear_calculator import calculate_eye_aspect_ratio
from .features import extract_blink_features

__all__ = ['BlinkDetector', 'DrowsinessAnalyzer', 'calculate_eye_aspect_ratio', 'extract_blink_features']


