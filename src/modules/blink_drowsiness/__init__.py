"""
Blink and Drowsiness Detection Module

This module provides real-time blink detection and drowsiness estimation
using eye aspect ratio (EAR) and machine learning models trained on
ZJU Eyeblink and NTHU-DDD datasets.
"""

from .blink_detector import BlinkDetector
from .drowsiness_detector import DrowsinessDetector
from .models import BlinkCNN, DrowsinessLSTM, EARCalculator
from .utils import calculate_ear, detect_blink_sequence, estimate_drowsiness_level
from .dataset import ZJUEyeblinkDataset, NTHUDDDDataset

__all__ = [
    'BlinkDetector',
    'DrowsinessDetector', 
    'BlinkCNN',
    'DrowsinessLSTM',
    'EARCalculator',
    'calculate_ear',
    'detect_blink_sequence',
    'estimate_drowsiness_level',
    'ZJUEyeblinkDataset',
    'NTHUDDDDataset'
]
