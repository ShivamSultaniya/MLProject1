"""
Eye Gaze Estimation Module

This module provides eye gaze estimation capabilities using deep learning models
trained on MPIIGaze and GazeCapture datasets.
"""

from .gaze_estimator import GazeEstimator
from .data_loader import GazeDataLoader
from .model import GazeNet
from .utils import preprocess_eye_region, calculate_gaze_vector

__all__ = ['GazeEstimator', 'GazeDataLoader', 'GazeNet', 'preprocess_eye_region', 'calculate_gaze_vector']


