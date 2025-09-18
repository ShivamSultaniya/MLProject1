"""
Engagement Recognition Module

This module provides facial engagement recognition capabilities
using deep learning models trained on DAiSEE dataset.
"""

from .engagement_recognizer import EngagementRecognizer
from .models import EngagementNet

__all__ = ['EngagementRecognizer', 'EngagementNet']

