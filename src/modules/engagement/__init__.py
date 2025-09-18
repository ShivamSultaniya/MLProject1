"""
Engagement Recognition Module

This module provides real-time engagement and attention analysis
using facial expression cues and behavioral patterns trained on
the DAiSEE dataset. It distinguishes between different levels of
cognitive involvement and engagement states.
"""

from .engagement_detector import EngagementDetector
from .models import EngagementCNN, EngagementTransformer, MultiModalEngagement
from .utils import extract_engagement_features, classify_engagement_level
from .utils import analyze_attention_patterns, detect_distraction_events
from .dataset import DAiSEEDataset, EngagementDataset

__all__ = [
    'EngagementDetector',
    'EngagementCNN',
    'EngagementTransformer',
    'MultiModalEngagement',
    'extract_engagement_features',
    'classify_engagement_level',
    'analyze_attention_patterns',
    'detect_distraction_events',
    'DAiSEEDataset',
    'EngagementDataset'
]
