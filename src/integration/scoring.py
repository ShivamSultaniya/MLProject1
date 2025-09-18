"""
Concentration Scoring System

Converts fused multi-modal scores into final concentration assessments.
"""

import numpy as np
from typing import Dict, Any, List
from collections import deque
import logging


class ConcentrationScorer:
    """
    System for computing final concentration scores from fused inputs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize concentration scorer.
        
        Args:
            config: Scoring configuration
        """
        self.config = config
        self.temporal_smoothing = config.get('temporal_smoothing', True)
        self.smoothing_window = config.get('smoothing_window', 10)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        
        # Alert thresholds
        self.alert_thresholds = config.get('alert_thresholds', {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        })
        
        # Score history for smoothing
        self.score_history = deque(maxlen=self.smoothing_window)
        
        self.logger = logging.getLogger(__name__)
    
    def compute_concentration_score(self, fused_results: Dict[str, Any]) -> float:
        """
        Compute final concentration score from fused results.
        
        Args:
            fused_results: Results from fusion engine
            
        Returns:
            Concentration score (0-1)
        """
        base_score = fused_results.get('fused_score', 0.5)
        confidence = fused_results.get('confidence', 0.5)
        
        # Apply confidence weighting
        if confidence < self.confidence_threshold:
            # Lower confidence -> more conservative score
            base_score = base_score * 0.7 + 0.5 * 0.3
        
        # Apply temporal smoothing
        if self.temporal_smoothing:
            base_score = self._apply_smoothing(base_score)
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, base_score))
        
        return final_score
    
    def _apply_smoothing(self, current_score: float) -> float:
        """Apply temporal smoothing to concentration scores."""
        self.score_history.append(current_score)
        
        if len(self.score_history) == 1:
            return current_score
        
        # Weighted moving average with recent scores having higher weight
        weights = np.exp(np.linspace(-1, 0, len(self.score_history)))
        weights = weights / np.sum(weights)
        
        smoothed_score = np.sum(np.array(self.score_history) * weights)
        
        return smoothed_score


