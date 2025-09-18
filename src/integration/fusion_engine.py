"""
Multi-Modal Fusion Engine

Combines outputs from different concentration analysis components
using various fusion strategies.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging


class MultiModalFusion:
    """
    Engine for fusing multiple modality scores into unified assessment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fusion engine.
        
        Args:
            config: Fusion configuration
        """
        self.config = config
        self.method = config.get('method', 'weighted_average')
        self.weights = config.get('weights', {
            'gaze_focus': 0.3,
            'alertness': 0.3,
            'head_stability': 0.2,
            'engagement': 0.2
        })
        self.temporal_smoothing = config.get('temporal_smoothing', True)
        self.smoothing_window = config.get('smoothing_window', 10)
        
        # History for temporal smoothing
        self.score_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def fuse_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Fuse individual component scores.
        
        Args:
            scores: Dictionary of component scores
            
        Returns:
            Fused scores dictionary
        """
        if self.method == 'weighted_average':
            return self._weighted_average_fusion(scores)
        elif self.method == 'neural_fusion':
            return self._neural_fusion(scores)
        else:
            self.logger.warning(f"Unknown fusion method: {self.method}")
            return self._weighted_average_fusion(scores)
    
    def _weighted_average_fusion(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Weighted average fusion strategy."""
        # Normalize weights
        total_weight = sum(self.weights.get(key, 0) for key in scores.keys())
        if total_weight == 0:
            total_weight = 1.0
        
        # Calculate weighted sum
        fused_score = 0.0
        confidence = 0.0
        
        for component, score in scores.items():
            if score is not None:
                weight = self.weights.get(component, 0) / total_weight
                fused_score += score * weight
                confidence += weight  # Confidence based on available components
        
        # Apply temporal smoothing if enabled
        if self.temporal_smoothing:
            fused_score = self._apply_temporal_smoothing(fused_score)
        
        return {
            'fused_score': fused_score,
            'confidence': confidence,
            'components': scores
        }
    
    def _neural_fusion(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Neural network-based fusion (placeholder for future implementation)."""
        # For now, fallback to weighted average
        # In future, this could use a trained neural network
        return self._weighted_average_fusion(scores)
    
    def _apply_temporal_smoothing(self, current_score: float) -> float:
        """Apply temporal smoothing to scores."""
        self.score_history.append(current_score)
        
        # Maintain history window
        if len(self.score_history) > self.smoothing_window:
            self.score_history.pop(0)
        
        # Exponential moving average
        if len(self.score_history) == 1:
            return current_score
        
        alpha = 0.3  # Smoothing factor
        smoothed_score = current_score
        
        for i in range(len(self.score_history) - 2, -1, -1):
            weight = alpha * (1 - alpha) ** (len(self.score_history) - 1 - i)
            smoothed_score += weight * self.score_history[i]
        
        return smoothed_score


