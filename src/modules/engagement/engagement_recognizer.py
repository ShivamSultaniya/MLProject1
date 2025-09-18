"""
Engagement Recognition System

Basic implementation of engagement recognition from facial expressions.
This is a simplified version that can be extended with more sophisticated models.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging


class EngagementRecognizer:
    """
    Basic engagement recognition system using facial features.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize engagement recognizer.
        
        Args:
            model_path: Path to trained engagement model (optional)
        """
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # For now, use a rule-based approach
        # In future, this would load a trained deep learning model
        self.model = None
        
        # Engagement thresholds (these would be learned from data)
        self.engagement_thresholds = {
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }
    
    def recognize_engagement(self, frame: np.ndarray, 
                           face_landmarks: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Recognize engagement level from facial features.
        
        Args:
            frame: Input video frame
            face_landmarks: Optional facial landmarks
            
        Returns:
            Dictionary with engagement analysis results
        """
        result = {
            'engagement_level': 'medium',
            'engagement_score': 0.5,
            'confidence': 0.6,
            'features': {}
        }
        
        if face_landmarks is not None and len(face_landmarks) >= 68:
            # Extract engagement-related features
            features = self._extract_engagement_features(face_landmarks)
            result['features'] = features
            
            # Calculate engagement score using rule-based approach
            engagement_score = self._calculate_engagement_score(features)
            result['engagement_score'] = engagement_score
            
            # Determine engagement level
            result['engagement_level'] = self._determine_engagement_level(engagement_score)
            
            # Calculate confidence based on feature reliability
            result['confidence'] = self._calculate_confidence(features)
        
        return result
    
    def _extract_engagement_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Extract features related to engagement from facial landmarks."""
        features = {}
        
        # Eye openness (related to attention)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_eye_openness = self._calculate_eye_openness(left_eye)
        right_eye_openness = self._calculate_eye_openness(right_eye)
        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
        
        features['eye_openness'] = avg_eye_openness
        
        # Mouth features (can indicate interest/boredom)
        mouth_landmarks = landmarks[48:68]
        mouth_openness = self._calculate_mouth_openness(mouth_landmarks)
        features['mouth_openness'] = mouth_openness
        
        # Eyebrow position (can indicate attention/concentration)
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]
        eyebrow_raise = self._calculate_eyebrow_raise(left_eyebrow, right_eyebrow, left_eye, right_eye)
        features['eyebrow_raise'] = eyebrow_raise
        
        # Face symmetry (can indicate head orientation/attention)
        face_symmetry = self._calculate_face_symmetry(landmarks)
        features['face_symmetry'] = face_symmetry
        
        return features
    
    def _calculate_eye_openness(self, eye_landmarks: np.ndarray) -> float:
        """Calculate eye openness ratio."""
        if len(eye_landmarks) < 6:
            return 0.5
        
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        if h == 0:
            return 0.5
        
        # Eye aspect ratio
        ear = (v1 + v2) / (2.0 * h)
        
        return ear
    
    def _calculate_mouth_openness(self, mouth_landmarks: np.ndarray) -> float:
        """Calculate mouth openness ratio."""
        if len(mouth_landmarks) < 12:
            return 0.0
        
        # Vertical distance (mouth height)
        mouth_height = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[9])
        
        # Horizontal distance (mouth width)
        mouth_width = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
        
        if mouth_width == 0:
            return 0.0
        
        # Mouth aspect ratio
        mar = mouth_height / mouth_width
        
        return mar
    
    def _calculate_eyebrow_raise(self, left_eyebrow: np.ndarray, right_eyebrow: np.ndarray,
                               left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """Calculate eyebrow raise indicator."""
        if len(left_eyebrow) < 5 or len(right_eyebrow) < 5:
            return 0.0
        
        # Distance between eyebrow and eye
        left_distance = np.mean([np.linalg.norm(left_eyebrow[i] - left_eye[i]) 
                                for i in range(min(len(left_eyebrow), len(left_eye)))])
        
        right_distance = np.mean([np.linalg.norm(right_eyebrow[i] - right_eye[i]) 
                                 for i in range(min(len(right_eyebrow), len(right_eye)))])
        
        avg_distance = (left_distance + right_distance) / 2
        
        # Normalize (this would be calibrated with training data)
        normalized_distance = min(1.0, avg_distance / 30.0)
        
        return normalized_distance
    
    def _calculate_face_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate face symmetry score."""
        if len(landmarks) < 68:
            return 0.5
        
        # Get face center (nose tip)
        nose_tip = landmarks[30]
        
        # Calculate distances from left and right face points to center
        left_points = landmarks[0:9]  # Left side of face
        right_points = landmarks[8:17]  # Right side of face
        
        left_distances = [np.linalg.norm(point - nose_tip) for point in left_points]
        right_distances = [np.linalg.norm(point - nose_tip) for point in right_points]
        
        # Calculate symmetry as inverse of distance difference
        distance_diffs = [abs(l - r) for l, r in zip(left_distances, right_distances)]
        avg_diff = np.mean(distance_diffs)
        
        # Convert to symmetry score (higher is more symmetric)
        symmetry = max(0, 1.0 - avg_diff / 50.0)  # 50 is normalization factor
        
        return symmetry
    
    def _calculate_engagement_score(self, features: Dict[str, float]) -> float:
        """Calculate overall engagement score from features."""
        score = 0.0
        
        # Eye openness contributes to engagement (alert vs drowsy)
        eye_openness = features.get('eye_openness', 0.3)
        if eye_openness > 0.25:  # Above blink threshold
            score += 0.3
        
        # Eyebrow raise can indicate interest/attention
        eyebrow_raise = features.get('eyebrow_raise', 0.0)
        if eyebrow_raise > 0.3:
            score += 0.2
        
        # Face symmetry indicates frontal orientation (attention)
        face_symmetry = features.get('face_symmetry', 0.5)
        score += face_symmetry * 0.3
        
        # Mouth features (neutral mouth indicates focus)
        mouth_openness = features.get('mouth_openness', 0.0)
        if 0.02 < mouth_openness < 0.1:  # Slight opening indicates engagement
            score += 0.2
        
        return min(1.0, score)
    
    def _determine_engagement_level(self, engagement_score: float) -> str:
        """Determine categorical engagement level from score."""
        if engagement_score >= self.engagement_thresholds['high']:
            return 'high'
        elif engagement_score >= self.engagement_thresholds['medium']:
            return 'medium'
        elif engagement_score >= self.engagement_thresholds['low']:
            return 'low'
        else:
            return 'disengaged'
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in engagement assessment."""
        # Simple confidence based on feature availability and reasonableness
        confidence = 0.6  # Base confidence
        
        # Higher confidence if features are in reasonable ranges
        eye_openness = features.get('eye_openness', 0.3)
        if 0.15 < eye_openness < 0.6:
            confidence += 0.1
        
        face_symmetry = features.get('face_symmetry', 0.5)
        if face_symmetry > 0.4:
            confidence += 0.1
        
        return min(1.0, confidence)

