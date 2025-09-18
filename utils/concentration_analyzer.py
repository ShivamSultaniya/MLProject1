"""
Concentration Analysis Module
Analyzes concentration levels based on eye tracking and face detection
"""

import numpy as np
import time
from collections import deque
from typing import Dict, List, Tuple
import logging


class ConcentrationAnalyzer:
    """Analyzes concentration based on various factors"""
    
    def __init__(self, 
                 time_window: int = 30,
                 blink_threshold: float = 0.25,
                 drowsiness_threshold: float = 0.3,
                 alert_threshold: float = 0.4,
                 good_concentration_threshold: float = 0.7):
        """
        Initialize concentration analyzer
        
        Args:
            time_window: Time window in seconds for analysis
            blink_threshold: EAR threshold for blink detection
            drowsiness_threshold: EAR threshold for drowsiness detection
            alert_threshold: Concentration score threshold for alerts
            good_concentration_threshold: Threshold for good concentration
        """
        self.time_window = time_window
        self.blink_threshold = blink_threshold
        self.drowsiness_threshold = drowsiness_threshold
        self.alert_threshold = alert_threshold
        self.good_concentration_threshold = good_concentration_threshold
        
        # Data storage for analysis
        self.ear_history = deque(maxlen=time_window * 30)  # Assuming 30 FPS
        self.blink_history = deque(maxlen=time_window * 30)
        self.face_detection_history = deque(maxlen=time_window * 30)
        self.timestamps = deque(maxlen=time_window * 30)
        
        # Blink detection state
        self.consecutive_frames_below_threshold = 0
        self.blink_count = 0
        self.last_blink_reset_time = time.time()
        
        # Concentration metrics
        self.current_concentration_score = 1.0
        self.concentration_history = deque(maxlen=100)
        
        self.logger = logging.getLogger(__name__)
    
    def update(self, ear_left: float, ear_right: float, face_detected: bool) -> Dict:
        """
        Update concentration analysis with new data
        
        Args:
            ear_left: Eye aspect ratio for left eye
            ear_right: Eye aspect ratio for right eye
            face_detected: Whether face was detected in frame
            
        Returns:
            Dictionary containing analysis results
        """
        current_time = time.time()
        avg_ear = (ear_left + ear_right) / 2.0
        
        # Store data
        self.ear_history.append(avg_ear)
        self.face_detection_history.append(face_detected)
        self.timestamps.append(current_time)
        
        # Detect blinks
        blink_detected = self._detect_blink(avg_ear)
        self.blink_history.append(blink_detected)
        
        # Calculate concentration score
        concentration_score = self._calculate_concentration_score(
            avg_ear, face_detected, current_time
        )
        
        self.current_concentration_score = concentration_score
        self.concentration_history.append(concentration_score)
        
        # Generate analysis results
        results = {
            'concentration_score': concentration_score,
            'concentration_level': self._get_concentration_level(concentration_score),
            'blink_detected': blink_detected,
            'blink_rate': self._calculate_blink_rate(current_time),
            'avg_ear': avg_ear,
            'face_detected': face_detected,
            'drowsiness_detected': avg_ear < self.drowsiness_threshold and face_detected,
            'attention_status': self._get_attention_status(concentration_score, face_detected)
        }
        
        return results
    
    def _detect_blink(self, avg_ear: float) -> bool:
        """Detect blinks based on EAR threshold"""
        if avg_ear < self.blink_threshold:
            self.consecutive_frames_below_threshold += 1
        else:
            if self.consecutive_frames_below_threshold >= 2:  # Minimum frames for blink
                self.blink_count += 1
                self.consecutive_frames_below_threshold = 0
                return True
            self.consecutive_frames_below_threshold = 0
        return False
    
    def _calculate_blink_rate(self, current_time: float) -> float:
        """Calculate blinks per minute"""
        if current_time - self.last_blink_reset_time >= 60:  # Reset every minute
            blink_rate = self.blink_count
            self.blink_count = 0
            self.last_blink_reset_time = current_time
            return blink_rate
        
        # Calculate current rate based on elapsed time
        elapsed_minutes = (current_time - self.last_blink_reset_time) / 60.0
        if elapsed_minutes > 0:
            return self.blink_count / elapsed_minutes
        return 0.0
    
    def _calculate_concentration_score(self, avg_ear: float, face_detected: bool, current_time: float) -> float:
        """
        Calculate concentration score based on multiple factors
        
        Returns:
            Concentration score between 0.0 and 1.0
        """
        if not face_detected:
            return 0.0
        
        # Get recent data within time window
        recent_data = self._get_recent_data(current_time)
        
        if not recent_data['ear_values']:
            return 1.0
        
        # Factor 1: Eye aspect ratio stability (less variation = better concentration)
        ear_stability = self._calculate_ear_stability(recent_data['ear_values'])
        
        # Factor 2: Face detection consistency
        face_consistency = self._calculate_face_consistency(recent_data['face_detected'])
        
        # Factor 3: Blink rate (normal blink rate indicates good concentration)
        blink_rate_score = self._calculate_blink_rate_score(recent_data['blinks'])
        
        # Factor 4: Drowsiness detection
        drowsiness_score = self._calculate_drowsiness_score(recent_data['ear_values'])
        
        # Combine factors with weights
        concentration_score = (
            ear_stability * 0.3 +
            face_consistency * 0.3 +
            blink_rate_score * 0.2 +
            drowsiness_score * 0.2
        )
        
        return np.clip(concentration_score, 0.0, 1.0)
    
    def _get_recent_data(self, current_time: float) -> Dict:
        """Get data within the time window"""
        cutoff_time = current_time - self.time_window
        
        recent_ear = []
        recent_face = []
        recent_blinks = []
        
        for i, timestamp in enumerate(self.timestamps):
            if timestamp >= cutoff_time:
                if i < len(self.ear_history):
                    recent_ear.append(self.ear_history[i])
                if i < len(self.face_detection_history):
                    recent_face.append(self.face_detection_history[i])
                if i < len(self.blink_history):
                    recent_blinks.append(self.blink_history[i])
        
        return {
            'ear_values': recent_ear,
            'face_detected': recent_face,
            'blinks': recent_blinks
        }
    
    def _calculate_ear_stability(self, ear_values: List[float]) -> float:
        """Calculate EAR stability score"""
        if len(ear_values) < 2:
            return 1.0
        
        # Lower standard deviation indicates better stability
        std_dev = np.std(ear_values)
        stability_score = max(0.0, 1.0 - (std_dev * 4))  # Scale factor
        return stability_score
    
    def _calculate_face_consistency(self, face_detected_values: List[bool]) -> float:
        """Calculate face detection consistency score"""
        if not face_detected_values:
            return 0.0
        
        consistency = sum(face_detected_values) / len(face_detected_values)
        return consistency
    
    def _calculate_blink_rate_score(self, blink_values: List[bool]) -> float:
        """Calculate blink rate score (normal rate is good)"""
        if not blink_values:
            return 1.0
        
        blink_count = sum(blink_values)
        # Normal blink rate is 15-20 blinks per minute
        # Convert to expected blinks in our time window
        expected_blinks = (self.time_window / 60.0) * 17.5  # Average normal rate
        
        if blink_count == 0:
            return 0.5  # No blinks might indicate staring or drowsiness
        
        ratio = blink_count / expected_blinks
        # Score is highest when ratio is close to 1
        score = max(0.0, 1.0 - abs(1.0 - ratio))
        return score
    
    def _calculate_drowsiness_score(self, ear_values: List[float]) -> float:
        """Calculate drowsiness score based on EAR values"""
        if not ear_values:
            return 1.0
        
        # Count frames below drowsiness threshold
        drowsy_frames = sum(1 for ear in ear_values if ear < self.drowsiness_threshold)
        drowsiness_ratio = drowsy_frames / len(ear_values)
        
        # Lower drowsiness ratio = higher score
        score = 1.0 - drowsiness_ratio
        return score
    
    def _get_concentration_level(self, score: float) -> str:
        """Get concentration level description"""
        if score >= self.good_concentration_threshold:
            return "Excellent"
        elif score >= 0.5:
            return "Good"
        elif score >= self.alert_threshold:
            return "Fair"
        else:
            return "Poor"
    
    def _get_attention_status(self, score: float, face_detected: bool) -> str:
        """Get attention status message"""
        if not face_detected:
            return "No face detected - Please look at the camera"
        elif score < self.alert_threshold:
            return "Low concentration detected - Take a break!"
        elif score >= self.good_concentration_threshold:
            return "Great concentration! Keep it up!"
        else:
            return "Maintaining focus..."
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        if not self.concentration_history:
            return {}
        
        recent_scores = list(self.concentration_history)[-100:]  # Last 100 measurements
        
        return {
            'current_score': self.current_concentration_score,
            'average_score': np.mean(recent_scores),
            'min_score': np.min(recent_scores),
            'max_score': np.max(recent_scores),
            'score_trend': self._calculate_trend(recent_scores)
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate concentration trend"""
        if len(scores) < 10:
            return "Insufficient data"
        
        recent = np.mean(scores[-10:])
        earlier = np.mean(scores[-20:-10]) if len(scores) >= 20 else np.mean(scores[:-10])
        
        diff = recent - earlier
        if diff > 0.05:
            return "Improving"
        elif diff < -0.05:
            return "Declining"
        else:
            return "Stable"

