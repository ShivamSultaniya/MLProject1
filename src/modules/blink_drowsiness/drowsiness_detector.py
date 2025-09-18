"""
Drowsiness detection system using temporal analysis of blink patterns and eye states.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import deque
import time
from enum import Enum

from .models import DrowsinessLSTM
from .blink_detector import BlinkDetector
from .utils import estimate_drowsiness_level, calculate_perclos

logger = logging.getLogger(__name__)


class DrowsinessLevel(Enum):
    """Drowsiness levels."""
    ALERT = 0
    SLIGHTLY_DROWSY = 1
    MODERATELY_DROWSY = 2
    VERY_DROWSY = 3
    EXTREMELY_DROWSY = 4


class DrowsinessDetector:
    """
    Advanced drowsiness detection system that analyzes:
    1. Blink frequency and patterns
    2. Eye closure duration (PERCLOS)
    3. Temporal sequence patterns using LSTM
    4. Statistical measures of alertness
    """
    
    def __init__(
        self,
        blink_detector: Optional[BlinkDetector] = None,
        model_path: Optional[str] = None,
        sequence_length: int = 60,  # frames to analyze
        device: str = 'auto',
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize drowsiness detector.
        
        Args:
            blink_detector: BlinkDetector instance
            model_path: Path to pre-trained LSTM model
            sequence_length: Length of temporal sequences
            device: Device for model inference
            alert_thresholds: Custom thresholds for drowsiness levels
        """
        self.sequence_length = sequence_length
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize or use provided blink detector
        self.blink_detector = blink_detector or BlinkDetector()
        
        # Initialize LSTM model
        self.lstm_model = DrowsinessLSTM(
            input_size=6,  # [EAR, blink, closure_duration, blink_rate, head_pose_x, head_pose_y]
            hidden_size=64,
            num_layers=2,
            num_classes=5  # 5 drowsiness levels
        )
        self.lstm_model.to(self.device)
        
        if model_path:
            self._load_lstm_model(model_path)
        
        # Default thresholds
        self.thresholds = alert_thresholds or {
            'perclos_mild': 0.15,      # 15% eye closure
            'perclos_moderate': 0.25,   # 25% eye closure
            'perclos_severe': 0.35,     # 35% eye closure
            'blink_rate_low': 5,        # blinks per minute
            'blink_rate_high': 25,      # blinks per minute
            'long_closure_duration': 2.0,  # seconds
            'microsleep_duration': 0.5     # seconds
        }
        
        # History buffers
        self.feature_history = deque(maxlen=sequence_length)
        self.drowsiness_history = deque(maxlen=30)  # Last 30 predictions
        self.alert_history = deque(maxlen=100)      # Longer history for trends
        
        # State tracking
        self.current_drowsiness_level = DrowsinessLevel.ALERT
        self.last_alert_time = time.time()
        self.microsleep_count = 0
        self.long_closure_count = 0
        
        # Statistics
        self.session_start_time = time.time()
        self.total_analysis_time = 0
        
        logger.info(f"DrowsinessDetector initialized on {self.device}")
    
    def _load_lstm_model(self, model_path: str):
        """Load pre-trained LSTM model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
            self.lstm_model.eval()
            logger.info(f"Loaded LSTM model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")
    
    def analyze_drowsiness(
        self, 
        frame: np.ndarray,
        head_pose: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze drowsiness from a single frame.
        
        Args:
            frame: Input video frame
            head_pose: Optional head pose (pitch, yaw) in degrees
            
        Returns:
            Dictionary containing drowsiness analysis results
        """
        current_time = time.time()
        
        # Get blink detection results
        blink_results = self.blink_detector.detect_blink(frame)
        
        # Extract features
        features = self._extract_features(blink_results, head_pose, current_time)
        
        # Add to history
        self.feature_history.append(features)
        
        # Calculate drowsiness metrics
        results = {
            'drowsiness_level': DrowsinessLevel.ALERT,
            'drowsiness_score': 0.0,
            'perclos': 0.0,
            'blink_rate': 0.0,
            'microsleep_detected': False,
            'long_closure_detected': False,
            'lstm_prediction': None,
            'features': features,
            'blink_results': blink_results,
            'alerts': []
        }
        
        try:
            # Calculate PERCLOS (percentage of eye closure)
            perclos = self._calculate_perclos()
            results['perclos'] = perclos
            
            # Get blink statistics
            blink_stats = self.blink_detector.get_blink_statistics()
            results['blink_rate'] = blink_stats['blink_rate']
            
            # LSTM-based prediction if enough history
            if len(self.feature_history) >= self.sequence_length:
                lstm_prediction = self._predict_drowsiness_lstm()
                results['lstm_prediction'] = lstm_prediction
                
                # Convert to drowsiness level
                drowsiness_level = DrowsinessLevel(int(lstm_prediction['predicted_class']))
                results['drowsiness_level'] = drowsiness_level
                results['drowsiness_score'] = lstm_prediction['confidence']
            else:
                # Fallback to rule-based analysis
                drowsiness_level, score = self._analyze_drowsiness_rules(perclos, blink_stats)
                results['drowsiness_level'] = drowsiness_level
                results['drowsiness_score'] = score
            
            # Detect specific events
            microsleep = self._detect_microsleep(blink_results)
            long_closure = self._detect_long_closure(blink_results)
            
            results['microsleep_detected'] = microsleep
            results['long_closure_detected'] = long_closure
            
            # Generate alerts
            alerts = self._generate_alerts(results)
            results['alerts'] = alerts
            
            # Update state
            self.current_drowsiness_level = results['drowsiness_level']
            self.drowsiness_history.append(results['drowsiness_score'])
            self.alert_history.append({
                'timestamp': current_time,
                'level': results['drowsiness_level'],
                'score': results['drowsiness_score'],
                'perclos': perclos
            })
            
            if microsleep:
                self.microsleep_count += 1
            if long_closure:
                self.long_closure_count += 1
            
            # Update total analysis time
            self.total_analysis_time = current_time - self.session_start_time
            
        except Exception as e:
            logger.error(f"Error in drowsiness analysis: {e}")
        
        return results
    
    def _extract_features(
        self, 
        blink_results: Dict[str, Any], 
        head_pose: Optional[Tuple[float, float]],
        timestamp: float
    ) -> np.ndarray:
        """Extract features for drowsiness analysis."""
        # Basic features
        ear = blink_results.get('ear', 0.0)
        blink = 1.0 if blink_results.get('blink_detected', False) else 0.0
        closure_duration = blink_results.get('blink_duration', 0.0)
        
        # Blink rate
        blink_stats = self.blink_detector.get_blink_statistics()
        blink_rate = blink_stats.get('blink_rate', 0.0)
        
        # Head pose features
        head_pitch = head_pose[0] if head_pose else 0.0
        head_yaw = head_pose[1] if head_pose else 0.0
        
        features = np.array([
            ear,
            blink,
            closure_duration,
            blink_rate,
            head_pitch,
            head_yaw
        ], dtype=np.float32)
        
        return features
    
    def _calculate_perclos(self, time_window: float = 60.0) -> float:
        """Calculate PERCLOS (Percentage of Eye Closure) over time window."""
        if len(self.feature_history) < 10:
            return 0.0
        
        # Get recent EAR values
        recent_features = list(self.feature_history)[-min(len(self.feature_history), int(time_window)):]
        ear_values = [f[0] for f in recent_features]  # EAR is first feature
        
        # Calculate percentage of time eyes are closed
        ear_threshold = self.blink_detector.ear_threshold
        closed_frames = sum(1 for ear in ear_values if ear < ear_threshold)
        
        perclos = closed_frames / len(ear_values) if ear_values else 0.0
        return perclos
    
    def _predict_drowsiness_lstm(self) -> Dict[str, Any]:
        """Predict drowsiness using LSTM model."""
        try:
            # Prepare sequence data
            sequence = np.array(list(self.feature_history))
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.lstm_model(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().flatten()
            }
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return {
                'predicted_class': 0,
                'confidence': 0.0,
                'probabilities': np.zeros(5)
            }
    
    def _analyze_drowsiness_rules(
        self, 
        perclos: float, 
        blink_stats: Dict[str, Any]
    ) -> Tuple[DrowsinessLevel, float]:
        """Rule-based drowsiness analysis as fallback."""
        score = 0.0
        level = DrowsinessLevel.ALERT
        
        # PERCLOS-based scoring
        if perclos >= self.thresholds['perclos_severe']:
            score += 0.4
            level = DrowsinessLevel.EXTREMELY_DROWSY
        elif perclos >= self.thresholds['perclos_moderate']:
            score += 0.3
            level = DrowsinessLevel.VERY_DROWSY
        elif perclos >= self.thresholds['perclos_mild']:
            score += 0.2
            level = DrowsinessLevel.MODERATELY_DROWSY
        
        # Blink rate analysis
        blink_rate = blink_stats.get('blink_rate', 0.0)
        if blink_rate < self.thresholds['blink_rate_low']:
            score += 0.2  # Too few blinks
        elif blink_rate > self.thresholds['blink_rate_high']:
            score += 0.1  # Too many blinks
        
        # Time since last blink
        time_since_blink = blink_stats.get('time_since_last_blink', 0.0)
        if time_since_blink > 5.0:  # No blink for 5 seconds
            score += 0.3
        
        # Adjust level based on total score
        if score >= 0.7:
            level = DrowsinessLevel.EXTREMELY_DROWSY
        elif score >= 0.5:
            level = DrowsinessLevel.VERY_DROWSY
        elif score >= 0.3:
            level = DrowsinessLevel.MODERATELY_DROWSY
        elif score >= 0.1:
            level = DrowsinessLevel.SLIGHTLY_DROWSY
        else:
            level = DrowsinessLevel.ALERT
        
        return level, min(1.0, score)
    
    def _detect_microsleep(self, blink_results: Dict[str, Any]) -> bool:
        """Detect microsleep episodes (brief involuntary sleep)."""
        blink_duration = blink_results.get('blink_duration', 0.0)
        
        # Microsleep: eye closure for 0.5-2 seconds
        if (self.thresholds['microsleep_duration'] <= blink_duration <= 
            self.thresholds['long_closure_duration']):
            return True
        
        return False
    
    def _detect_long_closure(self, blink_results: Dict[str, Any]) -> bool:
        """Detect prolonged eye closure."""
        blink_duration = blink_results.get('blink_duration', 0.0)
        
        # Long closure: eyes closed for more than 2 seconds
        if blink_duration > self.thresholds['long_closure_duration']:
            return True
        
        return False
    
    def _generate_alerts(self, results: Dict[str, Any]) -> List[str]:
        """Generate drowsiness alerts based on analysis."""
        alerts = []
        
        level = results['drowsiness_level']
        perclos = results['perclos']
        
        if level == DrowsinessLevel.EXTREMELY_DROWSY:
            alerts.append("CRITICAL: Extreme drowsiness detected!")
        elif level == DrowsinessLevel.VERY_DROWSY:
            alerts.append("WARNING: High drowsiness level!")
        elif level == DrowsinessLevel.MODERATELY_DROWSY:
            alerts.append("CAUTION: Moderate drowsiness detected")
        
        if perclos > self.thresholds['perclos_severe']:
            alerts.append(f"High eye closure rate: {perclos:.1%}")
        
        if results['microsleep_detected']:
            alerts.append("Microsleep episode detected!")
        
        if results['long_closure_detected']:
            alerts.append("Prolonged eye closure detected!")
        
        # Blink rate alerts
        blink_rate = results['blink_rate']
        if blink_rate < self.thresholds['blink_rate_low']:
            alerts.append(f"Low blink rate: {blink_rate:.1f}/min")
        elif blink_rate > self.thresholds['blink_rate_high']:
            alerts.append(f"High blink rate: {blink_rate:.1f}/min")
        
        return alerts
    
    def get_drowsiness_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drowsiness statistics."""
        current_time = time.time()
        
        # Calculate average drowsiness over different time windows
        recent_scores = list(self.drowsiness_history)
        avg_drowsiness = np.mean(recent_scores) if recent_scores else 0.0
        
        # Calculate trend
        if len(recent_scores) >= 10:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        else:
            trend = 0.0
        
        # Session statistics
        session_duration = current_time - self.session_start_time
        
        return {
            'current_level': self.current_drowsiness_level.name,
            'current_score': recent_scores[-1] if recent_scores else 0.0,
            'average_drowsiness': avg_drowsiness,
            'drowsiness_trend': trend,
            'microsleep_count': self.microsleep_count,
            'long_closure_count': self.long_closure_count,
            'session_duration': session_duration,
            'last_alert_time': self.last_alert_time,
            'perclos_current': self._calculate_perclos(),
            'blink_statistics': self.blink_detector.get_blink_statistics()
        }
    
    def reset_statistics(self):
        """Reset all drowsiness statistics."""
        self.feature_history.clear()
        self.drowsiness_history.clear()
        self.alert_history.clear()
        
        self.current_drowsiness_level = DrowsinessLevel.ALERT
        self.microsleep_count = 0
        self.long_closure_count = 0
        self.session_start_time = time.time()
        self.total_analysis_time = 0
        
        # Reset blink detector statistics
        self.blink_detector.reset_statistics()
        
        logger.info("Drowsiness statistics reset")
    
    def set_thresholds(self, thresholds: Dict[str, float]):
        """Update drowsiness detection thresholds."""
        self.thresholds.update(thresholds)
        logger.info(f"Thresholds updated: {thresholds}")
    
    def export_drowsiness_data(self, filepath: str):
        """Export drowsiness analysis data."""
        import json
        
        data = {
            'alert_history': [
                {
                    'timestamp': entry['timestamp'],
                    'level': entry['level'].name,
                    'score': entry['score'],
                    'perclos': entry['perclos']
                }
                for entry in self.alert_history
            ],
            'statistics': self.get_drowsiness_statistics(),
            'thresholds': self.thresholds,
            'model_info': {
                'sequence_length': self.sequence_length,
                'feature_names': ['EAR', 'Blink', 'Closure_Duration', 'Blink_Rate', 'Head_Pitch', 'Head_Yaw']
            }
        }
        
        # Convert enum to string
        data['statistics']['current_level'] = self.current_drowsiness_level.name
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Drowsiness data exported to {filepath}")
    
    def visualize_drowsiness(
        self, 
        frame: np.ndarray, 
        results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Visualize drowsiness detection results on frame.
        
        Args:
            frame: Input frame
            results: Drowsiness analysis results
            
        Returns:
            Frame with visualization overlay
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Color coding for drowsiness levels
        level_colors = {
            DrowsinessLevel.ALERT: (0, 255, 0),           # Green
            DrowsinessLevel.SLIGHTLY_DROWSY: (0, 255, 255), # Yellow
            DrowsinessLevel.MODERATELY_DROWSY: (0, 165, 255), # Orange
            DrowsinessLevel.VERY_DROWSY: (0, 100, 255),    # Dark Orange
            DrowsinessLevel.EXTREMELY_DROWSY: (0, 0, 255)   # Red
        }
        
        level = results['drowsiness_level']
        color = level_colors.get(level, (255, 255, 255))
        
        # Draw status box
        cv2.rectangle(vis_frame, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (10, 10), (400, 150), color, 2)
        
        # Draw level text
        level_text = f"Drowsiness: {level.name}"
        cv2.putText(vis_frame, level_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw score
        score_text = f"Score: {results['drowsiness_score']:.2f}"
        cv2.putText(vis_frame, score_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw PERCLOS
        perclos_text = f"PERCLOS: {results['perclos']:.1%}"
        cv2.putText(vis_frame, perclos_text, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw blink rate
        blink_rate_text = f"Blink Rate: {results['blink_rate']:.1f}/min"
        cv2.putText(vis_frame, blink_rate_text, (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw alerts
        if results['alerts']:
            alert_y = 180
            for alert in results['alerts'][:3]:  # Show max 3 alerts
                cv2.putText(vis_frame, alert, (10, alert_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                alert_y += 25
        
        # Draw drowsiness trend graph (mini)
        if len(self.drowsiness_history) > 1:
            graph_x = w - 200
            graph_y = 20
            graph_w = 180
            graph_h = 100
            
            # Background
            cv2.rectangle(vis_frame, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
            cv2.rectangle(vis_frame, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), (255, 255, 255), 1)
            
            # Plot trend
            scores = list(self.drowsiness_history)
            if len(scores) > 1:
                max_score = max(max(scores), 1.0)
                for i in range(1, len(scores)):
                    x1 = graph_x + int((i-1) * graph_w / len(scores))
                    y1 = graph_y + graph_h - int(scores[i-1] * graph_h / max_score)
                    x2 = graph_x + int(i * graph_w / len(scores))
                    y2 = graph_y + graph_h - int(scores[i] * graph_h / max_score)
                    
                    cv2.line(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        return vis_frame
