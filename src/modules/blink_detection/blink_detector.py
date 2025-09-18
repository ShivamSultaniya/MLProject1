"""
Blink Detection System

Implements real-time blink detection using Eye Aspect Ratio (EAR) and 
machine learning approaches. Supports both traditional CV methods and 
deep learning models trained on ZJU Eyeblink dataset.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any
import time
from collections import deque
import logging

from .ear_calculator import calculate_eye_aspect_ratio, detect_eye_landmarks
from .ear_calculator import calculate_mar
from .features import extract_blink_features


class BlinkDetector:
    """
    Real-time blink detection system with multiple detection methods.
    """
    
    def __init__(self, method: str = 'ear', ear_threshold: float = 0.25, 
                 consecutive_frames: int = 3, model_path: Optional[str] = None):
        """
        Initialize blink detector.
        
        Args:
            method: Detection method ('ear', 'ml', 'hybrid')
            ear_threshold: EAR threshold for blink detection
            consecutive_frames: Number of consecutive frames below threshold for blink
            model_path: Path to trained ML model (if using ML method)
        """
        self.method = method
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.model_path = model_path
        
        # Initialize detection state
        self.frame_count = 0
        self.blink_count = 0
        self.ear_history = deque(maxlen=30)  # Keep last 30 EAR values
        self.blink_history = deque(maxlen=100)  # Keep last 100 blinks
        self.consecutive_low_ear = 0
        self.last_blink_time = 0
        
        # Load ML model if specified
        self.ml_model = None
        if method in ['ml', 'hybrid'] and model_path:
            self.ml_model = self._load_ml_model(model_path)
        
        # Initialize face and landmark detection
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
            self.dlib_available = True
        except ImportError:
            logging.warning("dlib not available, using OpenCV cascade classifiers")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.dlib_available = False
        
        self.logger = logging.getLogger(__name__)
    
    def _load_ml_model(self, model_path: str) -> Optional[nn.Module]:
        """Load trained ML model for blink detection."""
        try:
            model = BlinkClassifier()
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.logger.info(f"Loaded ML model from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading ML model: {e}")
            return None
    
    def detect_blink(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect blinks in the current frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary containing blink detection results
        """
        self.frame_count += 1
        current_time = time.time()
        
        result = {
            'blink_detected': False,
            'ear_value': 0.0,
            'mar_value': 0.0,
            'yawn_detected': False,
            'confidence': 0.0,
            'eye_landmarks': None,
            'method_used': self.method
        }
        
        # Detect face and eye landmarks
        landmarks = self._detect_landmarks(frame)
        if landmarks is None:
            return result
        
        result['eye_landmarks'] = landmarks
        
        # Extract eye regions
        left_eye = landmarks[36:42]  # Left eye landmarks
        right_eye = landmarks[42:48]  # Right eye landmarks
        # Extract simple inner mouth landmarks if available (48-67)
        mouth = None
        if len(landmarks) >= 68:
            # Inner mouth usually 60-67, corners at 60 and 64; use a subset safely
            mouth = landmarks[60:68]
        
        # Calculate Eye Aspect Ratio
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        ear_value = (left_ear + right_ear) / 2.0
        
        result['ear_value'] = ear_value
        self.ear_history.append(ear_value)
        
        # Compute MAR if mouth landmarks available
        if mouth is not None and len(mouth) >= 6:
            # Reindex a compact 6-point set: corners and verticals
            compact = np.array([
                mouth[0],  # left corner (60)
                mouth[2],  # upper inner (62)
                mouth[4],  # top mid (64-? ensure inner top)
                mouth[4],  # right corner proxy; will override below
                mouth[6],  # lower inner (66)
                mouth[0],  # left bottom proxy; will override below
            ])
            # More robust explicit mapping for inner mouth: 60, 62, 64, 66
            left_corner = mouth[0]
            right_corner = mouth[4]
            top_lip = mouth[2]
            bottom_lip = mouth[6]
            mar = calculate_mar(np.array([left_corner, top_lip, top_lip, right_corner, bottom_lip, bottom_lip]))
            result['mar_value'] = float(mar)
            # Heuristic yawn detection threshold
            if mar > 0.6:
                result['yawn_detected'] = True
        
        # Detect blink based on selected method
        if self.method == 'ear':
            blink_detected = self._detect_blink_ear(ear_value)
            result['confidence'] = self._calculate_ear_confidence(ear_value)
        elif self.method == 'ml' and self.ml_model:
            blink_detected = self._detect_blink_ml(frame, landmarks)
            result['confidence'] = self._get_ml_confidence()
        elif self.method == 'hybrid':
            blink_detected = self._detect_blink_hybrid(frame, landmarks, ear_value)
            result['confidence'] = self._calculate_hybrid_confidence(ear_value)
        else:
            # Fallback to EAR method
            blink_detected = self._detect_blink_ear(ear_value)
            result['confidence'] = self._calculate_ear_confidence(ear_value)
        
        result['blink_detected'] = blink_detected
        
        # Update blink statistics
        if blink_detected:
            self.blink_count += 1
            self.blink_history.append(current_time)
            self.last_blink_time = current_time
            self.logger.debug(f"Blink detected! Total blinks: {self.blink_count}")
        
        return result
    
    def _detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect facial landmarks in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.dlib_available:
            # Use dlib for more accurate landmark detection
            faces = self.detector(gray)
            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda rect: rect.width() * rect.height())
                landmarks = self.predictor(gray, face)
                
                # Convert to numpy array
                coords = np.array([[p.x, p.y] for p in landmarks.parts()])
                return coords
        else:
            # Fallback to OpenCV cascade classifiers
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                # This is a simplified approach - in practice, you'd need
                # a more sophisticated landmark detection method
                x, y, w, h = faces[0]
                # Return approximate eye positions
                left_eye_center = (x + w//4, y + h//3)
                right_eye_center = (x + 3*w//4, y + h//3)
                
                # Create simplified landmark array (just eye corners)
                landmarks = np.array([
                    [left_eye_center[0] - 20, left_eye_center[1]],  # Left eye outer
                    [left_eye_center[0] - 10, left_eye_center[1] - 5],  # Left eye top
                    [left_eye_center[0], left_eye_center[1]],  # Left eye inner
                    [left_eye_center[0] - 10, left_eye_center[1] + 5],  # Left eye bottom
                    [right_eye_center[0], right_eye_center[1]],  # Right eye inner
                    [right_eye_center[0] + 10, right_eye_center[1] - 5],  # Right eye top
                    [right_eye_center[0] + 20, right_eye_center[1]],  # Right eye outer
                    [right_eye_center[0] + 10, right_eye_center[1] + 5],  # Right eye bottom
                ])
                
                # Extend to 68 landmarks with zeros (simplified)
                full_landmarks = np.zeros((68, 2))
                full_landmarks[36:44] = landmarks  # Eye region
                return full_landmarks
        
        return None
    
    def _detect_blink_ear(self, ear_value: float) -> bool:
        """Detect blink using Eye Aspect Ratio method."""
        if ear_value < self.ear_threshold:
            self.consecutive_low_ear += 1
        else:
            if self.consecutive_low_ear >= self.consecutive_frames:
                # Blink detected
                self.consecutive_low_ear = 0
                return True
            self.consecutive_low_ear = 0
        
        return False
    
    def _detect_blink_ml(self, frame: np.ndarray, landmarks: np.ndarray) -> bool:
        """Detect blink using machine learning model."""
        if self.ml_model is None:
            return False
        
        try:
            # Extract features for ML model
            features = extract_blink_features(frame, landmarks)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                output = self.ml_model(features_tensor)
                probability = torch.sigmoid(output).item()
                
                # Consider it a blink if probability > 0.5
                return probability > 0.5
        except Exception as e:
            self.logger.error(f"Error in ML blink detection: {e}")
            return False
    
    def _detect_blink_hybrid(self, frame: np.ndarray, landmarks: np.ndarray, ear_value: float) -> bool:
        """Detect blink using hybrid EAR + ML approach."""
        # First check EAR
        ear_blink = self._detect_blink_ear(ear_value)
        
        # If EAR suggests a blink, confirm with ML model
        if ear_blink and self.ml_model:
            ml_blink = self._detect_blink_ml(frame, landmarks)
            return ml_blink
        
        return ear_blink
    
    def _calculate_ear_confidence(self, ear_value: float) -> float:
        """Calculate confidence score for EAR-based detection."""
        if ear_value < self.ear_threshold:
            # Lower EAR = higher confidence for blink
            confidence = (self.ear_threshold - ear_value) / self.ear_threshold
            return min(1.0, max(0.0, confidence))
        else:
            # Higher EAR = lower confidence for blink
            confidence = 1.0 - ((ear_value - self.ear_threshold) / (0.5 - self.ear_threshold))
            return min(1.0, max(0.0, confidence))
    
    def _get_ml_confidence(self) -> float:
        """Get confidence from ML model."""
        # This would be implemented based on the specific ML model output
        return 0.8  # Placeholder
    
    def _calculate_hybrid_confidence(self, ear_value: float) -> float:
        """Calculate confidence for hybrid method."""
        ear_conf = self._calculate_ear_confidence(ear_value)
        ml_conf = self._get_ml_confidence()
        return (ear_conf + ml_conf) / 2.0
    
    def get_blink_rate(self, time_window: float = 60.0) -> float:
        """
        Calculate blink rate (blinks per minute) over specified time window.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Blink rate in blinks per minute
        """
        current_time = time.time()
        recent_blinks = [t for t in self.blink_history if current_time - t <= time_window]
        
        if len(recent_blinks) == 0:
            return 0.0
        
        # Convert to blinks per minute
        blinks_per_minute = (len(recent_blinks) / time_window) * 60.0
        return blinks_per_minute
    
    def get_average_ear(self, num_frames: int = 30) -> float:
        """Get average EAR over recent frames."""
        if len(self.ear_history) == 0:
            return 0.0
        
        recent_ears = list(self.ear_history)[-num_frames:]
        return sum(recent_ears) / len(recent_ears)
    
    def reset_statistics(self) -> None:
        """Reset blink detection statistics."""
        self.frame_count = 0
        self.blink_count = 0
        self.ear_history.clear()
        self.blink_history.clear()
        self.consecutive_low_ear = 0
        self.last_blink_time = 0
    
    def calibrate(self, calibration_frames: List[np.ndarray], 
                  known_blinks: List[bool]) -> bool:
        """
        Calibrate the detector using known blink/no-blink frames.
        
        Args:
            calibration_frames: List of frames
            known_blinks: List of boolean values indicating blink presence
            
        Returns:
            True if calibration successful
        """
        if len(calibration_frames) != len(known_blinks):
            return False
        
        ear_values = []
        
        # Process calibration frames
        for frame in calibration_frames:
            landmarks = self._detect_landmarks(frame)
            if landmarks is not None:
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                left_ear = calculate_eye_aspect_ratio(left_eye)
                right_ear = calculate_eye_aspect_ratio(right_eye)
                ear_value = (left_ear + right_ear) / 2.0
                ear_values.append(ear_value)
            else:
                ear_values.append(0.3)  # Default value
        
        if len(ear_values) == 0:
            return False
        
        # Find optimal threshold
        best_threshold = self.ear_threshold
        best_accuracy = 0
        
        for threshold in np.arange(0.15, 0.35, 0.01):
            correct_predictions = 0
            
            for ear_val, is_blink in zip(ear_values, known_blinks):
                predicted_blink = ear_val < threshold
                if predicted_blink == is_blink:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(known_blinks)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        # Update threshold if improvement is significant
        if best_accuracy > 0.7:  # Only update if reasonably accurate
            self.ear_threshold = best_threshold
            self.logger.info(f"Calibrated EAR threshold to {best_threshold:.3f} (accuracy: {best_accuracy:.3f})")
            return True
        
        return False
    
    def visualize_detection(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Visualize blink detection results on frame.
        
        Args:
            frame: Input frame
            detection_result: Result from detect_blink()
            
        Returns:
            Frame with visualization overlay
        """
        result_frame = frame.copy()
        
        # Draw eye landmarks if available
        if detection_result['eye_landmarks'] is not None:
            landmarks = detection_result['eye_landmarks']
            
            # Draw left eye
            left_eye = landmarks[36:42]
            for i in range(len(left_eye)):
                cv2.circle(result_frame, tuple(left_eye[i].astype(int)), 2, (0, 255, 0), -1)
            
            # Draw right eye
            right_eye = landmarks[42:48]
            for i in range(len(right_eye)):
                cv2.circle(result_frame, tuple(right_eye[i].astype(int)), 2, (0, 255, 0), -1)
        
        # Add text information
        y_offset = 30
        cv2.putText(result_frame, f"EAR: {detection_result['ear_value']:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(result_frame, f"Blinks: {self.blink_count}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        blink_rate = self.get_blink_rate()
        cv2.putText(result_frame, f"Rate: {blink_rate:.1f}/min", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Highlight if blink detected
        if detection_result['blink_detected']:
            cv2.putText(result_frame, "BLINK!", (frame.shape[1]//2 - 50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        return result_frame


class BlinkClassifier(nn.Module):
    """
    Neural network classifier for blink detection.
    """
    
    def __init__(self, input_features: int = 12):
        super(BlinkClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Binary classification
        )
    
    def forward(self, x):
        return self.classifier(x)


def create_blink_detector(config: Dict[str, Any]) -> BlinkDetector:
    """
    Factory function to create blink detector with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured BlinkDetector instance
    """
    return BlinkDetector(
        method=config.get('method', 'ear'),
        ear_threshold=config.get('ear_threshold', 0.25),
        consecutive_frames=config.get('consecutive_frames', 3),
        model_path=config.get('model_path', None)
    )


