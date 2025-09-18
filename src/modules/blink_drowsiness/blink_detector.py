"""
Real-time blink detection system.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
import logging
from collections import deque
import time

from .models import BlinkCNN, EARCalculator
from .utils import calculate_ear, detect_blink_sequence
from ..utils.face_detector import FaceDetector

logger = logging.getLogger(__name__)


class BlinkDetector:
    """
    Real-time blink detection system using multiple approaches:
    1. Eye Aspect Ratio (EAR) based detection
    2. CNN-based blink classification
    3. Temporal sequence analysis
    """
    
    def __init__(
        self,
        method: str = 'hybrid',  # 'ear', 'cnn', 'hybrid'
        model_path: Optional[str] = None,
        ear_threshold: float = 0.25,
        blink_frames: int = 3,
        device: str = 'auto',
        history_length: int = 30
    ):
        """
        Initialize blink detector.
        
        Args:
            method: Detection method ('ear', 'cnn', 'hybrid')
            model_path: Path to pre-trained CNN model
            ear_threshold: EAR threshold for blink detection
            blink_frames: Minimum frames for blink confirmation
            device: Device for model inference
            history_length: Length of history buffer
        """
        self.method = method
        self.ear_threshold = ear_threshold
        self.blink_frames = blink_frames
        self.history_length = history_length
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Initialize models
        self.ear_calculator = EARCalculator()
        
        if self.method in ['cnn', 'hybrid']:
            self.cnn_model = BlinkCNN()
            self.cnn_model.to(self.device)
            
            if model_path:
                self._load_cnn_model(model_path)
        
        # History buffers
        self.ear_history = deque(maxlen=history_length)
        self.blink_history = deque(maxlen=history_length)
        self.frame_timestamps = deque(maxlen=history_length)
        
        # Blink statistics
        self.total_blinks = 0
        self.last_blink_time = 0
        self.blink_rate = 0.0  # blinks per minute
        
        # State tracking
        self.eye_closed_frames = 0
        self.is_blinking = False
        self.blink_start_time = 0
        
        logger.info(f"BlinkDetector initialized with {method} method on {self.device}")
    
    def _load_cnn_model(self, model_path: str):
        """Load pre-trained CNN model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.cnn_model.eval()
            logger.info(f"Loaded CNN model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load CNN model: {e}")
    
    def detect_blink(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect blinks in a single frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary containing blink detection results
        """
        current_time = time.time()
        
        results = {
            'blink_detected': False,
            'ear': 0.0,
            'cnn_confidence': 0.0,
            'eye_regions': None,
            'landmarks': None,
            'blink_duration': 0.0,
            'method_used': self.method
        }
        
        try:
            # Detect faces and landmarks
            faces = self.face_detector.detect_faces(frame)
            if not faces:
                return results
            
            # Use largest face
            face_bbox = max(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            landmarks = self.face_detector.get_landmarks(frame, face_bbox)
            
            if landmarks is None:
                return results
            
            results['landmarks'] = landmarks
            
            # Extract eye regions
            left_eye_landmarks = landmarks[36:42]
            right_eye_landmarks = landmarks[42:48]
            
            # Calculate EAR
            if self.method in ['ear', 'hybrid']:
                left_ear = calculate_ear(left_eye_landmarks)
                right_ear = calculate_ear(right_eye_landmarks)
                avg_ear = (left_ear + right_ear) / 2.0
                
                results['ear'] = avg_ear
                self.ear_history.append(avg_ear)
            
            # CNN-based detection
            if self.method in ['cnn', 'hybrid']:
                eye_regions = self._extract_eye_regions(frame, landmarks)
                if eye_regions:
                    results['eye_regions'] = eye_regions
                    cnn_confidence = self._predict_blink_cnn(eye_regions)
                    results['cnn_confidence'] = cnn_confidence
            
            # Determine blink based on method
            blink_detected = False
            
            if self.method == 'ear':
                blink_detected = self._detect_blink_ear()
            elif self.method == 'cnn':
                blink_detected = results['cnn_confidence'] > 0.5
            elif self.method == 'hybrid':
                # Combine EAR and CNN predictions
                ear_blink = self._detect_blink_ear()
                cnn_blink = results['cnn_confidence'] > 0.5
                blink_detected = ear_blink or cnn_blink
            
            # Update blink state
            if blink_detected:
                if not self.is_blinking:
                    self.is_blinking = True
                    self.blink_start_time = current_time
                self.eye_closed_frames += 1
            else:
                if self.is_blinking and self.eye_closed_frames >= self.blink_frames:
                    # Blink completed
                    blink_duration = current_time - self.blink_start_time
                    results['blink_duration'] = blink_duration
                    results['blink_detected'] = True
                    
                    self._register_blink(current_time, blink_duration)
                
                self.is_blinking = False
                self.eye_closed_frames = 0
            
            # Update history
            self.blink_history.append(blink_detected)
            self.frame_timestamps.append(current_time)
            
            # Update blink rate
            self._update_blink_rate(current_time)
            
        except Exception as e:
            logger.error(f"Error in blink detection: {e}")
        
        return results
    
    def _detect_blink_ear(self) -> bool:
        """Detect blink using EAR method."""
        if len(self.ear_history) < self.blink_frames:
            return False
        
        # Check if recent EAR values are below threshold
        recent_ears = list(self.ear_history)[-self.blink_frames:]
        return all(ear < self.ear_threshold for ear in recent_ears)
    
    def _extract_eye_regions(
        self, 
        frame: np.ndarray, 
        landmarks: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract left and right eye regions from landmarks."""
        try:
            h, w = frame.shape[:2]
            
            # Left eye region
            left_eye_landmarks = landmarks[36:42]
            left_x_coords = left_eye_landmarks[:, 0]
            left_y_coords = left_eye_landmarks[:, 1]
            
            left_x1 = max(0, int(np.min(left_x_coords)) - 10)
            left_y1 = max(0, int(np.min(left_y_coords)) - 10)
            left_x2 = min(w, int(np.max(left_x_coords)) + 10)
            left_y2 = min(h, int(np.max(left_y_coords)) + 10)
            
            left_eye = frame[left_y1:left_y2, left_x1:left_x2]
            
            # Right eye region
            right_eye_landmarks = landmarks[42:48]
            right_x_coords = right_eye_landmarks[:, 0]
            right_y_coords = right_eye_landmarks[:, 1]
            
            right_x1 = max(0, int(np.min(right_x_coords)) - 10)
            right_y1 = max(0, int(np.min(right_y_coords)) - 10)
            right_x2 = min(w, int(np.max(right_x_coords)) + 10)
            right_y2 = min(h, int(np.max(right_y_coords)) + 10)
            
            right_eye = frame[right_y1:right_y2, right_x1:right_x2]
            
            if left_eye.size > 0 and right_eye.size > 0:
                return left_eye, right_eye
            
        except Exception as e:
            logger.error(f"Error extracting eye regions: {e}")
        
        return None
    
    def _predict_blink_cnn(self, eye_regions: Tuple[np.ndarray, np.ndarray]) -> float:
        """Predict blink probability using CNN."""
        try:
            left_eye, right_eye = eye_regions
            
            # Preprocess eye regions
            left_tensor = self._preprocess_eye_for_cnn(left_eye)
            right_tensor = self._preprocess_eye_for_cnn(right_eye)
            
            # Stack for batch processing
            eye_batch = torch.stack([left_tensor, right_tensor]).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.cnn_model(eye_batch)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Average predictions from both eyes
                avg_prob = torch.mean(probabilities[:, 1]).item()  # Blink class probability
                
            return avg_prob
            
        except Exception as e:
            logger.error(f"Error in CNN prediction: {e}")
            return 0.0
    
    def _preprocess_eye_for_cnn(self, eye_image: np.ndarray) -> torch.Tensor:
        """Preprocess eye image for CNN input."""
        # Resize to model input size
        eye_resized = cv2.resize(eye_image, (64, 32))
        
        # Convert to grayscale if needed
        if len(eye_resized.shape) == 3:
            eye_gray = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2GRAY)
        else:
            eye_gray = eye_resized
        
        # Normalize
        eye_normalized = eye_gray.astype(np.float32) / 255.0
        
        # Convert to tensor
        eye_tensor = torch.tensor(eye_normalized).unsqueeze(0)  # Add channel dimension
        
        return eye_tensor
    
    def _register_blink(self, timestamp: float, duration: float):
        """Register a completed blink."""
        self.total_blinks += 1
        self.last_blink_time = timestamp
        
        logger.debug(f"Blink registered: duration={duration:.3f}s, total={self.total_blinks}")
    
    def _update_blink_rate(self, current_time: float):
        """Update blink rate calculation."""
        if len(self.frame_timestamps) < 2:
            return
        
        # Calculate blinks in the last minute
        one_minute_ago = current_time - 60.0
        recent_blinks = 0
        
        for i, (timestamp, blink) in enumerate(zip(self.frame_timestamps, self.blink_history)):
            if timestamp >= one_minute_ago and blink:
                recent_blinks += 1
        
        self.blink_rate = recent_blinks
    
    def get_blink_statistics(self) -> Dict[str, Any]:
        """Get current blink statistics."""
        current_time = time.time()
        
        return {
            'total_blinks': self.total_blinks,
            'blink_rate': self.blink_rate,  # blinks per minute
            'last_blink_time': self.last_blink_time,
            'time_since_last_blink': current_time - self.last_blink_time if self.last_blink_time > 0 else 0,
            'average_ear': np.mean(self.ear_history) if self.ear_history else 0.0,
            'ear_std': np.std(self.ear_history) if len(self.ear_history) > 1 else 0.0
        }
    
    def reset_statistics(self):
        """Reset blink statistics."""
        self.total_blinks = 0
        self.last_blink_time = 0
        self.blink_rate = 0.0
        self.ear_history.clear()
        self.blink_history.clear()
        self.frame_timestamps.clear()
        self.eye_closed_frames = 0
        self.is_blinking = False
        
        logger.info("Blink statistics reset")
    
    def set_ear_threshold(self, threshold: float):
        """Update EAR threshold."""
        self.ear_threshold = threshold
        logger.info(f"EAR threshold updated to {threshold}")
    
    def calibrate_ear_threshold(self, calibration_frames: int = 100) -> float:
        """
        Calibrate EAR threshold based on recent frames.
        
        Args:
            calibration_frames: Number of frames to use for calibration
            
        Returns:
            Calibrated threshold value
        """
        if len(self.ear_history) < calibration_frames:
            logger.warning("Insufficient data for EAR calibration")
            return self.ear_threshold
        
        recent_ears = list(self.ear_history)[-calibration_frames:]
        mean_ear = np.mean(recent_ears)
        std_ear = np.std(recent_ears)
        
        # Set threshold as mean - 2*std (assuming blinks are outliers)
        calibrated_threshold = mean_ear - 2 * std_ear
        
        # Ensure reasonable bounds
        calibrated_threshold = max(0.15, min(0.35, calibrated_threshold))
        
        self.ear_threshold = calibrated_threshold
        logger.info(f"EAR threshold calibrated to {calibrated_threshold:.3f}")
        
        return calibrated_threshold
    
    def export_blink_data(self, filepath: str):
        """Export blink detection data for analysis."""
        import json
        
        data = {
            'timestamps': list(self.frame_timestamps),
            'ear_values': list(self.ear_history),
            'blink_detections': list(self.blink_history),
            'statistics': self.get_blink_statistics(),
            'parameters': {
                'method': self.method,
                'ear_threshold': self.ear_threshold,
                'blink_frames': self.blink_frames
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Blink data exported to {filepath}")
    
    def visualize_blink_detection(
        self, 
        frame: np.ndarray, 
        results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Visualize blink detection results on frame.
        
        Args:
            frame: Input frame
            results: Blink detection results
            
        Returns:
            Frame with visualization overlay
        """
        vis_frame = frame.copy()
        
        # Draw landmarks if available
        if results.get('landmarks') is not None:
            landmarks = results['landmarks']
            
            # Draw eye landmarks
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            for point in left_eye:
                cv2.circle(vis_frame, tuple(point.astype(int)), 2, (0, 255, 0), -1)
            
            for point in right_eye:
                cv2.circle(vis_frame, tuple(point.astype(int)), 2, (0, 255, 0), -1)
        
        # Draw status text
        y_offset = 30
        
        if results['blink_detected']:
            cv2.putText(vis_frame, "BLINK DETECTED", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_offset += 40
        
        # Draw EAR value
        if results['ear'] > 0:
            ear_text = f"EAR: {results['ear']:.3f}"
            cv2.putText(vis_frame, ear_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        # Draw CNN confidence
        if results['cnn_confidence'] > 0:
            cnn_text = f"CNN: {results['cnn_confidence']:.3f}"
            cv2.putText(vis_frame, cnn_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        # Draw statistics
        stats = self.get_blink_statistics()
        stats_text = f"Blinks: {stats['total_blinks']} | Rate: {stats['blink_rate']:.1f}/min"
        cv2.putText(vis_frame, stats_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
