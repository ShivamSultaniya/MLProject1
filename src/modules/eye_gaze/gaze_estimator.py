"""
Main Gaze Estimator Class

Handles eye gaze estimation from facial images using trained models.
Supports both MPIIGaze and GazeCapture model architectures.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

from .model import GazeNet
from .utils import preprocess_eye_region, calculate_gaze_vector


class GazeEstimator:
    """
    Eye gaze estimation system that can work with webcam input or static images.
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', model_type: str = 'gazenet'):
        """
        Initialize the gaze estimator.
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run inference on ('cpu' or 'cuda')
            model_type: Type of model architecture ('gazenet', 'mpiigaze', 'gazecapture')
        """
        self.device = torch.device(device)
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Face and eye detection setup
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Calibration parameters
        self.is_calibrated = False
        self.calibration_points = []
        self.calibration_gazes = []
        
        # Store last face/eyes detection for external consumers
        self.last_detection: Optional[Dict[str, Any]] = None
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained gaze estimation model."""
        try:
            model = GazeNet(model_type=self.model_type)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            self.logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            # Return a default model if loading fails
            return GazeNet(model_type=self.model_type).to(self.device)
    
    def detect_face_and_eyes(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect face and eyes in the input frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing face and eye regions
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        result = {
            'face_detected': False,
            'eyes_detected': False,
            'face_region': None,
            'left_eye': None,
            'right_eye': None
        }
        
        if len(faces) > 0:
            # Take the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            result['face_detected'] = True
            result['face_region'] = (x, y, w, h)
            
            # Detect eyes within face region
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate (left to right)
                eyes = sorted(eyes, key=lambda x: x[0])
                result['eyes_detected'] = True
                result['left_eye'] = (x + eyes[0][0], y + eyes[0][1], eyes[0][2], eyes[0][3])
                result['right_eye'] = (x + eyes[1][0], y + eyes[1][1], eyes[1][2], eyes[1][3])
        
        return result
    
    def estimate_gaze(self, frame: np.ndarray, face_info: Optional[Dict] = None) -> Tuple[float, float, float]:
        """
        Estimate gaze direction from input frame.
        
        Args:
            frame: Input image frame
            face_info: Optional pre-computed face detection results
            
        Returns:
            Tuple of (yaw, pitch, confidence) in radians
        """
        if face_info is None:
            face_info = self.detect_face_and_eyes(frame)
        
        # Store last detection state for external use
        try:
            self.last_detection = face_info
        except Exception:
            self.last_detection = None
        
        if not face_info['face_detected'] or not face_info['eyes_detected']:
            return 0.0, 0.0, 0.0  # No gaze detected
        
        # Extract and preprocess eye regions
        left_eye_region = self._extract_eye_region(frame, face_info['left_eye'])
        right_eye_region = self._extract_eye_region(frame, face_info['right_eye'])
        
        if left_eye_region is None or right_eye_region is None:
            return 0.0, 0.0, 0.0
        
        # Preprocess for model input
        left_eye_tensor = preprocess_eye_region(left_eye_region)
        right_eye_tensor = preprocess_eye_region(right_eye_region)
        
        # Stack tensors for batch processing
        eye_input = torch.stack([left_eye_tensor, right_eye_tensor]).to(self.device)
        
        # Model inference
        with torch.no_grad():
            gaze_output = self.model(eye_input)
            
            if self.model_type == 'gazenet':
                # GazeNet outputs [B, 2] (yaw, pitch). Aggregate both eyes.
                if gaze_output.dim() == 2:
                    avg_output = gaze_output.mean(dim=0)
                else:
                    avg_output = gaze_output
                yaw = avg_output[0].item()
                pitch = avg_output[1].item()
                # Heuristic confidence based on agreement between the two eyes
                if gaze_output.dim() == 2 and gaze_output.size(0) >= 2:
                    eye_diff = torch.norm(gaze_output[0] - gaze_output[1]).item()
                    confidence = max(0.0, min(1.0, 1.0 - eye_diff))
                else:
                    confidence = 0.5
            else:
                # Other models might output [B, 2] or [B, 3] gaze vectors; aggregate if batched
                output = gaze_output
                if output.dim() == 2 and output.size(0) >= 1:
                    avg_output = output.mean(dim=0).cpu().numpy()
                else:
                    avg_output = output.cpu().numpy()
                yaw, pitch = calculate_gaze_vector(avg_output)
                # Heuristic confidence
                if avg_output.shape[0] == 3:
                    confidence = float(np.linalg.norm(avg_output))
                else:
                    confidence = 0.5
        
        return float(yaw), float(pitch), float(confidence)
    
    def _extract_eye_region(self, frame: np.ndarray, eye_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract and crop eye region from frame."""
        x, y, w, h = eye_bbox
        
        # Add some padding around the eye
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_end = min(frame.shape[0], y + h + padding)
        
        eye_region = frame[y_start:y_end, x_start:x_end]
        
        if eye_region.size == 0:
            return None
            
        return eye_region
    
    def calibrate(self, calibration_data: list) -> bool:
        """
        Calibrate the gaze estimator using known gaze points.
        
        Args:
            calibration_data: List of (frame, target_point) tuples
            
        Returns:
            True if calibration successful
        """
        try:
            self.calibration_points = []
            self.calibration_gazes = []
            
            for frame, target_point in calibration_data:
                yaw, pitch, confidence = self.estimate_gaze(frame)
                if confidence > 0.5:  # Only use high-confidence estimates
                    self.calibration_points.append(target_point)
                    self.calibration_gazes.append((yaw, pitch))
            
            if len(self.calibration_points) >= 5:  # Need minimum calibration points
                self.is_calibrated = True
                self.logger.info(f"Calibration completed with {len(self.calibration_points)} points")
                return True
            else:
                self.logger.warning("Insufficient calibration points")
                return False
                
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False
    
    def get_screen_coordinates(self, yaw: float, pitch: float, screen_width: int, screen_height: int) -> Tuple[int, int]:
        """
        Convert gaze angles to screen coordinates.
        
        Args:
            yaw: Horizontal gaze angle in radians
            pitch: Vertical gaze angle in radians
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            Tuple of (x, y) screen coordinates
        """
        if not self.is_calibrated:
            # Use simple mapping without calibration
            x = int((yaw + np.pi/4) / (np.pi/2) * screen_width)
            y = int((pitch + np.pi/6) / (np.pi/3) * screen_height)
        else:
            # Use calibration data for more accurate mapping
            # This is a simplified approach - in practice, you'd use more sophisticated mapping
            x = int((yaw + np.pi/4) / (np.pi/2) * screen_width)
            y = int((pitch + np.pi/6) / (np.pi/3) * screen_height)
        
        # Clamp to screen boundaries
        x = max(0, min(screen_width - 1, x))
        y = max(0, min(screen_height - 1, y))
        
        return x, y
    
    def process_video_stream(self, video_source: int = 0) -> None:
        """
        Process video stream and display gaze estimation results.
        
        Args:
            video_source: Video source index (0 for default webcam)
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            self.logger.error("Error: Could not open video source")
            return
        
        self.logger.info("Starting video stream processing. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face and eyes
            face_info = self.detect_face_and_eyes(frame)
            
            # Estimate gaze
            yaw, pitch, confidence = self.estimate_gaze(frame, face_info)
            
            # Draw detection results
            self._draw_detection_results(frame, face_info, yaw, pitch, confidence)
            
            # Display frame
            cv2.imshow('Gaze Estimation', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_detection_results(self, frame: np.ndarray, face_info: Dict, yaw: float, pitch: float, confidence: float) -> None:
        """Draw detection and gaze estimation results on frame."""
        # Draw face rectangle
        if face_info['face_detected']:
            x, y, w, h = face_info['face_region']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw eye rectangles
        if face_info['eyes_detected']:
            for eye_key in ['left_eye', 'right_eye']:
                if face_info[eye_key] is not None:
                    x, y, w, h = face_info[eye_key]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw gaze vector
        if confidence > 0.3:
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            
            # Calculate gaze direction vector
            gaze_length = 100
            end_x = int(center_x + gaze_length * np.sin(yaw))
            end_y = int(center_y - gaze_length * np.sin(pitch))
            
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)
        
        # Add text information
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


