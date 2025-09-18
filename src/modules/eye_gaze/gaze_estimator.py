"""
Main Gaze Estimator class for real-time eye gaze estimation.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

from .models import ResNetGaze, MobileNetGaze
from .utils import preprocess_eye_image, postprocess_gaze_vector, extract_eye_regions
from ..utils.face_detector import FaceDetector

logger = logging.getLogger(__name__)


class GazeEstimator:
    """
    Real-time eye gaze estimation system.
    
    This class provides functionality for estimating gaze direction from webcam input
    using pre-trained deep learning models.
    """
    
    def __init__(
        self,
        model_type: str = 'resnet',
        model_path: Optional[str] = None,
        device: str = 'auto',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the gaze estimator.
        
        Args:
            model_type: Type of model to use ('resnet' or 'mobilenet')
            model_path: Path to pre-trained model weights
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
            confidence_threshold: Minimum confidence for valid predictions
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Initialize model
        self._load_model(model_path)
        
        # Calibration parameters
        self.is_calibrated = False
        self.calibration_points = []
        self.calibration_gazes = []
        
        logger.info(f"GazeEstimator initialized with {model_type} model on {self.device}")
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load the gaze estimation model."""
        if self.model_type == 'resnet':
            self.model = ResNetGaze()
        elif self.model_type == 'mobilenet':
            self.model = MobileNetGaze()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.to(self.device)
        
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            logger.warning("No pre-trained model loaded. Model will need training.")
        
        self.model.eval()
    
    def estimate_gaze(
        self, 
        frame: np.ndarray,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Estimate gaze direction from a single frame.
        
        Args:
            frame: Input image frame (BGR format)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing gaze estimation results
        """
        results = {
            'gaze_vector': None,
            'gaze_angles': None,
            'eye_regions': None,
            'face_bbox': None,
            'confidence': 0.0,
            'valid': False
        }
        
        try:
            # Detect face
            faces = self.face_detector.detect_faces(frame)
            if not faces:
                return results
            
            # Use the largest face
            face_bbox = max(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            results['face_bbox'] = face_bbox
            
            # Extract eye regions
            eye_regions = extract_eye_regions(frame, face_bbox)
            if eye_regions is None:
                return results
            
            results['eye_regions'] = eye_regions
            left_eye, right_eye = eye_regions
            
            # Preprocess eye images
            left_eye_tensor = preprocess_eye_image(left_eye)
            right_eye_tensor = preprocess_eye_image(right_eye)
            
            # Stack for batch processing
            eye_batch = torch.stack([left_eye_tensor, right_eye_tensor]).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(eye_batch)
                
                if isinstance(outputs, tuple):
                    gaze_preds, confidence_scores = outputs
                else:
                    gaze_preds = outputs
                    confidence_scores = torch.ones(gaze_preds.shape[0])
            
            # Average predictions from both eyes
            gaze_vector = torch.mean(gaze_preds, dim=0)
            confidence = torch.mean(confidence_scores).item()
            
            # Post-process gaze vector
            gaze_vector = postprocess_gaze_vector(gaze_vector.cpu().numpy())
            
            # Convert to angles (pitch, yaw)
            gaze_angles = self._vector_to_angles(gaze_vector)
            
            # Apply calibration if available
            if self.is_calibrated:
                gaze_angles = self._apply_calibration(gaze_angles)
            
            results.update({
                'gaze_vector': gaze_vector,
                'gaze_angles': gaze_angles,
                'confidence': confidence,
                'valid': confidence > self.confidence_threshold
            })
            
        except Exception as e:
            logger.error(f"Error in gaze estimation: {e}")
            
        return results
    
    def _vector_to_angles(self, gaze_vector: np.ndarray) -> Tuple[float, float]:
        """Convert 3D gaze vector to pitch and yaw angles."""
        x, y, z = gaze_vector
        
        # Calculate pitch (vertical angle)
        pitch = np.arctan2(-y, np.sqrt(x*x + z*z))
        
        # Calculate yaw (horizontal angle)
        yaw = np.arctan2(-x, -z)
        
        # Convert to degrees
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        
        return pitch_deg, yaw_deg
    
    def start_calibration(self, num_points: int = 9):
        """Start gaze calibration process."""
        self.calibration_points = []
        self.calibration_gazes = []
        self.is_calibrated = False
        logger.info(f"Starting calibration with {num_points} points")
    
    def add_calibration_point(
        self, 
        screen_point: Tuple[float, float], 
        frame: np.ndarray
    ) -> bool:
        """
        Add a calibration point during calibration process.
        
        Args:
            screen_point: (x, y) coordinates on screen
            frame: Current frame when user is looking at the point
            
        Returns:
            Success status
        """
        gaze_result = self.estimate_gaze(frame)
        
        if gaze_result['valid']:
            self.calibration_points.append(screen_point)
            self.calibration_gazes.append(gaze_result['gaze_angles'])
            logger.info(f"Added calibration point {len(self.calibration_points)}")
            return True
        
        return False
    
    def finish_calibration(self) -> bool:
        """Finish calibration and compute transformation parameters."""
        if len(self.calibration_points) < 4:
            logger.warning("Need at least 4 calibration points")
            return False
        
        try:
            # Compute calibration transformation
            points = np.array(self.calibration_points)
            gazes = np.array(self.calibration_gazes)
            
            # Simple linear transformation for now
            # In practice, you might want a more sophisticated calibration
            self.calibration_matrix = np.linalg.lstsq(
                np.column_stack([gazes, np.ones(len(gazes))]), 
                points, 
                rcond=None
            )[0]
            
            self.is_calibrated = True
            logger.info("Calibration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def _apply_calibration(self, gaze_angles: Tuple[float, float]) -> Tuple[float, float]:
        """Apply calibration transformation to gaze angles."""
        if not self.is_calibrated:
            return gaze_angles
        
        try:
            gaze_vec = np.array([gaze_angles[0], gaze_angles[1], 1])
            calibrated = self.calibration_matrix.T @ gaze_vec
            return calibrated[0], calibrated[1]
        except:
            return gaze_angles
    
    def get_attention_region(
        self, 
        gaze_angles: Tuple[float, float], 
        screen_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Convert gaze angles to screen coordinates.
        
        Args:
            gaze_angles: (pitch, yaw) in degrees
            screen_size: (width, height) of screen
            
        Returns:
            (x, y) coordinates on screen
        """
        pitch, yaw = gaze_angles
        width, height = screen_size
        
        # Simple mapping - in practice this needs calibration
        x = int(width * (0.5 + yaw / 60.0))  # Assume ±30° field of view
        y = int(height * (0.5 + pitch / 40.0))  # Assume ±20° field of view
        
        # Clamp to screen bounds
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        
        return x, y
    
    def reset_calibration(self):
        """Reset calibration data."""
        self.calibration_points = []
        self.calibration_gazes = []
        self.is_calibrated = False
        logger.info("Calibration reset")
    
    def save_model(self, path: str):
        """Save the current model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'calibration_matrix': getattr(self, 'calibration_matrix', None),
            'is_calibrated': self.is_calibrated
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'calibration_matrix' in checkpoint and checkpoint['calibration_matrix'] is not None:
            self.calibration_matrix = checkpoint['calibration_matrix']
            self.is_calibrated = checkpoint.get('is_calibrated', False)
        
        logger.info(f"Model loaded from {path}")
