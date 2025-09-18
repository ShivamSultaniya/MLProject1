"""
Head pose estimation system for real-time analysis.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import logging
from collections import deque
import time

from .models import ResNetPose, EfficientNetPose, PoseNet
from .utils import euler_angles_to_rotation_matrix, draw_pose_axes, project_3d_points
from ..utils.face_detector import FaceDetector

logger = logging.getLogger(__name__)


class HeadPoseEstimator:
    """
    Real-time head pose estimation system.
    
    Estimates head orientation (pitch, yaw, roll) from facial images
    to analyze attention direction and distraction patterns.
    """
    
    def __init__(
        self,
        model_type: str = 'resnet',  # 'resnet', 'efficientnet', 'posenet'
        model_path: Optional[str] = None,
        device: str = 'auto',
        input_size: Tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.5,
        smoothing_window: int = 5
    ):
        """
        Initialize head pose estimator.
        
        Args:
            model_type: Type of pose estimation model
            model_path: Path to pre-trained model weights
            device: Device for inference
            input_size: Input image size for model
            confidence_threshold: Minimum confidence for valid predictions
            smoothing_window: Window size for temporal smoothing
        """
        self.model_type = model_type
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Initialize pose model
        self._load_model(model_path)
        
        # History for smoothing
        self.pose_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        
        # Camera calibration parameters (default values)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # 3D model points for pose estimation (generic face model)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),      # Nose tip
            (0.0, -330.0, -65.0), # Chin
            (-225.0, 170.0, -135.0), # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)
        
        logger.info(f"HeadPoseEstimator initialized with {model_type} model on {self.device}")
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load the pose estimation model."""
        if self.model_type == 'resnet':
            self.model = ResNetPose(input_size=self.input_size)
        elif self.model_type == 'efficientnet':
            self.model = EfficientNetPose(input_size=self.input_size)
        elif self.model_type == 'posenet':
            self.model = PoseNet(input_size=self.input_size)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.to(self.device)
        
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
        else:
            logger.warning("No pre-trained model loaded. Model will need training.")
        
        self.model.eval()
    
    def estimate_pose(
        self, 
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Estimate head pose from a single frame.
        
        Args:
            frame: Input image frame (BGR format)
            face_bbox: Optional face bounding box
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing pose estimation results
        """
        results = {
            'pitch': 0.0,
            'yaw': 0.0, 
            'roll': 0.0,
            'rotation_matrix': np.eye(3),
            'translation_vector': np.zeros(3),
            'face_bbox': None,
            'landmarks': None,
            'confidence': 0.0,
            'valid': False,
            'method': 'deep_learning'
        }
        
        try:
            # Detect face if not provided
            if face_bbox is None:
                faces = self.face_detector.detect_faces(frame)
                if not faces:
                    return results
                
                # Use largest face
                face_bbox = max(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            
            results['face_bbox'] = face_bbox
            
            # Extract face region
            face_region = self._extract_face_region(frame, face_bbox)
            if face_region is None:
                return results
            
            # Deep learning based pose estimation
            dl_pose, dl_confidence = self._estimate_pose_deep_learning(face_region)
            
            # Try geometric method as backup/validation
            landmarks = self.face_detector.get_landmarks(frame, face_bbox)
            if landmarks is not None:
                results['landmarks'] = landmarks
                
                # Geometric pose estimation using PnP
                geom_pose = self._estimate_pose_pnp(landmarks, frame.shape[:2])
                
                # Use deep learning result if confidence is high, otherwise blend
                if dl_confidence > self.confidence_threshold:
                    pitch, yaw, roll = dl_pose
                    confidence = dl_confidence
                    results['method'] = 'deep_learning'
                elif geom_pose is not None:
                    pitch, yaw, roll = geom_pose['euler_angles']
                    confidence = 0.7  # Assume reasonable confidence for geometric method
                    results['method'] = 'geometric'
                    results['rotation_matrix'] = geom_pose['rotation_matrix']
                    results['translation_vector'] = geom_pose['translation_vector']
                else:
                    pitch, yaw, roll = dl_pose
                    confidence = dl_confidence
                    results['method'] = 'deep_learning_fallback'
            else:
                pitch, yaw, roll = dl_pose
                confidence = dl_confidence
                results['method'] = 'deep_learning_only'
            
            # Apply temporal smoothing
            if len(self.pose_history) > 0:
                pitch, yaw, roll = self._apply_temporal_smoothing((pitch, yaw, roll))
            
            # Update history
            self.pose_history.append((pitch, yaw, roll))
            self.confidence_history.append(confidence)
            
            # Update results
            results.update({
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'confidence': confidence,
                'valid': confidence > self.confidence_threshold
            })
            
            # Calculate rotation matrix if not already set
            if np.array_equal(results['rotation_matrix'], np.eye(3)):
                results['rotation_matrix'] = euler_angles_to_rotation_matrix(pitch, yaw, roll)
            
        except Exception as e:
            logger.error(f"Error in pose estimation: {e}")
        
        return results
    
    def _extract_face_region(
        self, 
        frame: np.ndarray, 
        face_bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Extract and preprocess face region for model input."""
        try:
            x1, y1, x2, y2 = face_bbox
            
            # Extract face region with some padding
            padding = 0.1
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * padding), int(h * padding)
            
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(frame.shape[1], x2 + pad_w)
            y2_pad = min(frame.shape[0], y2 + pad_h)
            
            face_region = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_region.size == 0:
                return None
            
            # Resize to model input size
            face_resized = cv2.resize(face_region, self.input_size)
            
            return face_resized
            
        except Exception as e:
            logger.error(f"Error extracting face region: {e}")
            return None
    
    def _estimate_pose_deep_learning(
        self, 
        face_image: np.ndarray
    ) -> Tuple[Tuple[float, float, float], float]:
        """Estimate pose using deep learning model."""
        try:
            # Preprocess image
            face_tensor = self._preprocess_face_image(face_image)
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(face_tensor)
                
                if isinstance(outputs, tuple):
                    pose_pred, confidence_pred = outputs
                else:
                    pose_pred = outputs
                    confidence_pred = torch.ones(pose_pred.shape[0])
                
                # Extract angles (assuming output is [pitch, yaw, roll])
                pose_angles = pose_pred.cpu().numpy().flatten()
                confidence = confidence_pred.cpu().numpy().flatten()[0]
                
                # Convert from radians to degrees if needed
                if np.max(np.abs(pose_angles)) < np.pi:  # Likely in radians
                    pose_angles = np.degrees(pose_angles)
                
                pitch, yaw, roll = pose_angles[:3]
                
                return (pitch, yaw, roll), confidence
                
        except Exception as e:
            logger.error(f"Error in deep learning pose estimation: {e}")
            return (0.0, 0.0, 0.0), 0.0
    
    def _preprocess_face_image(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess face image for model input."""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        face_normalized = (face_normalized - mean) / std
        
        # Convert to tensor and change to CHW format
        face_tensor = torch.tensor(face_normalized, dtype=torch.float32)
        face_tensor = face_tensor.permute(2, 0, 1)  # HWC to CHW
        
        return face_tensor
    
    def _estimate_pose_pnp(
        self, 
        landmarks: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:
        """Estimate pose using PnP algorithm with facial landmarks."""
        try:
            if len(landmarks) < 6:
                return None
            
            # Map facial landmarks to 3D model points
            # This mapping depends on the landmark format (68-point, MediaPipe, etc.)
            if len(landmarks) >= 68:  # dlib 68-point model
                image_points = np.array([
                    landmarks[30],  # Nose tip
                    landmarks[8],   # Chin
                    landmarks[36],  # Left eye left corner
                    landmarks[45],  # Right eye right corner
                    landmarks[48],  # Left mouth corner
                    landmarks[54]   # Right mouth corner
                ], dtype=np.float64)
            else:
                # Use first 6 points as approximation
                image_points = landmarks[:6].astype(np.float64)
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points_3d,
                image_points,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if not success:
                return None
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Convert rotation matrix to Euler angles
            pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
            
            return {
                'euler_angles': (pitch, yaw, roll),
                'rotation_matrix': rotation_matrix,
                'translation_vector': translation_vector.flatten(),
                'rotation_vector': rotation_vector.flatten()
            }
            
        except Exception as e:
            logger.error(f"Error in PnP pose estimation: {e}")
            return None
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (in degrees)."""
        # Extract Euler angles from rotation matrix
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        # Convert to degrees
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)
        
        return pitch, yaw, roll
    
    def _apply_temporal_smoothing(
        self, 
        current_pose: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Apply temporal smoothing to reduce jitter."""
        if len(self.pose_history) == 0:
            return current_pose
        
        # Simple exponential moving average
        alpha = 0.3  # Smoothing factor
        
        recent_poses = list(self.pose_history)
        if len(recent_poses) > 0:
            prev_pose = recent_poses[-1]
            
            smoothed_pitch = alpha * current_pose[0] + (1 - alpha) * prev_pose[0]
            smoothed_yaw = alpha * current_pose[1] + (1 - alpha) * prev_pose[1]
            smoothed_roll = alpha * current_pose[2] + (1 - alpha) * prev_pose[2]
            
            return smoothed_pitch, smoothed_yaw, smoothed_roll
        
        return current_pose
    
    def set_camera_parameters(
        self, 
        camera_matrix: np.ndarray, 
        dist_coeffs: np.ndarray
    ):
        """Set camera calibration parameters for geometric pose estimation."""
        self.camera_matrix = camera_matrix.copy()
        self.dist_coeffs = dist_coeffs.copy()
        logger.info("Camera parameters updated")
    
    def calibrate_camera(self, calibration_images: List[np.ndarray]) -> bool:
        """
        Perform camera calibration using chessboard patterns.
        
        Args:
            calibration_images: List of calibration images
            
        Returns:
            Success status
        """
        # This is a simplified version - in practice you'd use proper calibration
        logger.warning("Camera calibration not implemented. Using default parameters.")
        return False
    
    def get_attention_direction(
        self, 
        pose_angles: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """
        Analyze attention direction based on head pose.
        
        Args:
            pose_angles: (pitch, yaw, roll) in degrees
            
        Returns:
            Attention analysis results
        """
        pitch, yaw, roll = pose_angles
        
        # Define attention zones
        attention_zones = {
            'center': {'yaw': (-15, 15), 'pitch': (-10, 10)},
            'left': {'yaw': (-45, -15), 'pitch': (-20, 20)},
            'right': {'yaw': (15, 45), 'pitch': (-20, 20)},
            'up': {'yaw': (-30, 30), 'pitch': (-30, -10)},
            'down': {'yaw': (-30, 30), 'pitch': (10, 30)},
            'away': {'yaw': (-180, -45) + (45, 180), 'pitch': (-90, 90)}
        }
        
        # Determine primary attention direction
        attention_direction = 'unknown'
        
        for zone, ranges in attention_zones.items():
            if zone == 'away':
                if yaw < -45 or yaw > 45:
                    attention_direction = zone
                    break
            else:
                yaw_range = ranges['yaw']
                pitch_range = ranges['pitch']
                
                if (yaw_range[0] <= yaw <= yaw_range[1] and 
                    pitch_range[0] <= pitch <= pitch_range[1]):
                    attention_direction = zone
                    break
        
        # Calculate attention score (higher = more focused forward)
        yaw_score = max(0, 1 - abs(yaw) / 45.0)
        pitch_score = max(0, 1 - abs(pitch) / 30.0)
        attention_score = (yaw_score + pitch_score) / 2.0
        
        return {
            'direction': attention_direction,
            'attention_score': attention_score,
            'yaw_deviation': abs(yaw),
            'pitch_deviation': abs(pitch),
            'is_focused': attention_direction == 'center' and attention_score > 0.7
        }
    
    def get_pose_statistics(self) -> Dict[str, Any]:
        """Get pose estimation statistics."""
        if not self.pose_history:
            return {
                'mean_pose': (0.0, 0.0, 0.0),
                'pose_variance': (0.0, 0.0, 0.0),
                'mean_confidence': 0.0,
                'stability_score': 0.0
            }
        
        poses = np.array(list(self.pose_history))
        confidences = list(self.confidence_history)
        
        mean_pose = np.mean(poses, axis=0)
        pose_variance = np.var(poses, axis=0)
        mean_confidence = np.mean(confidences) if confidences else 0.0
        
        # Stability score (lower variance = higher stability)
        stability_score = 1.0 / (1.0 + np.mean(pose_variance))
        
        return {
            'mean_pose': tuple(mean_pose),
            'pose_variance': tuple(pose_variance),
            'mean_confidence': mean_confidence,
            'stability_score': stability_score,
            'sample_count': len(self.pose_history)
        }
    
    def reset_history(self):
        """Reset pose history."""
        self.pose_history.clear()
        self.confidence_history.clear()
        logger.info("Pose history reset")
    
    def visualize_pose(
        self, 
        frame: np.ndarray, 
        results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Visualize head pose estimation results on frame.
        
        Args:
            frame: Input frame
            results: Pose estimation results
            
        Returns:
            Frame with pose visualization
        """
        vis_frame = frame.copy()
        
        if not results['valid']:
            return vis_frame
        
        face_bbox = results.get('face_bbox')
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            
            # Draw face bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate face center
            face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Draw pose axes
            pitch, yaw, roll = results['pitch'], results['yaw'], results['roll']
            
            if 'rotation_matrix' in results:
                vis_frame = draw_pose_axes(
                    vis_frame, 
                    results['rotation_matrix'],
                    results.get('translation_vector', np.array([0, 0, 400])),
                    self.camera_matrix,
                    self.dist_coeffs,
                    face_center
                )
            
            # Draw pose information
            pose_text = f"Pitch: {pitch:.1f}° Yaw: {yaw:.1f}° Roll: {roll:.1f}°"
            cv2.putText(vis_frame, pose_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence
            conf_text = f"Conf: {results['confidence']:.2f}"
            cv2.putText(vis_frame, conf_text, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw attention direction
            attention = self.get_attention_direction((pitch, yaw, roll))
            attention_text = f"Attention: {attention['direction']} ({attention['attention_score']:.2f})"
            cv2.putText(vis_frame, attention_text, (x1, y2 + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis_frame
