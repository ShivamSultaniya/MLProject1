"""
Head Pose Estimation System

Implements real-time head pose estimation using multiple approaches:
1. PnP (Perspective-n-Point) algorithm with facial landmarks
2. Deep learning models trained on BIWI dataset
3. Hybrid approaches combining multiple methods

Outputs head pose in terms of yaw, pitch, and roll angles.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List
import logging
import time
from collections import deque

try:
    from .pose_calculator import calculate_head_pose_pnp, estimate_pose_from_landmarks
    from .models import HeadPoseNet, load_pretrained_head_pose_model
    from .utils import draw_pose_axes, euler_to_rotation_matrix, smooth_pose_estimates
except ImportError:
    # Fallback for when modules can't be found
    def calculate_head_pose_pnp(*args, **kwargs):
        return False, None, None
    def estimate_pose_from_landmarks(*args, **kwargs):
        return 0.0, 0.0, 0.0
    def draw_pose_axes(frame, *args, **kwargs):
        return frame
    def euler_to_rotation_matrix(*args, **kwargs):
        return np.eye(3)
    def smooth_pose_estimates(*args, **kwargs):
        return args[1] if len(args) > 1 else (0, 0, 0)
    
    class HeadPoseNet:
        def __init__(self, *args, **kwargs):
            pass
        def eval(self):
            pass
        def to(self, device):
            return self
    
    def load_pretrained_head_pose_model(*args, **kwargs):
        return HeadPoseNet()


class HeadPoseEstimator:
    """
    Comprehensive head pose estimation system with multiple estimation methods.
    """
    
    def __init__(self, method: str = 'pnp', model_path: Optional[str] = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None,
                 smoothing_window: int = 5):
        """
        Initialize head pose estimator.
        
        Args:
            method: Estimation method ('pnp', 'dnn', 'hybrid')
            model_path: Path to trained model (for DNN method)
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            smoothing_window: Window size for pose smoothing
        """
        self.method = method
        self.model_path = model_path
        self.smoothing_window = smoothing_window
        
        # Camera parameters
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        else:
            # Default camera matrix for typical webcam
            self.camera_matrix = np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ], dtype=np.float32)
        
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs
        else:
            # Assume no distortion by default
            self.dist_coeffs = np.zeros((4, 1))
        
        # Load DNN model if specified
        self.dnn_model = None
        if method in ['dnn', 'hybrid'] and model_path:
            self.dnn_model = load_pretrained_head_pose_model(model_path)
        
        # Initialize face detection
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
            self.dlib_available = True
        except ImportError:
            logging.warning("dlib not available, using OpenCV cascade classifiers")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.dlib_available = False
        
        # Pose history for smoothing
        self.pose_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        
        # 3D model points (canonical face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        self.logger = logging.getLogger(__name__)
    
    def detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect facial landmarks in the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            68 facial landmarks or None if detection fails
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.dlib_available:
            faces = self.detector(gray)
            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda rect: rect.width() * rect.height())
                landmarks = self.predictor(gray, face)
                
                # Convert to numpy array
                coords = np.array([[p.x, p.y] for p in landmarks.parts()])
                return coords
        else:
            # Fallback using OpenCV (simplified landmark detection)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Generate approximate landmarks
                landmarks = self._generate_approximate_landmarks(x, y, w, h)
                return landmarks
        
        return None
    
    def _generate_approximate_landmarks(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Generate approximate 68 landmarks from face bounding box."""
        landmarks = np.zeros((68, 2))
        
        # Key points for pose estimation
        landmarks[30] = [x + w//2, y + h//2]  # Nose tip
        landmarks[8] = [x + w//2, y + h - 10]  # Chin
        landmarks[36] = [x + w//4, y + h//3]   # Left eye corner
        landmarks[45] = [x + 3*w//4, y + h//3]  # Right eye corner
        landmarks[48] = [x + w//3, y + 2*h//3]  # Left mouth corner
        landmarks[54] = [x + 2*w//3, y + 2*h//3]  # Right mouth corner
        
        return landmarks
    
    def estimate_pose(self, frame: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Estimate head pose from frame.
        
        Args:
            frame: Input video frame
            landmarks: Optional pre-detected landmarks
            
        Returns:
            Dictionary containing pose estimation results
        """
        result = {
            'success': False,
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'confidence': 0.0,
            'method_used': self.method,
            'landmarks': None,
            'rotation_vector': None,
            'translation_vector': None
        }
        
        # Detect landmarks if not provided
        if landmarks is None:
            landmarks = self.detect_landmarks(frame)
        
        if landmarks is None:
            return result
        
        result['landmarks'] = landmarks
        
        # Estimate pose based on selected method
        if self.method == 'pnp':
            pose_result = self._estimate_pose_pnp(landmarks)
        elif self.method == 'dnn' and self.dnn_model:
            pose_result = self._estimate_pose_dnn(frame, landmarks)
        elif self.method == 'hybrid':
            pose_result = self._estimate_pose_hybrid(frame, landmarks)
        else:
            # Fallback to PnP
            pose_result = self._estimate_pose_pnp(landmarks)
        
        if pose_result['success']:
            result.update(pose_result)
            
            # Apply smoothing
            if len(self.pose_history) > 0:
                smoothed_pose = self._apply_smoothing(
                    (result['yaw'], result['pitch'], result['roll'])
                )
                result['yaw'], result['pitch'], result['roll'] = smoothed_pose
            
            # Update history
            self.pose_history.append((result['yaw'], result['pitch'], result['roll']))
            self.confidence_history.append(result['confidence'])
        
        return result
    
    def _estimate_pose_pnp(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Estimate pose using PnP algorithm."""
        try:
            # Extract 2D image points for key facial landmarks
            image_points = np.array([
                landmarks[30],    # Nose tip
                landmarks[8],     # Chin
                landmarks[36],    # Left eye left corner
                landmarks[45],    # Right eye right corner
                landmarks[48],    # Left mouth corner
                landmarks[54]     # Right mouth corner
            ], dtype=np.float32)
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                yaw, pitch, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
                
                # Calculate confidence based on reprojection error
                projected_points, _ = cv2.projectPoints(
                    self.model_points, rotation_vector, translation_vector,
                    self.camera_matrix, self.dist_coeffs
                )
                
                reprojection_error = np.mean(np.linalg.norm(
                    image_points - projected_points.reshape(-1, 2), axis=1
                ))
                
                confidence = max(0.0, 1.0 - reprojection_error / 50.0)
                
                return {
                    'success': True,
                    'yaw': float(yaw),
                    'pitch': float(pitch),
                    'roll': float(roll),
                    'confidence': float(confidence),
                    'rotation_vector': rotation_vector,
                    'translation_vector': translation_vector
                }
        
        except Exception as e:
            self.logger.error(f"PnP pose estimation failed: {e}")
        
        return {'success': False}
    
    def _estimate_pose_dnn(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """Estimate pose using deep neural network."""
        if self.dnn_model is None:
            return {'success': False}
        
        try:
            # Preprocess frame for model input
            input_tensor = self._preprocess_for_dnn(frame, landmarks)
            
            with torch.no_grad():
                output = self.dnn_model(input_tensor)
                
                if len(output) == 3:
                    yaw, pitch, roll = output.cpu().numpy()
                    confidence = 0.8  # Default confidence for DNN
                else:
                    # Model might output confidence as well
                    yaw, pitch, roll, confidence = output.cpu().numpy()
            
            return {
                'success': True,
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'confidence': float(confidence)
            }
        
        except Exception as e:
            self.logger.error(f"DNN pose estimation failed: {e}")
        
        return {'success': False}
    
    def _estimate_pose_hybrid(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """Estimate pose using hybrid PnP + DNN approach."""
        # Get estimates from both methods
        pnp_result = self._estimate_pose_pnp(landmarks)
        dnn_result = self._estimate_pose_dnn(frame, landmarks)
        
        # If both succeed, combine them
        if pnp_result['success'] and dnn_result['success']:
            # Weighted average based on confidence
            pnp_weight = pnp_result['confidence']
            dnn_weight = dnn_result['confidence']
            total_weight = pnp_weight + dnn_weight
            
            if total_weight > 0:
                yaw = (pnp_result['yaw'] * pnp_weight + dnn_result['yaw'] * dnn_weight) / total_weight
                pitch = (pnp_result['pitch'] * pnp_weight + dnn_result['pitch'] * dnn_weight) / total_weight
                roll = (pnp_result['roll'] * pnp_weight + dnn_result['roll'] * dnn_weight) / total_weight
                confidence = max(pnp_result['confidence'], dnn_result['confidence'])
                
                return {
                    'success': True,
                    'yaw': float(yaw),
                    'pitch': float(pitch),
                    'roll': float(roll),
                    'confidence': float(confidence),
                    'rotation_vector': pnp_result.get('rotation_vector'),
                    'translation_vector': pnp_result.get('translation_vector')
                }
        
        # Fallback to the better result
        if pnp_result['success']:
            return pnp_result
        elif dnn_result['success']:
            return dnn_result
        
        return {'success': False}
    
    def _preprocess_for_dnn(self, frame: np.ndarray, landmarks: np.ndarray) -> torch.Tensor:
        """Preprocess frame and landmarks for DNN input."""
        # Extract face region based on landmarks
        face_bbox = self._get_face_bbox(landmarks)
        face_region = self._extract_face_region(frame, face_bbox)
        
        # Resize to model input size
        face_resized = cv2.resize(face_region, (224, 224))
        
        # Normalize
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return face_tensor
    
    def _get_face_bbox(self, landmarks: np.ndarray, padding: int = 20) -> Tuple[int, int, int, int]:
        """Get face bounding box from landmarks."""
        min_x = int(np.min(landmarks[:, 0])) - padding
        max_x = int(np.max(landmarks[:, 0])) + padding
        min_y = int(np.min(landmarks[:, 1])) - padding
        max_y = int(np.max(landmarks[:, 1])) + padding
        
        width = max_x - min_x
        height = max_y - min_y
        
        return (max(0, min_x), max(0, min_y), max(1, width), max(1, height))
    
    def _extract_face_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face region from frame."""
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]
        
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        x2 = max(x + 1, min(x + w, frame_w))
        y2 = max(y + 1, min(y + h, frame_h))
        
        return frame[y:y2, x:x2]
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (yaw, pitch, roll)."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)
        
        return yaw, pitch, roll
    
    def _apply_smoothing(self, current_pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply temporal smoothing to pose estimates."""
        if len(self.pose_history) == 0:
            return current_pose
        
        # Simple exponential moving average
        alpha = 0.7  # Smoothing factor
        
        prev_yaw, prev_pitch, prev_roll = self.pose_history[-1]
        curr_yaw, curr_pitch, curr_roll = current_pose
        
        # Handle angle wrapping for yaw
        yaw_diff = curr_yaw - prev_yaw
        if yaw_diff > 180:
            curr_yaw -= 360
        elif yaw_diff < -180:
            curr_yaw += 360
        
        smoothed_yaw = alpha * curr_yaw + (1 - alpha) * prev_yaw
        smoothed_pitch = alpha * curr_pitch + (1 - alpha) * prev_pitch
        smoothed_roll = alpha * curr_roll + (1 - alpha) * prev_roll
        
        return smoothed_yaw, smoothed_pitch, smoothed_roll
    
    def get_average_pose(self, window_size: int = 10) -> Optional[Tuple[float, float, float]]:
        """Get average pose over recent estimates."""
        if len(self.pose_history) == 0:
            return None
        
        recent_poses = list(self.pose_history)[-window_size:]
        
        avg_yaw = sum(pose[0] for pose in recent_poses) / len(recent_poses)
        avg_pitch = sum(pose[1] for pose in recent_poses) / len(recent_poses)
        avg_roll = sum(pose[2] for pose in recent_poses) / len(recent_poses)
        
        return avg_yaw, avg_pitch, avg_roll
    
    def get_pose_stability(self) -> float:
        """Calculate pose stability (lower values = more stable)."""
        if len(self.pose_history) < 3:
            return 0.0
        
        recent_poses = list(self.pose_history)[-10:]
        
        # Calculate standard deviation of angles
        yaw_values = [pose[0] for pose in recent_poses]
        pitch_values = [pose[1] for pose in recent_poses]
        roll_values = [pose[2] for pose in recent_poses]
        
        yaw_std = np.std(yaw_values)
        pitch_std = np.std(pitch_values)
        roll_std = np.std(roll_values)
        
        # Combined stability metric
        stability = (yaw_std + pitch_std + roll_std) / 3.0
        
        return stability
    
    def calibrate_camera(self, calibration_images: List[np.ndarray], 
                        chessboard_size: Tuple[int, int] = (9, 6)) -> bool:
        """
        Calibrate camera using chessboard images.
        
        Args:
            calibration_images: List of chessboard images
            chessboard_size: Size of chessboard (width, height)
            
        Returns:
            True if calibration successful
        """
        try:
            # Prepare object points
            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
            
            objpoints = []  # 3D points in real world space
            imgpoints = []  # 2D points in image plane
            
            for img in calibration_images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                
                if ret:
                    objpoints.append(objp)
                    
                    # Refine corners
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    imgpoints.append(corners2)
            
            if len(objpoints) > 0:
                # Calibrate camera
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )
                
                if ret:
                    self.camera_matrix = camera_matrix
                    self.dist_coeffs = dist_coeffs
                    self.logger.info("Camera calibration successful")
                    return True
        
        except Exception as e:
            self.logger.error(f"Camera calibration failed: {e}")
        
        return False
    
    def visualize_pose(self, frame: np.ndarray, pose_result: Dict[str, Any]) -> np.ndarray:
        """
        Visualize head pose estimation results on frame.
        
        Args:
            frame: Input frame
            pose_result: Result from estimate_pose()
            
        Returns:
            Frame with pose visualization
        """
        result_frame = frame.copy()
        
        if not pose_result['success']:
            cv2.putText(result_frame, "No pose detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result_frame
        
        # Draw facial landmarks if available
        if pose_result['landmarks'] is not None:
            landmarks = pose_result['landmarks']
            for point in landmarks:
                cv2.circle(result_frame, tuple(point.astype(int)), 2, (0, 255, 0), -1)
        
        # Draw pose axes if rotation vector is available
        if pose_result.get('rotation_vector') is not None:
            result_frame = draw_pose_axes(
                result_frame,
                pose_result['rotation_vector'],
                pose_result['translation_vector'],
                self.camera_matrix,
                self.dist_coeffs
            )
        
        # Add text information
        y_offset = 30
        cv2.putText(result_frame, f"Yaw: {pose_result['yaw']:.1f}°",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(result_frame, f"Pitch: {pose_result['pitch']:.1f}°",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(result_frame, f"Roll: {pose_result['roll']:.1f}°",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(result_frame, f"Confidence: {pose_result['confidence']:.2f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def reset(self) -> None:
        """Reset pose estimation state."""
        self.pose_history.clear()
        self.confidence_history.clear()


def create_head_pose_estimator(config: Dict[str, Any]) -> HeadPoseEstimator:
    """
    Factory function to create head pose estimator with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured HeadPoseEstimator instance
    """
    return HeadPoseEstimator(
        method=config.get('method', 'pnp'),
        model_path=config.get('model_path', None),
        camera_matrix=config.get('camera_matrix', None),
        dist_coeffs=config.get('dist_coeffs', None),
        smoothing_window=config.get('smoothing_window', 5)
    )


