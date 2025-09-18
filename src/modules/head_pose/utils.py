"""
Utility functions for head pose estimation.
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def euler_angles_to_rotation_matrix(
    pitch: float, 
    yaw: float, 
    roll: float, 
    degrees: bool = True
) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        pitch: Rotation around X-axis
        yaw: Rotation around Y-axis  
        roll: Rotation around Z-axis
        degrees: Whether angles are in degrees (True) or radians (False)
        
    Returns:
        3x3 rotation matrix
    """
    if degrees:
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        roll = math.radians(roll)
    
    # Rotation matrices for each axis
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])
    
    R_y = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])
    
    R_z = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix (order: Z * Y * X)
    R = R_z @ R_y @ R_x
    
    return R


def rotation_matrix_to_euler_angles(
    R: np.ndarray, 
    degrees: bool = True
) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles.
    
    Args:
        R: 3x3 rotation matrix
        degrees: Whether to return angles in degrees
        
    Returns:
        Tuple of (pitch, yaw, roll)
    """
    # Check for gimbal lock
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0
    
    if degrees:
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        roll = math.degrees(roll)
    
    return pitch, yaw, roll


def project_3d_points(
    points_3d: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: 3D points [N, 3]
        rotation_matrix: 3x3 rotation matrix
        translation_vector: 3x1 translation vector
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        
    Returns:
        2D projected points [N, 2]
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))
    
    # Convert rotation matrix to rotation vector
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    
    # Project points
    projected_points, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        rotation_vector,
        translation_vector.reshape(3, 1),
        camera_matrix,
        dist_coeffs
    )
    
    return projected_points.reshape(-1, 2)


def draw_pose_axes(
    image: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    origin: Optional[Tuple[int, int]] = None,
    axis_length: float = 100.0
) -> np.ndarray:
    """
    Draw 3D pose axes on image.
    
    Args:
        image: Input image
        rotation_matrix: 3x3 rotation matrix
        translation_vector: Translation vector
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        origin: 2D origin point (if None, uses image center)
        axis_length: Length of axes in pixels
        
    Returns:
        Image with pose axes drawn
    """
    result_image = image.copy()
    
    if origin is None:
        origin = (image.shape[1] // 2, image.shape[0] // 2)
    
    # Define 3D axis points
    axis_points_3d = np.array([
        [0, 0, 0],                    # Origin
        [axis_length, 0, 0],          # X-axis (red)
        [0, axis_length, 0],          # Y-axis (green)
        [0, 0, axis_length]           # Z-axis (blue)
    ], dtype=np.float32)
    
    try:
        # Project 3D points to 2D
        projected_points = project_3d_points(
            axis_points_3d,
            rotation_matrix,
            translation_vector,
            camera_matrix,
            dist_coeffs
        )
        
        # Adjust projected points relative to origin
        if len(projected_points) == 4:
            origin_2d = projected_points[0].astype(int)
            x_axis_2d = projected_points[1].astype(int)
            y_axis_2d = projected_points[2].astype(int)
            z_axis_2d = projected_points[3].astype(int)
            
            # Offset to desired origin
            offset = np.array(origin) - origin_2d
            x_axis_2d += offset
            y_axis_2d += offset
            z_axis_2d += offset
            origin_2d = np.array(origin, dtype=int)
            
            # Draw axes
            cv2.arrowedLine(result_image, tuple(origin_2d), tuple(x_axis_2d), 
                           (0, 0, 255), 3, tipLength=0.3)  # X-axis: Red
            cv2.arrowedLine(result_image, tuple(origin_2d), tuple(y_axis_2d), 
                           (0, 255, 0), 3, tipLength=0.3)  # Y-axis: Green
            cv2.arrowedLine(result_image, tuple(origin_2d), tuple(z_axis_2d), 
                           (255, 0, 0), 3, tipLength=0.3)  # Z-axis: Blue
            
            # Draw labels
            cv2.putText(result_image, 'X', tuple(x_axis_2d + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(result_image, 'Y', tuple(y_axis_2d + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, 'Z', tuple(z_axis_2d + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    except Exception as e:
        logger.error(f"Error drawing pose axes: {e}")
    
    return result_image


def draw_pose_cube(
    image: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    origin: Optional[Tuple[int, int]] = None,
    cube_size: float = 100.0
) -> np.ndarray:
    """
    Draw a 3D cube to visualize pose.
    
    Args:
        image: Input image
        rotation_matrix: 3x3 rotation matrix
        translation_vector: Translation vector
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        origin: 2D origin point
        cube_size: Size of the cube
        
    Returns:
        Image with pose cube drawn
    """
    result_image = image.copy()
    
    if origin is None:
        origin = (image.shape[1] // 2, image.shape[0] // 2)
    
    # Define 3D cube points
    s = cube_size / 2
    cube_points_3d = np.array([
        [-s, -s, -s],  # 0: bottom-left-back
        [s, -s, -s],   # 1: bottom-right-back
        [s, s, -s],    # 2: top-right-back
        [-s, s, -s],   # 3: top-left-back
        [-s, -s, s],   # 4: bottom-left-front
        [s, -s, s],    # 5: bottom-right-front
        [s, s, s],     # 6: top-right-front
        [-s, s, s]     # 7: top-left-front
    ], dtype=np.float32)
    
    try:
        # Project 3D points to 2D
        projected_points = project_3d_points(
            cube_points_3d,
            rotation_matrix,
            translation_vector,
            camera_matrix,
            dist_coeffs
        )
        
        # Adjust to origin
        if len(projected_points) == 8:
            cube_center = np.mean(projected_points, axis=0)
            offset = np.array(origin) - cube_center
            projected_points += offset
            
            points_2d = projected_points.astype(int)
            
            # Define cube edges
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
            ]
            
            # Draw edges
            for start_idx, end_idx in edges:
                start_point = tuple(points_2d[start_idx])
                end_point = tuple(points_2d[end_idx])
                cv2.line(result_image, start_point, end_point, (0, 255, 255), 2)
            
            # Highlight front face
            front_face = [4, 5, 6, 7, 4]
            for i in range(len(front_face) - 1):
                start_point = tuple(points_2d[front_face[i]])
                end_point = tuple(points_2d[front_face[i + 1]])
                cv2.line(result_image, start_point, end_point, (0, 0, 255), 3)
    
    except Exception as e:
        logger.error(f"Error drawing pose cube: {e}")
    
    return result_image


def calculate_pose_accuracy(
    predicted_angles: np.ndarray,
    ground_truth_angles: np.ndarray,
    metric: str = 'mae'
) -> float:
    """
    Calculate pose estimation accuracy.
    
    Args:
        predicted_angles: Predicted angles [N, 3] (pitch, yaw, roll)
        ground_truth_angles: Ground truth angles [N, 3]
        metric: Accuracy metric ('mae', 'mse', 'angular_error')
        
    Returns:
        Accuracy score
    """
    if predicted_angles.shape != ground_truth_angles.shape:
        raise ValueError("Predicted and ground truth shapes must match")
    
    if metric == 'mae':
        # Mean Absolute Error
        return np.mean(np.abs(predicted_angles - ground_truth_angles))
    
    elif metric == 'mse':
        # Mean Squared Error
        return np.mean((predicted_angles - ground_truth_angles) ** 2)
    
    elif metric == 'angular_error':
        # Mean angular error between rotation matrices
        errors = []
        for pred, gt in zip(predicted_angles, ground_truth_angles):
            pred_matrix = euler_angles_to_rotation_matrix(*pred)
            gt_matrix = euler_angles_to_rotation_matrix(*gt)
            
            # Calculate angular difference
            relative_rotation = pred_matrix.T @ gt_matrix
            trace = np.trace(relative_rotation)
            
            # Clamp trace to valid range for arccos
            trace = np.clip(trace, -1, 3)
            angular_error = math.degrees(math.acos((trace - 1) / 2))
            errors.append(angular_error)
        
        return np.mean(errors)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def normalize_angles(angles: np.ndarray, degrees: bool = True) -> np.ndarray:
    """
    Normalize angles to [-180, 180] degrees or [-π, π] radians.
    
    Args:
        angles: Input angles
        degrees: Whether angles are in degrees
        
    Returns:
        Normalized angles
    """
    if degrees:
        return ((angles + 180) % 360) - 180
    else:
        return ((angles + np.pi) % (2 * np.pi)) - np.pi


def smooth_pose_sequence(
    pose_sequence: List[Tuple[float, float, float]],
    window_size: int = 5,
    method: str = 'moving_average'
) -> List[Tuple[float, float, float]]:
    """
    Smooth pose sequence to reduce jitter.
    
    Args:
        pose_sequence: Sequence of (pitch, yaw, roll) tuples
        window_size: Smoothing window size
        method: Smoothing method
        
    Returns:
        Smoothed pose sequence
    """
    if len(pose_sequence) < window_size:
        return pose_sequence
    
    smoothed_sequence = []
    
    for i in range(len(pose_sequence)):
        if method == 'moving_average':
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(pose_sequence), i + window_size // 2 + 1)
            
            window_poses = pose_sequence[start_idx:end_idx]
            
            # Average each angle component
            avg_pitch = np.mean([pose[0] for pose in window_poses])
            avg_yaw = np.mean([pose[1] for pose in window_poses])
            avg_roll = np.mean([pose[2] for pose in window_poses])
            
            smoothed_sequence.append((avg_pitch, avg_yaw, avg_roll))
        
        elif method == 'exponential':
            if i == 0:
                smoothed_sequence.append(pose_sequence[i])
            else:
                alpha = 0.3  # Smoothing factor
                prev_smooth = smoothed_sequence[-1]
                current = pose_sequence[i]
                
                smooth_pitch = alpha * current[0] + (1 - alpha) * prev_smooth[0]
                smooth_yaw = alpha * current[1] + (1 - alpha) * prev_smooth[1]
                smooth_roll = alpha * current[2] + (1 - alpha) * prev_smooth[2]
                
                smoothed_sequence.append((smooth_pitch, smooth_yaw, smooth_roll))
        
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed_sequence


def detect_head_movements(
    pose_sequence: List[Tuple[float, float, float]],
    threshold: float = 10.0,  # degrees
    min_duration: int = 3     # frames
) -> List[Dict[str, any]]:
    """
    Detect significant head movements in pose sequence.
    
    Args:
        pose_sequence: Sequence of pose angles
        threshold: Movement threshold in degrees
        min_duration: Minimum duration for valid movement
        
    Returns:
        List of detected movements
    """
    if len(pose_sequence) < min_duration:
        return []
    
    movements = []
    current_movement = None
    
    for i in range(1, len(pose_sequence)):
        prev_pose = pose_sequence[i - 1]
        curr_pose = pose_sequence[i]
        
        # Calculate angular difference
        pitch_diff = abs(curr_pose[0] - prev_pose[0])
        yaw_diff = abs(curr_pose[1] - prev_pose[1])
        roll_diff = abs(curr_pose[2] - prev_pose[2])
        
        max_diff = max(pitch_diff, yaw_diff, roll_diff)
        
        if max_diff > threshold:
            if current_movement is None:
                # Start new movement
                current_movement = {
                    'start_frame': i - 1,
                    'end_frame': i,
                    'movement_type': 'unknown',
                    'max_displacement': max_diff,
                    'primary_axis': 'pitch' if pitch_diff == max_diff else 
                                   'yaw' if yaw_diff == max_diff else 'roll'
                }
            else:
                # Continue current movement
                current_movement['end_frame'] = i
                current_movement['max_displacement'] = max(
                    current_movement['max_displacement'], max_diff
                )
        else:
            if current_movement is not None:
                # End current movement if it's long enough
                duration = current_movement['end_frame'] - current_movement['start_frame']
                if duration >= min_duration:
                    # Classify movement type
                    start_pose = pose_sequence[current_movement['start_frame']]
                    end_pose = pose_sequence[current_movement['end_frame']]
                    
                    yaw_change = end_pose[1] - start_pose[1]
                    pitch_change = end_pose[0] - start_pose[0]
                    
                    if abs(yaw_change) > abs(pitch_change):
                        if yaw_change > 0:
                            current_movement['movement_type'] = 'turn_right'
                        else:
                            current_movement['movement_type'] = 'turn_left'
                    else:
                        if pitch_change > 0:
                            current_movement['movement_type'] = 'look_up'
                        else:
                            current_movement['movement_type'] = 'look_down'
                    
                    movements.append(current_movement)
                
                current_movement = None
    
    return movements


def estimate_attention_focus(
    pose_angles: Tuple[float, float, float],
    screen_bounds: Tuple[float, float, float, float] = (-30, 30, -20, 20)
) -> Dict[str, any]:
    """
    Estimate attention focus based on head pose.
    
    Args:
        pose_angles: (pitch, yaw, roll) in degrees
        screen_bounds: (yaw_min, yaw_max, pitch_min, pitch_max) for screen region
        
    Returns:
        Attention focus analysis
    """
    pitch, yaw, roll = pose_angles
    yaw_min, yaw_max, pitch_min, pitch_max = screen_bounds
    
    # Check if looking at screen
    looking_at_screen = (yaw_min <= yaw <= yaw_max and 
                        pitch_min <= pitch <= pitch_max)
    
    # Calculate attention score (0-1, higher = more focused on screen)
    yaw_score = max(0, 1 - abs(yaw) / 45.0)  # Normalize by ±45°
    pitch_score = max(0, 1 - abs(pitch) / 30.0)  # Normalize by ±30°
    attention_score = (yaw_score + pitch_score) / 2.0
    
    # Determine gaze direction
    if abs(yaw) < 10 and abs(pitch) < 10:
        direction = 'center'
    elif yaw < -15:
        direction = 'left'
    elif yaw > 15:
        direction = 'right'
    elif pitch < -10:
        direction = 'up'
    elif pitch > 10:
        direction = 'down'
    else:
        direction = 'center'
    
    return {
        'looking_at_screen': looking_at_screen,
        'attention_score': attention_score,
        'gaze_direction': direction,
        'yaw_deviation': abs(yaw),
        'pitch_deviation': abs(pitch),
        'distraction_level': 1.0 - attention_score
    }
