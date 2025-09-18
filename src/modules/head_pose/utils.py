"""
Head Pose Estimation Utilities

Utility functions for head pose estimation including visualization,
smoothing, and coordinate transformations.
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Optional
from collections import deque


def draw_pose_axes(image: np.ndarray, 
                   rotation_vector: np.ndarray, 
                   translation_vector: np.ndarray,
                   camera_matrix: np.ndarray, 
                   dist_coeffs: np.ndarray,
                   axis_length: float = 100.0) -> np.ndarray:
    """
    Draw pose axes on image.
    
    Args:
        image: Input image
        rotation_vector: Rotation vector from PnP
        translation_vector: Translation vector from PnP
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Camera distortion coefficients
        axis_length: Length of axes in pixels
        
    Returns:
        Image with pose axes drawn
    """
    # Define 3D axes points
    axes_points = np.array([
        [0, 0, 0],                    # Origin
        [axis_length, 0, 0],          # X-axis (red)
        [0, axis_length, 0],          # Y-axis (green)
        [0, 0, -axis_length]          # Z-axis (blue)
    ], dtype=np.float32)
    
    # Project 3D points to image plane
    projected_points, _ = cv2.projectPoints(
        axes_points, rotation_vector, translation_vector,
        camera_matrix, dist_coeffs
    )
    
    projected_points = projected_points.reshape(-1, 2).astype(int)
    
    # Draw axes
    origin = tuple(projected_points[0])
    x_axis = tuple(projected_points[1])
    y_axis = tuple(projected_points[2])
    z_axis = tuple(projected_points[3])
    
    result_image = image.copy()
    
    # Draw axes lines
    cv2.arrowedLine(result_image, origin, x_axis, (0, 0, 255), 3)  # X-axis: Red
    cv2.arrowedLine(result_image, origin, y_axis, (0, 255, 0), 3)  # Y-axis: Green
    cv2.arrowedLine(result_image, origin, z_axis, (255, 0, 0), 3)  # Z-axis: Blue
    
    return result_image


def euler_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        yaw: Yaw angle in radians
        pitch: Pitch angle in radians
        roll: Roll angle in radians
        
    Returns:
        3x3 rotation matrix
    """
    # Rotation matrices for each axis
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Combined rotation matrix (ZYX order)
    R = R_yaw @ R_pitch @ R_roll
    
    return R


def smooth_pose_estimates(pose_history: deque, current_pose: Tuple[float, float, float],
                         smoothing_factor: float = 0.7) -> Tuple[float, float, float]:
    """
    Apply temporal smoothing to pose estimates.
    
    Args:
        pose_history: Deque of previous pose estimates
        current_pose: Current pose estimate (yaw, pitch, roll)
        smoothing_factor: Smoothing factor (0-1)
        
    Returns:
        Smoothed pose estimate
    """
    if len(pose_history) == 0:
        return current_pose
    
    prev_yaw, prev_pitch, prev_roll = pose_history[-1]
    curr_yaw, curr_pitch, curr_roll = current_pose
    
    # Handle angle wrapping for yaw
    yaw_diff = curr_yaw - prev_yaw
    if yaw_diff > 180:
        curr_yaw -= 360
    elif yaw_diff < -180:
        curr_yaw += 360
    
    # Apply exponential smoothing
    smoothed_yaw = smoothing_factor * curr_yaw + (1 - smoothing_factor) * prev_yaw
    smoothed_pitch = smoothing_factor * curr_pitch + (1 - smoothing_factor) * prev_pitch
    smoothed_roll = smoothing_factor * curr_roll + (1 - smoothing_factor) * prev_roll
    
    return smoothed_yaw, smoothed_pitch, smoothed_roll


def calculate_pose_stability(pose_history: List[Tuple[float, float, float]]) -> float:
    """
    Calculate pose stability metric.
    
    Args:
        pose_history: List of recent pose estimates
        
    Returns:
        Stability score (lower is more stable)
    """
    if len(pose_history) < 2:
        return 0.0
    
    # Calculate standard deviations
    yaw_values = [pose[0] for pose in pose_history]
    pitch_values = [pose[1] for pose in pose_history]
    roll_values = [pose[2] for pose in pose_history]
    
    yaw_std = np.std(yaw_values)
    pitch_std = np.std(pitch_values)
    roll_std = np.std(roll_values)
    
    # Combined stability metric
    stability = (yaw_std + pitch_std + roll_std) / 3.0
    
    return stability


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-180, 180] range.
    
    Args:
        angle: Angle in degrees
        
    Returns:
        Normalized angle
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the smallest difference between two angles.
    
    Args:
        angle1: First angle in degrees
        angle2: Second angle in degrees
        
    Returns:
        Angle difference in degrees
    """
    diff = angle1 - angle2
    return normalize_angle(diff)


def pose_distance(pose1: Tuple[float, float, float], 
                 pose2: Tuple[float, float, float]) -> float:
    """
    Calculate distance between two pose estimates.
    
    Args:
        pose1: First pose (yaw, pitch, roll)
        pose2: Second pose (yaw, pitch, roll)
        
    Returns:
        Euclidean distance between poses
    """
    yaw_diff = angle_difference(pose1[0], pose2[0])
    pitch_diff = angle_difference(pose1[1], pose2[1])
    roll_diff = angle_difference(pose1[2], pose2[2])
    
    distance = math.sqrt(yaw_diff**2 + pitch_diff**2 + roll_diff**2)
    
    return distance


def draw_pose_info(image: np.ndarray, 
                  yaw: float, pitch: float, roll: float,
                  confidence: float = None,
                  position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Draw pose information text on image.
    
    Args:
        image: Input image
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        roll: Roll angle in degrees
        confidence: Optional confidence score
        position: Text position (x, y)
        
    Returns:
        Image with pose information
    """
    result_image = image.copy()
    x, y = position
    
    # Draw pose angles
    cv2.putText(result_image, f"Yaw: {yaw:.1f}°", 
               (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(result_image, f"Pitch: {pitch:.1f}°", 
               (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(result_image, f"Roll: {roll:.1f}°", 
               (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if confidence is not None:
        cv2.putText(result_image, f"Conf: {confidence:.2f}", 
                   (x, y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image


def get_default_camera_matrix(image_width: int, image_height: int) -> np.ndarray:
    """
    Get default camera matrix for given image dimensions.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        3x3 camera matrix
    """
    # Approximate focal length based on image size
    focal_length = max(image_width, image_height)
    
    camera_matrix = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return camera_matrix


def validate_pose_estimate(yaw: float, pitch: float, roll: float,
                          max_yaw: float = 90.0,
                          max_pitch: float = 60.0,
                          max_roll: float = 45.0) -> bool:
    """
    Validate if pose estimate is within reasonable bounds.
    
    Args:
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        roll: Roll angle in degrees
        max_yaw: Maximum allowed yaw angle
        max_pitch: Maximum allowed pitch angle
        max_roll: Maximum allowed roll angle
        
    Returns:
        True if pose is valid
    """
    return (abs(yaw) <= max_yaw and 
            abs(pitch) <= max_pitch and 
            abs(roll) <= max_roll)

