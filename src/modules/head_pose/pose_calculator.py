"""
Head Pose Calculation Utilities

Implements various methods for calculating head pose from facial landmarks
and images, including PnP algorithm and geometric approaches.
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional, List


def calculate_head_pose_pnp(landmarks: np.ndarray, 
                           camera_matrix: np.ndarray,
                           dist_coeffs: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Calculate head pose using PnP algorithm.
    
    Args:
        landmarks: 68 facial landmarks
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Camera distortion coefficients
        
    Returns:
        Tuple of (success, rotation_vector, translation_vector)
    """
    # 3D model points (canonical face model in mm)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float32)
    
    # 2D image points from landmarks
    if len(landmarks) >= 68:
        image_points = np.array([
            landmarks[30],    # Nose tip
            landmarks[8],     # Chin
            landmarks[36],    # Left eye left corner
            landmarks[45],    # Right eye right corner
            landmarks[48],    # Left mouth corner
            landmarks[54]     # Right mouth corner
        ], dtype=np.float32)
    else:
        return False, None, None
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    return success, rotation_vector, translation_vector


def estimate_pose_from_landmarks(landmarks: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate head pose from facial landmarks using geometric approach.
    
    Args:
        landmarks: 68 facial landmarks
        
    Returns:
        Tuple of (yaw, pitch, roll) in degrees
    """
    if len(landmarks) < 68:
        return 0.0, 0.0, 0.0
    
    # Key points
    nose_tip = landmarks[30]
    chin = landmarks[8]
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    left_mouth = landmarks[48]
    right_mouth = landmarks[54]
    
    # Calculate face center
    face_center = np.mean([left_eye, right_eye, left_mouth, right_mouth], axis=0)
    
    # Calculate yaw (horizontal rotation)
    eye_center = (left_eye + right_eye) / 2
    nose_to_eye_center = eye_center - nose_tip
    yaw = math.degrees(math.atan2(nose_to_eye_center[0], abs(nose_to_eye_center[1]) + 1e-6))
    
    # Calculate pitch (vertical rotation)
    nose_to_chin = chin - nose_tip
    pitch = math.degrees(math.atan2(nose_to_chin[1], abs(nose_to_chin[0]) + 1e-6))
    
    # Calculate roll (tilt rotation)
    eye_diff = right_eye - left_eye
    roll = math.degrees(math.atan2(eye_diff[1], eye_diff[0]))
    
    return yaw, pitch, roll


def rotation_vector_to_euler_angles(rvec: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation vector to Euler angles.
    
    Args:
        rvec: Rotation vector from cv2.solvePnP
        
    Returns:
        Tuple of (yaw, pitch, roll) in degrees
    """
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # Extract Euler angles from rotation matrix
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + 
                   rotation_matrix[1, 0] * rotation_matrix[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    else:
        yaw = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0
    
    # Convert to degrees
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def euler_angles_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees  
        roll: Roll angle in degrees
        
    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    
    # Rotation matrices for each axis
    R_yaw = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
        [math.sin(yaw_rad), math.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    R_pitch = np.array([
        [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
        [0, 1, 0],
        [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
    ])
    
    R_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll_rad), -math.sin(roll_rad)],
        [0, math.sin(roll_rad), math.cos(roll_rad)]
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    
    return R

