"""
Head Pose Estimation Module

This module provides head pose estimation capabilities using various methods
including PnP algorithm, deep learning models, and hybrid approaches.
Trained and evaluated on BIWI Kinect Head Pose Dataset.
"""

from .head_pose_estimator import HeadPoseEstimator
from .pose_calculator import calculate_head_pose_pnp, estimate_pose_from_landmarks
from .models import HeadPoseNet, load_pretrained_head_pose_model
from .utils import draw_pose_axes, euler_to_rotation_matrix

__all__ = [
    'HeadPoseEstimator', 
    'calculate_head_pose_pnp', 
    'estimate_pose_from_landmarks',
    'HeadPoseNet', 
    'load_pretrained_head_pose_model',
    'draw_pose_axes', 
    'euler_to_rotation_matrix'
]


