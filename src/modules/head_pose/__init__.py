"""
Head Pose Estimation Module

This module provides real-time head pose estimation capabilities
using deep learning models trained on the BIWI Kinect Head Pose Dataset.
Head pose is crucial for analyzing attention and distraction patterns.
"""

from .pose_estimator import HeadPoseEstimator
from .models import ResNetPose, EfficientNetPose, PoseNet
from .utils import euler_angles_to_rotation_matrix, rotation_matrix_to_euler_angles
from .utils import project_3d_points, draw_pose_axes, calculate_pose_accuracy
from .dataset import BIWIDataset, CustomPoseDataset

__all__ = [
    'HeadPoseEstimator',
    'ResNetPose',
    'EfficientNetPose', 
    'PoseNet',
    'euler_angles_to_rotation_matrix',
    'rotation_matrix_to_euler_angles',
    'project_3d_points',
    'draw_pose_axes',
    'calculate_pose_accuracy',
    'BIWIDataset',
    'CustomPoseDataset'
]
