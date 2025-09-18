"""
Head Pose Estimation Models

Neural network models for head pose estimation, including architectures
compatible with BIWI Kinect Head Pose Dataset and other benchmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class HeadPoseNet(nn.Module):
    """
    Convolutional Neural Network for head pose estimation.
    
    Outputs yaw, pitch, and roll angles from face images.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224), 
                 num_outputs: int = 3, backbone: str = 'resnet18'):
        """
        Initialize HeadPoseNet.
        
        Args:
            input_size: Input image size (height, width)
            num_outputs: Number of output angles (3 for yaw, pitch, roll)
            backbone: Backbone architecture ('resnet18', 'mobilenet', 'custom')
        """
        super(HeadPoseNet, self).__init__()
        
        self.input_size = input_size
        self.num_outputs = num_outputs
        self.backbone = backbone
        
        if backbone == 'resnet18':
            self._build_resnet18_backbone()
        elif backbone == 'mobilenet':
            self._build_mobilenet_backbone()
        else:
            self._build_custom_backbone()
    
    def _build_resnet18_backbone(self):
        """Build ResNet18-based backbone."""
        # Import ResNet18 from torchvision
        try:
            from torchvision.models import resnet18
            self.backbone_net = resnet18(pretrained=False)
            
            # Modify final layer for pose estimation
            self.backbone_net.fc = nn.Linear(self.backbone_net.fc.in_features, 512)
            
            # Add pose estimation head
            self.pose_head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_outputs)
            )
            
        except ImportError:
            # Fallback to custom backbone if torchvision not available
            self._build_custom_backbone()
    
    def _build_mobilenet_backbone(self):
        """Build MobileNet-based backbone."""
        try:
            from torchvision.models import mobilenet_v2
            self.backbone_net = mobilenet_v2(pretrained=False)
            
            # Modify classifier
            self.backbone_net.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.backbone_net.last_channel, 512),
                nn.ReLU(inplace=True)
            )
            
            # Add pose estimation head
            self.pose_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.num_outputs)
            )
            
        except ImportError:
            # Fallback to custom backbone
            self._build_custom_backbone()
    
    def _build_custom_backbone(self):
        """Build custom CNN backbone."""
        self.backbone_net = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate feature size
        feature_size = 256 * 4 * 4
        
        # Pose estimation head
        self.pose_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_outputs)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        # Extract features
        if hasattr(self, 'backbone_net') and hasattr(self.backbone_net, 'features'):
            # For MobileNet-style architectures
            features = self.backbone_net.features(x)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            features = self.backbone_net.classifier(features)
        elif hasattr(self, 'backbone_net') and hasattr(self.backbone_net, 'conv1'):
            # For ResNet-style architectures
            features = self.backbone_net(x)
        else:
            # For custom backbone
            features = self.backbone_net(x)
            features = features.view(features.size(0), -1)
        
        # Pose estimation
        pose_output = self.pose_head(features)
        
        return pose_output


class MultiTaskHeadPoseNet(nn.Module):
    """
    Multi-task network that estimates head pose and additional facial attributes.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        super(MultiTaskHeadPoseNet, self).__init__()
        
        self.input_size = input_size
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        feature_size = 512 * 4 * 4
        
        # Head pose estimation branch
        self.pose_branch = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)  # yaw, pitch, roll
        )
        
        # Gaze estimation branch (optional)
        self.gaze_branch = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)  # gaze yaw, pitch
        )
        
        # Confidence estimation branch
        self.confidence_branch = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through multi-task network."""
        # Extract shared features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Task-specific outputs
        pose = self.pose_branch(features)
        gaze = self.gaze_branch(features)
        confidence = self.confidence_branch(features)
        
        return {
            'pose': pose,
            'gaze': gaze,
            'confidence': confidence
        }


class HeadPoseLoss(nn.Module):
    """Custom loss function for head pose estimation."""
    
    def __init__(self, angular_weight: float = 1.0, 
                 confidence_weight: float = 0.1,
                 use_geodesic_loss: bool = False):
        super(HeadPoseLoss, self).__init__()
        
        self.angular_weight = angular_weight
        self.confidence_weight = confidence_weight
        self.use_geodesic_loss = use_geodesic_loss
    
    def forward(self, predicted_pose, target_pose, confidence=None):
        """
        Compute head pose estimation loss.
        
        Args:
            predicted_pose: Predicted pose angles [B, 3]
            target_pose: Ground truth pose angles [B, 3]
            confidence: Prediction confidence [B, 1] (optional)
        """
        if self.use_geodesic_loss:
            # Geodesic loss on rotation matrices
            loss = self._geodesic_loss(predicted_pose, target_pose)
        else:
            # Simple angular loss
            loss = F.mse_loss(predicted_pose, target_pose)
        
        total_loss = self.angular_weight * loss
        
        # Confidence regularization if available
        if confidence is not None:
            # Encourage high confidence for accurate predictions
            angular_error = torch.norm(predicted_pose - target_pose, dim=1, keepdim=True)
            accuracy = torch.exp(-angular_error / 30.0)  # 30 degrees normalization
            confidence_loss = F.mse_loss(confidence, accuracy)
            total_loss += self.confidence_weight * confidence_loss
        
        return total_loss
    
    def _geodesic_loss(self, pred_angles, true_angles):
        """Compute geodesic loss on rotation matrices."""
        # Convert angles to rotation matrices
        pred_matrices = self._euler_to_rotation_matrix(pred_angles)
        true_matrices = self._euler_to_rotation_matrix(true_angles)
        
        # Geodesic distance between rotation matrices
        R_diff = torch.bmm(pred_matrices, true_matrices.transpose(-1, -2))
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
        
        # Clamp to avoid numerical issues
        trace = torch.clamp(trace, -1.0, 3.0)
        
        # Geodesic distance
        geodesic_distance = torch.acos((trace - 1) / 2)
        
        return torch.mean(geodesic_distance)
    
    def _euler_to_rotation_matrix(self, angles):
        """Convert Euler angles to rotation matrices."""
        batch_size = angles.size(0)
        
        # Extract angles
        yaw = angles[:, 0]
        pitch = angles[:, 1] 
        roll = angles[:, 2]
        
        # Convert to radians
        yaw = yaw * np.pi / 180
        pitch = pitch * np.pi / 180
        roll = roll * np.pi / 180
        
        # Create rotation matrices
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        cos_pitch = torch.cos(pitch)
        sin_pitch = torch.sin(pitch)
        cos_roll = torch.cos(roll)
        sin_roll = torch.sin(roll)
        
        # Rotation matrix (ZYX order)
        R = torch.zeros(batch_size, 3, 3, device=angles.device)
        
        R[:, 0, 0] = cos_yaw * cos_pitch
        R[:, 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
        R[:, 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
        
        R[:, 1, 0] = sin_yaw * cos_pitch
        R[:, 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
        R[:, 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
        
        R[:, 2, 0] = -sin_pitch
        R[:, 2, 1] = cos_pitch * sin_roll
        R[:, 2, 2] = cos_pitch * cos_roll
        
        return R


def load_pretrained_head_pose_model(model_path: str, 
                                   device: str = 'cpu',
                                   backbone: str = 'resnet18') -> nn.Module:
    """
    Load a pretrained head pose estimation model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        backbone: Model backbone architecture
    
    Returns:
        Loaded model
    """
    model = HeadPoseNet(backbone=backbone)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained head pose model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
        print("Using randomly initialized model")
    
    model.to(device)
    model.eval()
    return model


def create_head_pose_model(config: dict) -> nn.Module:
    """
    Factory function to create head pose model from configuration.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Head pose model
    """
    model_type = config.get('type', 'HeadPoseNet')
    backbone = config.get('backbone', 'resnet18')
    input_size = config.get('input_size', (224, 224))
    
    if model_type == 'MultiTaskHeadPoseNet':
        model = MultiTaskHeadPoseNet(input_size=input_size)
    else:
        model = HeadPoseNet(input_size=input_size, backbone=backbone)
    
    return model

