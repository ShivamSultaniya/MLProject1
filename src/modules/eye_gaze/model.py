"""
Gaze Estimation Neural Network Models

Implements various architectures for eye gaze estimation including:
- GazeNet: Custom CNN architecture
- MPIIGaze-style models
- GazeCapture-style models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class GazeNet(nn.Module):
    """
    Main gaze estimation network architecture.
    
    Supports multiple model types for different datasets and use cases.
    """
    
    def __init__(self, model_type: str = 'gazenet', input_size: Tuple[int, int] = (60, 36), 
                 num_classes: int = 2):
        """
        Initialize GazeNet model.
        
        Args:
            model_type: Type of model ('gazenet', 'mpiigaze', 'gazecapture')
            input_size: Input image size (width, height)
            num_classes: Number of output classes (2 for yaw/pitch)
        """
        super(GazeNet, self).__init__()
        
        self.model_type = model_type
        self.input_size = input_size
        self.num_classes = num_classes
        
        if model_type == 'gazenet':
            self._build_gazenet()
        elif model_type == 'mpiigaze':
            self._build_mpiigaze_model()
        elif model_type == 'gazecapture':
            self._build_gazecapture_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _build_gazenet(self):
        """Build custom GazeNet architecture."""
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
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
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate feature size after conv layers
        self.feature_size = 256 * 4 * 4
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes)
        )
        
        # Confidence estimation branch
        self.confidence_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _build_mpiigaze_model(self):
        """Build MPIIGaze-style model architecture."""
        # Based on the original MPIIGaze paper architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 3))
        )
        
        self.feature_size = 50 * 4 * 3
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, self.num_classes)
        )
        
        self.confidence_branch = nn.Sequential(
            nn.Linear(500, 1),
            nn.Sigmoid()
        )
    
    def _build_gazecapture_model(self):
        """Build GazeCapture-style model architecture."""
        # Inspired by the GazeCapture mobile architecture
        self.conv_layers = nn.Sequential(
            # Eye feature extraction
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        self.feature_size = 256 * 2 * 2
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes)
        )
        
        self.confidence_branch = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        # Extract features through conv layers
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        
        # Get intermediate representation for confidence
        if self.model_type == 'gazenet':
            intermediate = self.fc_layers[:-1](features)  # All layers except final
            gaze_output = self.fc_layers[-1](intermediate)
            confidence = self.confidence_branch(intermediate)
        else:
            # For other models, use simpler approach
            gaze_output = self.fc_layers(features)
            confidence = torch.ones(x.size(0), 1, device=x.device) * 0.8  # Default confidence
        
        if self.training:
            return gaze_output, confidence
        else:
            return gaze_output


class MultiModalGazeNet(nn.Module):
    """
    Multi-modal gaze estimation network that can incorporate additional inputs
    like head pose, face landmarks, etc.
    """
    
    def __init__(self, eye_input_size: Tuple[int, int] = (60, 36), 
                 head_pose_dim: int = 3, face_landmark_dim: int = 136):
        super(MultiModalGazeNet, self).__init__()
        
        # Eye region processing
        self.eye_net = GazeNet(model_type='gazenet', input_size=eye_input_size)
        
        # Head pose processing
        self.head_pose_net = nn.Sequential(
            nn.Linear(head_pose_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        
        # Face landmark processing
        self.landmark_net = nn.Sequential(
            nn.Linear(face_landmark_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        
        # Fusion layer
        fusion_input_size = 2 + 32 + 32  # gaze (2) + head_pose (32) + landmarks (32)
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # Final gaze output
        )
    
    def forward(self, eye_images, head_pose=None, face_landmarks=None):
        """
        Forward pass with multiple modalities.
        
        Args:
            eye_images: Eye region images [B, C, H, W]
            head_pose: Head pose parameters [B, 3] (optional)
            face_landmarks: Face landmark coordinates [B, 136] (optional)
        """
        # Process eye images
        eye_gaze = self.eye_net(eye_images)
        
        features = [eye_gaze]
        
        # Process head pose if available
        if head_pose is not None:
            head_features = self.head_pose_net(head_pose)
            features.append(head_features)
        
        # Process face landmarks if available
        if face_landmarks is not None:
            landmark_features = self.landmark_net(face_landmarks)
            features.append(landmark_features)
        
        # Fuse all features
        if len(features) > 1:
            fused_features = torch.cat(features, dim=1)
            final_gaze = self.fusion_net(fused_features)
        else:
            final_gaze = eye_gaze
        
        return final_gaze


class GazeLoss(nn.Module):
    """Custom loss function for gaze estimation."""
    
    def __init__(self, angular_weight: float = 1.0, confidence_weight: float = 0.1):
        super(GazeLoss, self).__init__()
        self.angular_weight = angular_weight
        self.confidence_weight = confidence_weight
    
    def forward(self, predicted_gaze, target_gaze, confidence=None):
        """
        Compute gaze estimation loss.
        
        Args:
            predicted_gaze: Predicted gaze direction [B, 2]
            target_gaze: Ground truth gaze direction [B, 2]
            confidence: Prediction confidence [B, 1] (optional)
        """
        # Angular loss (mean squared error on angles)
        angular_loss = F.mse_loss(predicted_gaze, target_gaze)
        
        total_loss = self.angular_weight * angular_loss
        
        # Confidence regularization if available
        if confidence is not None:
            # Encourage high confidence for accurate predictions
            accuracy = torch.exp(-torch.norm(predicted_gaze - target_gaze, dim=1, keepdim=True))
            confidence_loss = F.mse_loss(confidence, accuracy)
            total_loss += self.confidence_weight * confidence_loss
        
        return total_loss


def create_model(model_type: str = 'gazenet', **kwargs) -> nn.Module:
    """
    Factory function to create gaze estimation models.
    
    Args:
        model_type: Type of model to create
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Initialized model
    """
    if model_type == 'multimodal':
        return MultiModalGazeNet(**kwargs)
    else:
        return GazeNet(model_type=model_type, **kwargs)


def load_pretrained_model(model_path: str, model_type: str = 'gazenet', device: str = 'cpu') -> nn.Module:
    """
    Load a pretrained gaze estimation model.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = create_model(model_type)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
        print("Using randomly initialized model")
    
    model.to(device)
    return model


