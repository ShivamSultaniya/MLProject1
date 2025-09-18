"""
Engagement Recognition Models

Neural network models for engagement recognition from facial expressions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EngagementNet(nn.Module):
    """
    Simple CNN for engagement recognition from facial images.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224), 
                 num_classes: int = 4):
        """
        Initialize EngagementNet.
        
        Args:
            input_size: Input image size (height, width)
            num_classes: Number of engagement classes (e.g., high, medium, low, disengaged)
        """
        super(EngagementNet, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate feature size
        feature_size = 512 * 4 * 4
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        # Extract features
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        
        return output

