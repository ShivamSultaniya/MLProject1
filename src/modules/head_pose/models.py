"""
Deep learning models for head pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional
import math


class ResNetPose(nn.Module):
    """
    ResNet-based head pose estimation model.
    
    Predicts Euler angles (pitch, yaw, roll) from face images.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        input_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.5,
        pretrained: bool = True,
        output_confidence: bool = True
    ):
        """
        Initialize ResNet pose model.
        
        Args:
            backbone: ResNet variant
            input_size: Input image size
            dropout: Dropout rate
            pretrained: Use ImageNet pretrained weights
            output_confidence: Whether to output confidence scores
        """
        super(ResNetPose, self).__init__()
        
        self.input_size = input_size
        self.output_confidence = output_confidence
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Pose regression head
        self.pose_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)  # pitch, yaw, roll
        )
        
        # Confidence estimation head
        if self.output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in [self.pose_head] + ([self.confidence_head] if self.output_confidence else []):
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            Pose predictions [batch_size, 3] or tuple with confidence
        """
        # Feature extraction
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Pose prediction (in radians)
        pose_angles = self.pose_head(features)
        
        if self.output_confidence:
            confidence = self.confidence_head(features)
            return pose_angles, confidence.squeeze()
        else:
            return pose_angles


class EfficientNetPose(nn.Module):
    """
    EfficientNet-based pose estimation model for better efficiency.
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        input_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.4,
        pretrained: bool = True,
        output_confidence: bool = True
    ):
        """
        Initialize EfficientNet pose model.
        
        Args:
            model_name: EfficientNet variant
            input_size: Input image size
            dropout: Dropout rate
            pretrained: Use ImageNet pretrained weights
            output_confidence: Whether to output confidence scores
        """
        super(EfficientNetPose, self).__init__()
        
        try:
            import timm
        except ImportError:
            raise ImportError("timm package required for EfficientNet models")
        
        self.input_size = input_size
        self.output_confidence = output_confidence
        
        # Load EfficientNet backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        feature_dim = self.backbone.num_features
        
        # Pose regression head
        self.pose_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.Swish(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Swish(),
            nn.Linear(128, 3)  # pitch, yaw, roll
        )
        
        # Confidence estimation head
        if self.output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 64),
                nn.Swish(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in [self.pose_head] + ([self.confidence_head] if self.output_confidence else []):
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Feature extraction
        features = self.backbone(x)
        
        # Pose prediction
        pose_angles = self.pose_head(features)
        
        if self.output_confidence:
            confidence = self.confidence_head(features)
            return pose_angles, confidence.squeeze()
        else:
            return pose_angles


class PoseNet(nn.Module):
    """
    Lightweight pose estimation network inspired by PoseNet architecture.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.3,
        output_confidence: bool = True
    ):
        """
        Initialize PoseNet.
        
        Args:
            input_size: Input image size
            dropout: Dropout rate
            output_confidence: Whether to output confidence scores
        """
        super(PoseNet, self).__init__()
        
        self.input_size = input_size
        self.output_confidence = output_confidence
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        
        # Pose regression head
        self.pose_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)  # pitch, yaw, roll
        )
        
        # Confidence estimation head
        if self.output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Pose prediction
        pose_angles = self.pose_head(self.dropout(x))
        
        if self.output_confidence:
            confidence = self.confidence_head(x)
            return pose_angles, confidence.squeeze()
        else:
            return pose_angles


class MultiTaskPoseModel(nn.Module):
    """
    Multi-task model that predicts both pose and face attributes.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        input_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.5,
        num_attributes: int = 8  # age, gender, glasses, etc.
    ):
        """
        Initialize multi-task pose model.
        
        Args:
            backbone: Backbone architecture
            input_size: Input image size
            dropout: Dropout rate
            num_attributes: Number of face attributes to predict
        """
        super(MultiTaskPoseModel, self).__init__()
        
        self.input_size = input_size
        self.num_attributes = num_attributes
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Shared feature layers
        self.shared_features = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True)
        )
        
        # Pose estimation head
        self.pose_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)  # pitch, yaw, roll
        )
        
        # Attribute prediction head
        self.attribute_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_attributes)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            Tuple of (pose_angles, attributes, confidence)
        """
        # Feature extraction
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Shared features
        shared = self.shared_features(features)
        
        # Task-specific predictions
        pose_angles = self.pose_head(shared)
        attributes = self.attribute_head(shared)
        confidence = self.confidence_head(shared)
        
        return pose_angles, attributes, confidence.squeeze()


class TemporalPoseModel(nn.Module):
    """
    Temporal model that uses sequence of frames for more stable pose estimation.
    """
    
    def __init__(
        self,
        cnn_backbone: str = 'resnet18',
        sequence_length: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize temporal pose model.
        
        Args:
            cnn_backbone: CNN backbone for feature extraction
            sequence_length: Number of frames in sequence
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(TemporalPoseModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # CNN backbone for feature extraction
        if cnn_backbone == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {cnn_backbone}")
        
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        
        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_size)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Pose regression head
        self.pose_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # pitch, yaw, roll
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequences [batch_size, sequence_length, 3, height, width]
            
        Returns:
            Tuple of (pose_angles, confidence)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape for CNN processing
        x = x.view(batch_size * seq_len, c, h, w)
        
        # CNN feature extraction
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size * seq_len, -1)
        
        # Project features
        projected_features = self.feature_proj(cnn_features)
        
        # Reshape for LSTM
        lstm_input = projected_features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Predictions
        pose_angles = self.pose_head(last_output)
        confidence = self.confidence_head(last_output)
        
        return pose_angles, confidence.squeeze()


class PoseLoss(nn.Module):
    """
    Custom loss function for pose estimation combining multiple terms.
    """
    
    def __init__(
        self,
        angle_weight: float = 1.0,
        confidence_weight: float = 0.1,
        smoothness_weight: float = 0.05
    ):
        """
        Initialize pose loss.
        
        Args:
            angle_weight: Weight for angle prediction loss
            confidence_weight: Weight for confidence loss
            smoothness_weight: Weight for temporal smoothness loss
        """
        super(PoseLoss, self).__init__()
        
        self.angle_weight = angle_weight
        self.confidence_weight = confidence_weight
        self.smoothness_weight = smoothness_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        pred_angles: torch.Tensor,
        target_angles: torch.Tensor,
        pred_confidence: Optional[torch.Tensor] = None,
        target_confidence: Optional[torch.Tensor] = None,
        prev_angles: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pose loss.
        
        Args:
            pred_angles: Predicted angles [batch_size, 3]
            target_angles: Target angles [batch_size, 3]
            pred_confidence: Predicted confidence scores
            target_confidence: Target confidence scores
            prev_angles: Previous frame angles for smoothness
            
        Returns:
            Total loss value
        """
        # Angle prediction loss
        angle_loss = self.mse_loss(pred_angles, target_angles)
        total_loss = self.angle_weight * angle_loss
        
        # Confidence loss
        if pred_confidence is not None and target_confidence is not None:
            conf_loss = self.bce_loss(pred_confidence, target_confidence)
            total_loss += self.confidence_weight * conf_loss
        
        # Temporal smoothness loss
        if prev_angles is not None:
            smoothness_loss = self.mse_loss(pred_angles, prev_angles)
            total_loss += self.smoothness_weight * smoothness_loss
        
        return total_loss
