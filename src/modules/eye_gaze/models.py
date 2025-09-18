"""
Deep learning models for eye gaze estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional


class ResNetGaze(nn.Module):
    """
    ResNet-based gaze estimation model.
    
    This model uses a ResNet backbone for feature extraction followed by
    regression layers for gaze vector prediction.
    """
    
    def __init__(
        self, 
        backbone: str = 'resnet18',
        input_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Initialize ResNet gaze model.
        
        Args:
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            input_size: Input image size (height, width)
            dropout: Dropout rate
            pretrained: Use ImageNet pretrained weights
        """
        super(ResNetGaze, self).__init__()
        
        self.input_size = input_size
        
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
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Gaze regression head
        self.gaze_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # 3D gaze vector
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in [self.gaze_head, self.confidence_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input eye images [batch_size, 3, height, width]
            
        Returns:
            Tuple of (gaze_vectors, confidence_scores)
        """
        # Feature extraction
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Gaze prediction
        gaze_vector = self.gaze_head(features)
        
        # Normalize to unit vector
        gaze_vector = F.normalize(gaze_vector, p=2, dim=1)
        
        # Confidence prediction
        confidence = self.confidence_head(features)
        
        return gaze_vector, confidence.squeeze()


class MobileNetGaze(nn.Module):
    """
    MobileNet-based lightweight gaze estimation model.
    
    This model uses MobileNet for efficient real-time gaze estimation
    with reduced computational requirements.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.3,
        pretrained: bool = True
    ):
        """
        Initialize MobileNet gaze model.
        
        Args:
            input_size: Input image size (height, width)
            dropout: Dropout rate
            pretrained: Use ImageNet pretrained weights
        """
        super(MobileNetGaze, self).__init__()
        
        self.input_size = input_size
        
        # Load MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        feature_dim = 1280
        
        # Gaze regression head
        self.gaze_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, 3)  # 3D gaze vector
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 32),
            nn.ReLU6(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in [self.gaze_head, self.confidence_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input eye images [batch_size, 3, height, width]
            
        Returns:
            Tuple of (gaze_vectors, confidence_scores)
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Gaze prediction
        gaze_vector = self.gaze_head(features)
        
        # Normalize to unit vector
        gaze_vector = F.normalize(gaze_vector, p=2, dim=1)
        
        # Confidence prediction
        confidence = self.confidence_head(features)
        
        return gaze_vector, confidence.squeeze()


class EfficientNetGaze(nn.Module):
    """
    EfficientNet-based gaze estimation model.
    
    Balances accuracy and efficiency using EfficientNet architecture.
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        input_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.4,
        pretrained: bool = True
    ):
        """
        Initialize EfficientNet gaze model.
        
        Args:
            model_name: EfficientNet variant
            input_size: Input image size
            dropout: Dropout rate
            pretrained: Use ImageNet pretrained weights
        """
        super(EfficientNetGaze, self).__init__()
        
        try:
            import timm
        except ImportError:
            raise ImportError("timm package required for EfficientNet models")
        
        self.input_size = input_size
        
        # Load EfficientNet backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        feature_dim = self.backbone.num_features
        
        # Gaze regression head
        self.gaze_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.Swish(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.Swish(),
            nn.Linear(128, 3)  # 3D gaze vector
        )
        
        # Confidence estimation head
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
        for m in [self.gaze_head, self.confidence_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input eye images [batch_size, 3, height, width]
            
        Returns:
            Tuple of (gaze_vectors, confidence_scores)
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Gaze prediction
        gaze_vector = self.gaze_head(features)
        
        # Normalize to unit vector
        gaze_vector = F.normalize(gaze_vector, p=2, dim=1)
        
        # Confidence prediction
        confidence = self.confidence_head(features)
        
        return gaze_vector, confidence.squeeze()


class MultiScaleGaze(nn.Module):
    """
    Multi-scale gaze estimation model that processes eye images at different scales.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        scales: Tuple[int, ...] = (224, 112, 56),
        dropout: float = 0.5
    ):
        """
        Initialize multi-scale gaze model.
        
        Args:
            backbone: Backbone architecture
            scales: Different input scales to process
            dropout: Dropout rate
        """
        super(MultiScaleGaze, self).__init__()
        
        self.scales = scales
        
        # Create multiple backbone networks for different scales
        self.backbones = nn.ModuleList()
        for scale in scales:
            if backbone == 'resnet18':
                model = models.resnet18(pretrained=True)
                feature_dim = 512
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            # Remove final layer
            model = nn.Sequential(*list(model.children())[:-1])
            self.backbones.append(model)
        
        # Fusion layer
        total_features = feature_dim * len(scales)
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Gaze head
        self.gaze_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-scale processing.
        
        Args:
            x: Input eye images [batch_size, 3, height, width]
            
        Returns:
            Tuple of (gaze_vectors, confidence_scores)
        """
        batch_size = x.size(0)
        features_list = []
        
        # Process at different scales
        for i, (backbone, scale) in enumerate(zip(self.backbones, self.scales)):
            # Resize input to current scale
            x_scaled = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
            
            # Extract features
            features = backbone(x_scaled)
            features = features.view(batch_size, -1)
            features_list.append(features)
        
        # Concatenate multi-scale features
        combined_features = torch.cat(features_list, dim=1)
        
        # Fusion
        fused_features = self.fusion(combined_features)
        
        # Predictions
        gaze_vector = self.gaze_head(fused_features)
        gaze_vector = F.normalize(gaze_vector, p=2, dim=1)
        
        confidence = self.confidence_head(fused_features)
        
        return gaze_vector, confidence.squeeze()
