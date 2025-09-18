"""
Deep learning models for blink and drowsiness detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class BlinkCNN(nn.Module):
    """
    Convolutional Neural Network for blink detection from eye images.
    
    This model classifies eye images as 'open' or 'closed' to detect blinks.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (32, 64),  # (height, width)
        dropout: float = 0.5
    ):
        """
        Initialize Blink CNN.
        
        Args:
            input_size: Input image size (height, width)
            dropout: Dropout rate
        """
        super(BlinkCNN, self).__init__()
        
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)
            dummy_output = self.pool3(self.bn3(self.conv3(
                self.pool2(self.bn2(self.conv2(
                    self.pool1(self.bn1(self.conv1(dummy_input)))
                )))
            )))
            self.flattened_size = dummy_output.numel()
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # open/closed
        
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
        """
        Forward pass.
        
        Args:
            x: Input eye images [batch_size, 1, height, width]
            
        Returns:
            Classification logits [batch_size, 2]
        """
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)
        
        return x


class DrowsinessLSTM(nn.Module):
    """
    LSTM-based model for drowsiness detection from temporal sequences.
    
    This model analyzes temporal patterns in eye and head movement features
    to detect different levels of drowsiness.
    """
    
    def __init__(
        self,
        input_size: int = 6,  # Number of input features
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,  # 5 drowsiness levels
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize Drowsiness LSTM.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of drowsiness classes
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(DrowsinessLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequences [batch_size, sequence_length, input_size]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        batch_size, seq_len, _ = x.size()
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


class EARCalculator(nn.Module):
    """
    Eye Aspect Ratio (EAR) calculator using facial landmarks.
    
    EAR is a simple but effective measure for blink detection.
    """
    
    def __init__(self):
        super(EARCalculator, self).__init__()
    
    def forward(self, eye_landmarks: torch.Tensor) -> torch.Tensor:
        """
        Calculate EAR from eye landmarks.
        
        Args:
            eye_landmarks: Eye landmarks [batch_size, 6, 2] for single eye
                          or [batch_size, 12, 2] for both eyes
            
        Returns:
            EAR values [batch_size] or [batch_size, 2] for both eyes
        """
        if eye_landmarks.size(1) == 6:
            # Single eye
            return self._calculate_single_ear(eye_landmarks)
        elif eye_landmarks.size(1) == 12:
            # Both eyes
            left_eye = eye_landmarks[:, :6, :]
            right_eye = eye_landmarks[:, 6:, :]
            
            left_ear = self._calculate_single_ear(left_eye)
            right_ear = self._calculate_single_ear(right_eye)
            
            return torch.stack([left_ear, right_ear], dim=1)
        else:
            raise ValueError(f"Expected 6 or 12 landmarks, got {eye_landmarks.size(1)}")
    
    def _calculate_single_ear(self, eye_landmarks: torch.Tensor) -> torch.Tensor:
        """Calculate EAR for a single eye."""
        # Eye landmarks order: [p1, p2, p3, p4, p5, p6]
        # where p1 and p4 are horizontal corners
        # p2, p3, p5, p6 are vertical points
        
        # Vertical distances
        vertical_1 = torch.norm(eye_landmarks[:, 1] - eye_landmarks[:, 5], dim=1)
        vertical_2 = torch.norm(eye_landmarks[:, 2] - eye_landmarks[:, 4], dim=1)
        
        # Horizontal distance
        horizontal = torch.norm(eye_landmarks[:, 0] - eye_landmarks[:, 3], dim=1)
        
        # EAR calculation
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-8)  # Add small epsilon
        
        return ear


class BlinkSequenceClassifier(nn.Module):
    """
    Sequence classifier for detecting blink patterns and anomalies.
    """
    
    def __init__(
        self,
        input_size: int = 1,  # EAR values
        hidden_size: int = 32,
        num_layers: int = 1,
        sequence_length: int = 30,
        num_classes: int = 3  # normal_blink, long_blink, no_blink
    ):
        """
        Initialize Blink Sequence Classifier.
        
        Args:
            input_size: Number of input features (typically just EAR)
            hidden_size: GRU hidden size
            num_layers: Number of GRU layers
            sequence_length: Expected sequence length
            num_classes: Number of blink pattern classes
        """
        super(BlinkSequenceClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # GRU layer (lighter than LSTM for this task)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequences [batch_size, sequence_length, input_size]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]
        
        # Classification
        output = self.classifier(last_hidden)
        
        return output


class MultiTaskBlinkModel(nn.Module):
    """
    Multi-task model that simultaneously predicts:
    1. Blink detection (binary classification)
    2. Blink duration estimation (regression)
    3. Eye state confidence (regression)
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (32, 64),
        dropout: float = 0.4
    ):
        """
        Initialize Multi-task Blink Model.
        
        Args:
            input_size: Input image size (height, width)
            dropout: Dropout rate
        """
        super(MultiTaskBlinkModel, self).__init__()
        
        # Shared convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 8))
        )
        
        # Calculate feature size
        feature_size = 128 * 4 * 8  # 4096
        
        # Shared feature layers
        self.shared_features = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        # Blink detection (binary classification)
        self.blink_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
        # Blink duration regression
        self.duration_regressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure positive duration
        )
        
        # Eye state confidence
        self.confidence_regressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input eye images [batch_size, 1, height, width]
            
        Returns:
            Tuple of (blink_logits, duration_prediction, confidence_score)
        """
        # Backbone feature extraction
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Shared features
        shared = self.shared_features(features)
        
        # Task-specific predictions
        blink_logits = self.blink_classifier(shared)
        duration_pred = self.duration_regressor(shared)
        confidence_score = self.confidence_regressor(shared)
        
        return blink_logits, duration_pred.squeeze(), confidence_score.squeeze()


class AdaptiveBlinkThreshold(nn.Module):
    """
    Adaptive threshold model that learns optimal EAR thresholds for different users.
    """
    
    def __init__(
        self,
        input_features: int = 10,  # Recent EAR statistics
        hidden_size: int = 32
    ):
        """
        Initialize Adaptive Threshold Model.
        
        Args:
            input_features: Number of input statistical features
            hidden_size: Hidden layer size
        """
        super(AdaptiveBlinkThreshold, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output threshold between 0 and 1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict optimal EAR threshold.
        
        Args:
            x: Statistical features [batch_size, input_features]
            
        Returns:
            Predicted thresholds [batch_size]
        """
        threshold = self.network(x)
        
        # Scale to reasonable EAR threshold range (0.15 to 0.35)
        threshold = 0.15 + threshold * 0.2
        
        return threshold.squeeze()
    
    def extract_features(self, ear_history: list) -> torch.Tensor:
        """
        Extract statistical features from EAR history.
        
        Args:
            ear_history: List of recent EAR values
            
        Returns:
            Feature tensor
        """
        if len(ear_history) < 10:
            return torch.zeros(10)
        
        ear_array = np.array(ear_history[-100:])  # Use last 100 values
        
        features = [
            np.mean(ear_array),           # Mean EAR
            np.std(ear_array),            # Standard deviation
            np.median(ear_array),         # Median EAR
            np.percentile(ear_array, 25), # 25th percentile
            np.percentile(ear_array, 75), # 75th percentile
            np.min(ear_array),            # Minimum EAR
            np.max(ear_array),            # Maximum EAR
            len(ear_array),               # History length
            np.var(ear_array),            # Variance
            np.sum(ear_array < 0.25) / len(ear_array)  # Fraction below typical threshold
        ]
        
        return torch.tensor(features, dtype=torch.float32)
