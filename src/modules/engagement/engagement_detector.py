"""
Real-time engagement detection system.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import logging
from collections import deque
import time
from enum import Enum

from .models import EngagementCNN, EngagementTransformer, MultiModalEngagement
from .utils import extract_engagement_features, analyze_attention_patterns
from ..utils.face_detector import FaceDetector

logger = logging.getLogger(__name__)


class EngagementLevel(Enum):
    """Engagement levels based on DAiSEE dataset."""
    VERY_LOW = 0
    LOW = 1
    HIGH = 2
    VERY_HIGH = 3


class EngagementDetector:
    """
    Real-time engagement detection system that analyzes:
    1. Facial expressions and micro-expressions
    2. Eye contact and gaze patterns
    3. Head pose and movement patterns
    4. Temporal behavioral patterns
    5. Multi-modal fusion for robust engagement estimation
    """
    
    def __init__(
        self,
        model_type: str = 'cnn',  # 'cnn', 'transformer', 'multimodal'
        model_path: Optional[str] = None,
        device: str = 'auto',
        input_size: Tuple[int, int] = (224, 224),
        sequence_length: int = 16,
        confidence_threshold: float = 0.6,
        temporal_window: int = 30
    ):
        """
        Initialize engagement detector.
        
        Args:
            model_type: Type of engagement model
            model_path: Path to pre-trained model
            device: Device for inference
            input_size: Input image size
            sequence_length: Length of temporal sequences
            confidence_threshold: Minimum confidence threshold
            temporal_window: Window for temporal analysis
        """
        self.model_type = model_type
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Initialize engagement model
        self._load_model(model_path)
        
        # History buffers
        self.frame_buffer = deque(maxlen=sequence_length)
        self.feature_history = deque(maxlen=temporal_window)
        self.engagement_history = deque(maxlen=temporal_window)
        self.attention_history = deque(maxlen=temporal_window)
        
        # Engagement tracking
        self.current_engagement = EngagementLevel.HIGH
        self.engagement_confidence = 0.0
        self.attention_score = 0.0
        self.distraction_events = []
        
        # Session statistics
        self.session_start_time = time.time()
        self.total_frames_processed = 0
        self.engagement_distribution = {level: 0 for level in EngagementLevel}
        
        logger.info(f"EngagementDetector initialized with {model_type} model on {self.device}")
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load the engagement detection model."""
        if self.model_type == 'cnn':
            self.model = EngagementCNN(
                input_size=self.input_size,
                num_classes=4  # 4 engagement levels
            )
        elif self.model_type == 'transformer':
            self.model = EngagementTransformer(
                input_size=self.input_size,
                sequence_length=self.sequence_length,
                num_classes=4
            )
        elif self.model_type == 'multimodal':
            self.model = MultiModalEngagement(
                visual_input_size=self.input_size,
                sequence_length=self.sequence_length,
                num_classes=4
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.to(self.device)
        
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        else:
            logger.warning("No pre-trained model loaded.")
        
        self.model.eval()
    
    def detect_engagement(
        self,
        frame: np.ndarray,
        gaze_info: Optional[Dict[str, Any]] = None,
        pose_info: Optional[Dict[str, Any]] = None,
        blink_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect engagement level from a single frame with optional multi-modal inputs.
        
        Args:
            frame: Input video frame
            gaze_info: Optional gaze estimation results
            pose_info: Optional head pose results
            blink_info: Optional blink/drowsiness results
            
        Returns:
            Dictionary containing engagement analysis results
        """
        current_time = time.time()
        self.total_frames_processed += 1
        
        results = {
            'engagement_level': EngagementLevel.HIGH,
            'engagement_confidence': 0.0,
            'attention_score': 0.0,
            'distraction_detected': False,
            'facial_features': None,
            'behavioral_features': None,
            'temporal_patterns': None,
            'face_bbox': None,
            'processing_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Detect face
            faces = self.face_detector.detect_faces(frame)
            if not faces:
                return results
            
            # Use largest face
            face_bbox = max(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            results['face_bbox'] = face_bbox
            
            # Extract face region
            face_region = self._extract_face_region(frame, face_bbox)
            if face_region is None:
                return results
            
            # Add to frame buffer
            self.frame_buffer.append(face_region)
            
            # Extract visual features
            facial_features = self._extract_facial_features(face_region)
            results['facial_features'] = facial_features
            
            # Extract behavioral features from multi-modal inputs
            behavioral_features = self._extract_behavioral_features(
                gaze_info, pose_info, blink_info, current_time
            )
            results['behavioral_features'] = behavioral_features
            
            # Combine features
            combined_features = self._combine_features(facial_features, behavioral_features)
            
            # Add to feature history
            self.feature_history.append(combined_features)
            
            # Engagement prediction
            if self.model_type == 'cnn':
                engagement_pred = self._predict_engagement_cnn(face_region)
            elif self.model_type == 'transformer':
                if len(self.frame_buffer) >= self.sequence_length:
                    engagement_pred = self._predict_engagement_transformer()
                else:
                    engagement_pred = {'level': EngagementLevel.HIGH, 'confidence': 0.5}
            elif self.model_type == 'multimodal':
                engagement_pred = self._predict_engagement_multimodal(combined_features)
            
            # Update results
            engagement_level = engagement_pred['level']
            engagement_confidence = engagement_pred['confidence']
            
            results['engagement_level'] = engagement_level
            results['engagement_confidence'] = engagement_confidence
            
            # Calculate attention score
            attention_score = self._calculate_attention_score(
                engagement_level, gaze_info, pose_info, blink_info
            )
            results['attention_score'] = attention_score
            
            # Detect distraction events
            distraction_detected = self._detect_distraction(
                engagement_level, attention_score, behavioral_features
            )
            results['distraction_detected'] = distraction_detected
            
            # Temporal pattern analysis
            if len(self.feature_history) >= 10:
                temporal_patterns = analyze_attention_patterns(
                    list(self.engagement_history)[-10:],
                    list(self.attention_history)[-10:]
                )
                results['temporal_patterns'] = temporal_patterns
            
            # Update tracking
            self.current_engagement = engagement_level
            self.engagement_confidence = engagement_confidence
            self.attention_score = attention_score
            
            # Update histories
            self.engagement_history.append(engagement_level.value)
            self.attention_history.append(attention_score)
            
            # Update statistics
            self.engagement_distribution[engagement_level] += 1
            
            # Record distraction events
            if distraction_detected:
                self.distraction_events.append({
                    'timestamp': current_time,
                    'type': 'low_engagement',
                    'engagement_level': engagement_level.name,
                    'attention_score': attention_score
                })
            
            results['processing_time'] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error in engagement detection: {e}")
            results['processing_time'] = time.time() - start_time
        
        return results
    
    def _extract_face_region(
        self, 
        frame: np.ndarray, 
        face_bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Extract and preprocess face region."""
        try:
            x1, y1, x2, y2 = face_bbox
            
            # Add padding
            padding = 0.15
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * padding), int(h * padding)
            
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(frame.shape[1], x2 + pad_w)
            y2_pad = min(frame.shape[0], y2 + pad_h)
            
            face_region = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_region.size == 0:
                return None
            
            # Resize to model input size
            face_resized = cv2.resize(face_region, self.input_size)
            
            return face_resized
            
        except Exception as e:
            logger.error(f"Error extracting face region: {e}")
            return None
    
    def _extract_facial_features(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Extract facial features for engagement analysis."""
        features = {
            'face_available': True,
            'face_quality': 0.0,
            'expression_features': None,
            'eye_features': None,
            'mouth_features': None
        }
        
        try:
            # Basic face quality assessment
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features['face_quality'] = min(1.0, laplacian_var / 100.0)
            
            # Expression analysis (simplified)
            # In practice, you'd use more sophisticated facial expression analysis
            features['expression_features'] = {
                'brightness': np.mean(gray) / 255.0,
                'contrast': np.std(gray) / 255.0,
                'texture_energy': laplacian_var
            }
            
            # Eye region analysis
            h, w = face_image.shape[:2]
            eye_region = face_image[int(h*0.25):int(h*0.6), int(w*0.2):int(w*0.8)]
            
            if eye_region.size > 0:
                eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                features['eye_features'] = {
                    'eye_openness': np.mean(eye_gray) / 255.0,
                    'eye_activity': np.std(eye_gray) / 255.0
                }
            
            # Mouth region analysis
            mouth_region = face_image[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
            
            if mouth_region.size > 0:
                mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
                features['mouth_features'] = {
                    'mouth_activity': np.std(mouth_gray) / 255.0
                }
            
        except Exception as e:
            logger.error(f"Error extracting facial features: {e}")
            features['face_available'] = False
        
        return features
    
    def _extract_behavioral_features(
        self,
        gaze_info: Optional[Dict[str, Any]],
        pose_info: Optional[Dict[str, Any]], 
        blink_info: Optional[Dict[str, Any]],
        timestamp: float
    ) -> Dict[str, Any]:
        """Extract behavioral features from multi-modal inputs."""
        features = {
            'gaze_stability': 0.0,
            'attention_direction': 'unknown',
            'head_movement': 0.0,
            'blink_pattern': 'normal',
            'engagement_indicators': {}
        }
        
        try:
            # Gaze-based features
            if gaze_info and gaze_info.get('valid', False):
                gaze_angles = gaze_info.get('gaze_angles', (0, 0))
                features['attention_direction'] = self._classify_gaze_direction(gaze_angles)
                features['gaze_stability'] = gaze_info.get('confidence', 0.0)
            
            # Head pose features
            if pose_info and pose_info.get('valid', False):
                pitch, yaw, roll = pose_info.get('pitch', 0), pose_info.get('yaw', 0), pose_info.get('roll', 0)
                
                # Calculate head movement magnitude
                head_movement = np.sqrt(pitch**2 + yaw**2 + roll**2)
                features['head_movement'] = min(1.0, head_movement / 45.0)  # Normalize by 45 degrees
                
                # Attention indicators from head pose
                if abs(yaw) < 15 and abs(pitch) < 10:
                    features['engagement_indicators']['frontal_pose'] = True
                else:
                    features['engagement_indicators']['frontal_pose'] = False
            
            # Blink-based features
            if blink_info:
                blink_rate = blink_info.get('blink_rate', 15)
                
                if blink_rate < 5:
                    features['blink_pattern'] = 'too_low'
                elif blink_rate > 25:
                    features['blink_pattern'] = 'too_high'
                else:
                    features['blink_pattern'] = 'normal'
                
                # Drowsiness indicators
                drowsiness_level = blink_info.get('drowsiness_level', 0)
                if hasattr(drowsiness_level, 'value'):
                    drowsiness_level = drowsiness_level.value
                
                features['engagement_indicators']['alert'] = drowsiness_level < 2
            
        except Exception as e:
            logger.error(f"Error extracting behavioral features: {e}")
        
        return features
    
    def _classify_gaze_direction(self, gaze_angles: Tuple[float, float]) -> str:
        """Classify gaze direction from gaze angles."""
        pitch, yaw = gaze_angles
        
        if abs(yaw) < 10 and abs(pitch) < 8:
            return 'center'
        elif yaw < -10:
            return 'left'
        elif yaw > 10:
            return 'right'
        elif pitch < -8:
            return 'up'
        elif pitch > 8:
            return 'down'
        else:
            return 'center'
    
    def _combine_features(
        self,
        facial_features: Dict[str, Any],
        behavioral_features: Dict[str, Any]
    ) -> np.ndarray:
        """Combine facial and behavioral features into a single vector."""
        try:
            feature_vector = []
            
            # Facial features
            if facial_features.get('face_available', False):
                feature_vector.append(facial_features['face_quality'])
                
                expr_features = facial_features.get('expression_features', {})
                feature_vector.extend([
                    expr_features.get('brightness', 0.0),
                    expr_features.get('contrast', 0.0),
                    expr_features.get('texture_energy', 0.0) / 1000.0  # Normalize
                ])
                
                eye_features = facial_features.get('eye_features', {})
                feature_vector.extend([
                    eye_features.get('eye_openness', 0.0),
                    eye_features.get('eye_activity', 0.0)
                ])
                
                mouth_features = facial_features.get('mouth_features', {})
                feature_vector.append(mouth_features.get('mouth_activity', 0.0))
            else:
                feature_vector.extend([0.0] * 7)  # Placeholder values
            
            # Behavioral features
            feature_vector.extend([
                behavioral_features.get('gaze_stability', 0.0),
                behavioral_features.get('head_movement', 0.0),
                1.0 if behavioral_features.get('attention_direction') == 'center' else 0.0,
                1.0 if behavioral_features.get('blink_pattern') == 'normal' else 0.0
            ])
            
            # Engagement indicators
            indicators = behavioral_features.get('engagement_indicators', {})
            feature_vector.extend([
                1.0 if indicators.get('frontal_pose', False) else 0.0,
                1.0 if indicators.get('alert', True) else 0.0
            ])
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error combining features: {e}")
            return np.zeros(13, dtype=np.float32)  # Default feature vector
    
    def _predict_engagement_cnn(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Predict engagement using CNN model."""
        try:
            # Preprocess image
            face_tensor = self._preprocess_face_image(face_image)
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            engagement_level = EngagementLevel(predicted_class)
            
            return {
                'level': engagement_level,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().flatten()
            }
            
        except Exception as e:
            logger.error(f"Error in CNN prediction: {e}")
            return {
                'level': EngagementLevel.HIGH,
                'confidence': 0.5,
                'probabilities': np.array([0.25, 0.25, 0.25, 0.25])
            }
    
    def _predict_engagement_transformer(self) -> Dict[str, Any]:
        """Predict engagement using Transformer model."""
        try:
            # Prepare sequence
            sequence = list(self.frame_buffer)
            if len(sequence) < self.sequence_length:
                return {'level': EngagementLevel.HIGH, 'confidence': 0.5}
            
            # Preprocess sequence
            sequence_tensor = torch.stack([
                self._preprocess_face_image(frame) for frame in sequence
            ]).unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            engagement_level = EngagementLevel(predicted_class)
            
            return {
                'level': engagement_level,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().flatten()
            }
            
        except Exception as e:
            logger.error(f"Error in Transformer prediction: {e}")
            return {'level': EngagementLevel.HIGH, 'confidence': 0.5}
    
    def _predict_engagement_multimodal(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict engagement using multi-modal model."""
        try:
            # Get current frame
            if not self.frame_buffer:
                return {'level': EngagementLevel.HIGH, 'confidence': 0.5}
            
            current_frame = self.frame_buffer[-1]
            face_tensor = self._preprocess_face_image(current_frame).unsqueeze(0).to(self.device)
            
            # Prepare features
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(face_tensor, features_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            engagement_level = EngagementLevel(predicted_class)
            
            return {
                'level': engagement_level,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy().flatten()
            }
            
        except Exception as e:
            logger.error(f"Error in multi-modal prediction: {e}")
            return {'level': EngagementLevel.HIGH, 'confidence': 0.5}
    
    def _preprocess_face_image(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess face image for model input."""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        face_normalized = (face_normalized - mean) / std
        
        # Convert to tensor
        face_tensor = torch.tensor(face_normalized, dtype=torch.float32)
        face_tensor = face_tensor.permute(2, 0, 1)  # HWC to CHW
        
        return face_tensor
    
    def _calculate_attention_score(
        self,
        engagement_level: EngagementLevel,
        gaze_info: Optional[Dict[str, Any]],
        pose_info: Optional[Dict[str, Any]],
        blink_info: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall attention score from multiple modalities."""
        score = 0.0
        weight_sum = 0.0
        
        # Engagement level contribution (40% weight)
        engagement_score = engagement_level.value / 3.0  # Normalize to [0, 1]
        score += engagement_score * 0.4
        weight_sum += 0.4
        
        # Gaze contribution (25% weight)
        if gaze_info and gaze_info.get('valid', False):
            gaze_confidence = gaze_info.get('confidence', 0.0)
            score += gaze_confidence * 0.25
            weight_sum += 0.25
        
        # Head pose contribution (20% weight)
        if pose_info and pose_info.get('valid', False):
            pose_confidence = pose_info.get('confidence', 0.0)
            pitch, yaw = pose_info.get('pitch', 0), pose_info.get('yaw', 0)
            
            # Penalty for looking away
            pose_penalty = 1.0 - (abs(yaw) + abs(pitch)) / 60.0  # Normalize by 60 degrees
            pose_score = pose_confidence * max(0.0, pose_penalty)
            
            score += pose_score * 0.20
            weight_sum += 0.20
        
        # Blink/alertness contribution (15% weight)
        if blink_info:
            alertness = 1.0  # Default to alert
            
            if 'drowsiness_level' in blink_info:
                drowsiness_level = blink_info['drowsiness_level']
                if hasattr(drowsiness_level, 'value'):
                    drowsiness_level = drowsiness_level.value
                alertness = max(0.0, 1.0 - drowsiness_level / 4.0)
            
            score += alertness * 0.15
            weight_sum += 0.15
        
        # Normalize by actual weights used
        if weight_sum > 0:
            score = score / weight_sum
        else:
            score = 0.5  # Default neutral score
        
        return max(0.0, min(1.0, score))
    
    def _detect_distraction(
        self,
        engagement_level: EngagementLevel,
        attention_score: float,
        behavioral_features: Dict[str, Any]
    ) -> bool:
        """Detect distraction events based on multiple indicators."""
        distraction_indicators = []
        
        # Low engagement level
        if engagement_level in [EngagementLevel.VERY_LOW, EngagementLevel.LOW]:
            distraction_indicators.append('low_engagement')
        
        # Low attention score
        if attention_score < 0.4:
            distraction_indicators.append('low_attention')
        
        # Looking away
        if behavioral_features.get('attention_direction') not in ['center', 'unknown']:
            distraction_indicators.append('looking_away')
        
        # Excessive head movement
        if behavioral_features.get('head_movement', 0.0) > 0.7:
            distraction_indicators.append('excessive_movement')
        
        # Abnormal blink patterns
        if behavioral_features.get('blink_pattern') != 'normal':
            distraction_indicators.append('abnormal_blinks')
        
        # Consider distracted if multiple indicators present
        return len(distraction_indicators) >= 2
    
    def get_engagement_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engagement statistics."""
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        # Calculate engagement distribution percentages
        total_frames = sum(self.engagement_distribution.values())
        engagement_percentages = {}
        
        for level, count in self.engagement_distribution.items():
            percentage = (count / total_frames * 100) if total_frames > 0 else 0
            engagement_percentages[level.name] = percentage
        
        # Calculate average engagement and attention
        recent_engagement = list(self.engagement_history)[-30:] if self.engagement_history else [2]
        recent_attention = list(self.attention_history)[-30:] if self.attention_history else [0.5]
        
        avg_engagement = np.mean(recent_engagement)
        avg_attention = np.mean(recent_attention)
        
        # Engagement trend
        if len(recent_engagement) >= 10:
            trend = np.polyfit(range(len(recent_engagement)), recent_engagement, 1)[0]
        else:
            trend = 0.0
        
        return {
            'current_engagement': self.current_engagement.name,
            'current_confidence': self.engagement_confidence,
            'current_attention_score': self.attention_score,
            'session_duration': session_duration,
            'total_frames_processed': self.total_frames_processed,
            'engagement_distribution': engagement_percentages,
            'average_engagement': avg_engagement,
            'average_attention': avg_attention,
            'engagement_trend': trend,
            'distraction_events_count': len(self.distraction_events),
            'processing_fps': self.total_frames_processed / session_duration if session_duration > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset all engagement statistics."""
        self.frame_buffer.clear()
        self.feature_history.clear()
        self.engagement_history.clear()
        self.attention_history.clear()
        
        self.current_engagement = EngagementLevel.HIGH
        self.engagement_confidence = 0.0
        self.attention_score = 0.0
        self.distraction_events = []
        
        self.session_start_time = time.time()
        self.total_frames_processed = 0
        self.engagement_distribution = {level: 0 for level in EngagementLevel}
        
        logger.info("Engagement statistics reset")
    
    def visualize_engagement(
        self,
        frame: np.ndarray,
        results: Dict[str, Any]
    ) -> np.ndarray:
        """Visualize engagement detection results on frame."""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Color coding for engagement levels
        level_colors = {
            EngagementLevel.VERY_LOW: (0, 0, 255),      # Red
            EngagementLevel.LOW: (0, 100, 255),         # Orange
            EngagementLevel.HIGH: (0, 255, 255),        # Yellow
            EngagementLevel.VERY_HIGH: (0, 255, 0)      # Green
        }
        
        engagement_level = results.get('engagement_level', EngagementLevel.HIGH)
        color = level_colors.get(engagement_level, (255, 255, 255))
        
        # Draw face bounding box
        face_bbox = results.get('face_bbox')
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw engagement information panel
        panel_height = 120
        cv2.rectangle(vis_frame, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (10, 10), (400, panel_height), color, 2)
        
        # Engagement level
        level_text = f"Engagement: {engagement_level.name}"
        cv2.putText(vis_frame, level_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Confidence
        confidence = results.get('engagement_confidence', 0.0)
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(vis_frame, conf_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Attention score
        attention_score = results.get('attention_score', 0.0)
        att_text = f"Attention: {attention_score:.2f}"
        cv2.putText(vis_frame, att_text, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Distraction warning
        if results.get('distraction_detected', False):
            cv2.putText(vis_frame, "DISTRACTION DETECTED", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw attention score bar
        bar_x = w - 50
        bar_y = 50
        bar_height = 200
        bar_width = 20
        
        # Background bar
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Attention level bar
        fill_height = int(attention_score * bar_height)
        fill_y = bar_y + bar_height - fill_height
        
        # Color based on attention level
        if attention_score > 0.7:
            bar_color = (0, 255, 0)  # Green
        elif attention_score > 0.4:
            bar_color = (0, 255, 255)  # Yellow
        else:
            bar_color = (0, 0, 255)  # Red
        
        cv2.rectangle(vis_frame, (bar_x, fill_y), (bar_x + bar_width, bar_y + bar_height), 
                     bar_color, -1)
        
        # Bar label
        cv2.putText(vis_frame, "Attention", (bar_x - 20, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
