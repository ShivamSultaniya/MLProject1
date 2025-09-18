"""
Eye Aspect Ratio (EAR) Calculator

Implements the Eye Aspect Ratio calculation for blink detection.
Based on the paper "Real-Time Eye Blink Detection using Facial Landmarks"
by Soukupová and Čech.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import math


def calculate_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for a given set of eye landmarks.
    
    The EAR is calculated as:
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    
    Where p1-p6 are the 6 eye landmark points in order:
    p1, p4: horizontal eye corners
    p2, p3, p5, p6: vertical eye points
    
    Args:
        eye_landmarks: Array of 6 eye landmark points [(x,y), ...]
        
    Returns:
        Eye aspect ratio value (typically 0.2-0.4 for open eyes, <0.2 for closed)
    """
    if len(eye_landmarks) < 6:
        return 0.0
    
    # Extract landmark points
    p1 = eye_landmarks[0]  # Left corner
    p2 = eye_landmarks[1]  # Top left
    p3 = eye_landmarks[2]  # Top right
    p4 = eye_landmarks[3]  # Right corner
    p5 = eye_landmarks[4]  # Bottom right
    p6 = eye_landmarks[5]  # Bottom left
    
    # Calculate euclidean distances
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    
    # Avoid division by zero
    if horizontal == 0:
        return 0.0
    
    # Calculate EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    return ear


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_mar(mouth_landmarks: np.ndarray) -> float:
    """
    Calculate the Mouth Aspect Ratio (MAR) for drowsiness detection.
    
    Args:
        mouth_landmarks: Array of mouth landmark points
        
    Returns:
        Mouth aspect ratio value
    """
    if len(mouth_landmarks) < 6:
        return 0.0
    
    # Use key mouth points for MAR calculation
    # Typically points from the inner mouth boundary
    top_lip = mouth_landmarks[2]
    bottom_lip = mouth_landmarks[4]
    left_corner = mouth_landmarks[0]
    right_corner = mouth_landmarks[5]
    
    vertical = euclidean_distance(top_lip, bottom_lip)
    horizontal = euclidean_distance(left_corner, right_corner)
    
    if horizontal == 0:
        return 0.0
    
    mar = vertical / horizontal
    return mar


def detect_eye_landmarks(frame: np.ndarray, face_landmarks: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect eye landmarks from a frame or extract from face landmarks.
    
    Args:
        frame: Input video frame
        face_landmarks: Optional pre-detected face landmarks (68 points)
        
    Returns:
        Tuple of (left_eye_landmarks, right_eye_landmarks)
    """
    if face_landmarks is not None and len(face_landmarks) >= 68:
        # Extract eye landmarks from 68-point face landmarks
        left_eye = face_landmarks[36:42]   # Points 36-41 for left eye
        right_eye = face_landmarks[42:48]  # Points 42-47 for right eye
        return left_eye, right_eye
    
    # Fallback: detect eyes using OpenCV cascades
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces first
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, None
    
    # Use the largest face
    face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = face
    
    # Detect eyes within face region
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    face_roi = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)
    
    if len(eyes) < 2:
        return None, None
    
    # Sort eyes by x-coordinate (left to right)
    eyes = sorted(eyes, key=lambda x: x[0])
    
    # Generate approximate eye landmarks for the two detected eyes
    left_eye_landmarks = generate_eye_landmarks(eyes[0], x, y)
    right_eye_landmarks = generate_eye_landmarks(eyes[1], x, y)
    
    return left_eye_landmarks, right_eye_landmarks


def generate_eye_landmarks(eye_rect: Tuple[int, int, int, int], 
                          face_x: int, face_y: int) -> np.ndarray:
    """
    Generate approximate eye landmarks from eye bounding box.
    
    Args:
        eye_rect: Eye bounding box (x, y, w, h) relative to face
        face_x: Face x-coordinate offset
        face_y: Face y-coordinate offset
        
    Returns:
        Array of 6 eye landmark points
    """
    ex, ey, ew, eh = eye_rect
    
    # Convert to absolute coordinates
    abs_x = face_x + ex
    abs_y = face_y + ey
    
    # Generate 6 key eye points in clockwise order
    landmarks = np.array([
        [abs_x, abs_y + eh//2],                    # Left corner
        [abs_x + ew//4, abs_y],                    # Top left
        [abs_x + 3*ew//4, abs_y],                  # Top right
        [abs_x + ew, abs_y + eh//2],               # Right corner
        [abs_x + 3*ew//4, abs_y + eh],             # Bottom right
        [abs_x + ew//4, abs_y + eh]                # Bottom left
    ])
    
    return landmarks


def smooth_ear_values(ear_history: List[float], window_size: int = 5) -> float:
    """
    Smooth EAR values using moving average to reduce noise.
    
    Args:
        ear_history: List of recent EAR values
        window_size: Size of smoothing window
        
    Returns:
        Smoothed EAR value
    """
    if not ear_history:
        return 0.0
    
    # Use the most recent values within window
    recent_values = ear_history[-window_size:]
    return sum(recent_values) / len(recent_values)


def adaptive_ear_threshold(ear_history: List[float], 
                          percentile: float = 50.0) -> float:
    """
    Calculate adaptive EAR threshold based on historical values.
    
    Args:
        ear_history: List of historical EAR values
        percentile: Percentile to use for threshold (lower = more sensitive)
        
    Returns:
        Adaptive threshold value
    """
    if len(ear_history) < 10:
        return 0.25  # Default threshold
    
    # Calculate threshold based on percentile of recent EAR values
    threshold = np.percentile(ear_history, percentile)
    
    # Clamp to reasonable range
    threshold = max(0.15, min(0.35, threshold))
    
    return threshold


def validate_eye_landmarks(landmarks: np.ndarray) -> bool:
    """
    Validate that eye landmarks are reasonable.
    
    Args:
        landmarks: Eye landmark points
        
    Returns:
        True if landmarks appear valid
    """
    if len(landmarks) < 6:
        return False
    
    # Check if landmarks form a reasonable eye shape
    # Eye should have some width and height
    min_x = np.min(landmarks[:, 0])
    max_x = np.max(landmarks[:, 0])
    min_y = np.min(landmarks[:, 1])
    max_y = np.max(landmarks[:, 1])
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Eye should have reasonable aspect ratio
    if width < 5 or height < 3:
        return False
    
    if width / height > 10 or height / width > 3:
        return False
    
    return True


def calculate_ear_confidence(ear_value: float, baseline_ear: float = 0.3) -> float:
    """
    Calculate confidence score for EAR-based blink detection.
    
    Args:
        ear_value: Current EAR value
        baseline_ear: Baseline (open eye) EAR value
        
    Returns:
        Confidence score (0-1)
    """
    # Higher confidence for values further from baseline
    deviation = abs(ear_value - baseline_ear)
    confidence = min(1.0, deviation / baseline_ear)
    
    return confidence


def detect_partial_blink(ear_value: float, threshold: float = 0.25, 
                        partial_threshold: float = 0.28) -> str:
    """
    Detect different types of eye closure.
    
    Args:
        ear_value: Current EAR value
        threshold: Full blink threshold
        partial_threshold: Partial blink threshold
        
    Returns:
        Eye state: 'open', 'partial', 'closed'
    """
    if ear_value < threshold:
        return 'closed'
    elif ear_value < partial_threshold:
        return 'partial'
    else:
        return 'open'


class EARAnalyzer:
    """
    Advanced EAR analysis with temporal features.
    """
    
    def __init__(self, history_length: int = 30):
        self.history_length = history_length
        self.ear_history = []
        self.timestamps = []
    
    def add_measurement(self, ear_value: float, timestamp: float) -> None:
        """Add new EAR measurement."""
        self.ear_history.append(ear_value)
        self.timestamps.append(timestamp)
        
        # Maintain history length
        if len(self.ear_history) > self.history_length:
            self.ear_history.pop(0)
            self.timestamps.pop(0)
    
    def get_statistics(self) -> dict:
        """Get statistical measures of EAR values."""
        if not self.ear_history:
            return {}
        
        ear_array = np.array(self.ear_history)
        
        return {
            'mean': np.mean(ear_array),
            'std': np.std(ear_array),
            'min': np.min(ear_array),
            'max': np.max(ear_array),
            'median': np.median(ear_array),
            'range': np.max(ear_array) - np.min(ear_array)
        }
    
    def detect_trend(self) -> str:
        """Detect trend in EAR values."""
        if len(self.ear_history) < 5:
            return 'stable'
        
        # Simple trend detection using linear regression slope
        x = np.arange(len(self.ear_history))
        y = np.array(self.ear_history)
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_variability(self) -> float:
        """Calculate EAR variability (coefficient of variation)."""
        if not self.ear_history:
            return 0.0
        
        mean_ear = np.mean(self.ear_history)
        std_ear = np.std(self.ear_history)
        
        if mean_ear == 0:
            return 0.0
        
        return std_ear / mean_ear


