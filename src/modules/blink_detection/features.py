"""
Feature extraction for blink detection and drowsiness analysis.

Extracts various features from eye regions and facial landmarks
for machine learning-based blink detection and drowsiness assessment.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math

from .ear_calculator import calculate_eye_aspect_ratio, euclidean_distance


def extract_blink_features(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive features for blink detection.
    
    Args:
        frame: Input video frame
        landmarks: Facial landmarks (68 points)
        
    Returns:
        Feature vector for blink classification
    """
    features = []
    
    if len(landmarks) < 68:
        # Return zero features if landmarks are insufficient
        return np.zeros(12)
    
    # Extract eye landmarks
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    
    # 1. Eye Aspect Ratios
    left_ear = calculate_eye_aspect_ratio(left_eye)
    right_ear = calculate_eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    
    features.extend([left_ear, right_ear, avg_ear])
    
    # 2. Eye width and height ratios
    left_width = euclidean_distance(left_eye[0], left_eye[3])
    left_height = (euclidean_distance(left_eye[1], left_eye[5]) + 
                   euclidean_distance(left_eye[2], left_eye[4])) / 2.0
    
    right_width = euclidean_distance(right_eye[0], right_eye[3])
    right_height = (euclidean_distance(right_eye[1], right_eye[5]) + 
                    euclidean_distance(right_eye[2], right_eye[4])) / 2.0
    
    left_ratio = left_height / left_width if left_width > 0 else 0
    right_ratio = right_height / right_width if right_width > 0 else 0
    
    features.extend([left_ratio, right_ratio])
    
    # 3. Eye center distances (measure of eye opening)
    left_center_dist = euclidean_distance(left_eye[1], left_eye[5])
    right_center_dist = euclidean_distance(right_eye[1], right_eye[5])
    
    features.extend([left_center_dist, right_center_dist])
    
    # 4. Eyebrow to eye distances (can indicate drowsiness)
    left_eyebrow = landmarks[17:22]
    right_eyebrow = landmarks[22:27]
    
    left_brow_eye_dist = np.mean([euclidean_distance(left_eyebrow[i], left_eye[i+1]) 
                                 for i in range(min(len(left_eyebrow), len(left_eye)-1))])
    right_brow_eye_dist = np.mean([euclidean_distance(right_eyebrow[i], right_eye[i+1]) 
                                  for i in range(min(len(right_eyebrow), len(right_eye)-1))])
    
    features.extend([left_brow_eye_dist, right_brow_eye_dist])
    
    # 5. Eye symmetry measure
    eye_symmetry = abs(left_ear - right_ear)
    features.append(eye_symmetry)
    
    # 6. Additional geometric features
    # Distance between eye corners
    eye_corner_distance = euclidean_distance(left_eye[3], right_eye[0])
    features.append(eye_corner_distance)
    
    return np.array(features, dtype=np.float32)


def extract_drowsiness_features(frame: np.ndarray, landmarks: np.ndarray, 
                               ear_history: List[float], 
                               timestamps: List[float]) -> np.ndarray:
    """
    Extract features specifically for drowsiness detection.
    
    Args:
        frame: Current video frame
        landmarks: Facial landmarks
        ear_history: Recent EAR values
        timestamps: Corresponding timestamps
        
    Returns:
        Feature vector for drowsiness classification
    """
    features = []
    
    # Basic blink features
    blink_features = extract_blink_features(frame, landmarks)
    features.extend(blink_features)
    
    # Temporal features from EAR history
    if len(ear_history) > 0:
        # Statistical measures
        features.append(np.mean(ear_history))
        features.append(np.std(ear_history))
        features.append(np.min(ear_history))
        features.append(np.max(ear_history))
        features.append(np.median(ear_history))
        
        # Trend analysis
        if len(ear_history) >= 5:
            recent_trend = np.polyfit(range(len(ear_history)), ear_history, 1)[0]
            features.append(recent_trend)
        else:
            features.append(0.0)
        
        # Variability measures
        if len(ear_history) > 1:
            variability = np.std(ear_history) / np.mean(ear_history) if np.mean(ear_history) > 0 else 0
            features.append(variability)
        else:
            features.append(0.0)
    else:
        # Add zeros if no history available
        features.extend([0.0] * 7)
    
    # Mouth features (for yawning detection)
    if len(landmarks) >= 68:
        mouth_landmarks = landmarks[48:68]
        mouth_features = extract_mouth_features(mouth_landmarks)
        features.extend(mouth_features)
    else:
        features.extend([0.0] * 4)  # Add zeros for missing mouth features
    
    return np.array(features, dtype=np.float32)


def extract_mouth_features(mouth_landmarks: np.ndarray) -> List[float]:
    """
    Extract mouth-related features for yawning detection.
    
    Args:
        mouth_landmarks: Mouth landmark points
        
    Returns:
        List of mouth features
    """
    features = []
    
    if len(mouth_landmarks) < 12:
        return [0.0] * 4
    
    # Mouth Aspect Ratio (MAR)
    # Using key mouth points
    top_lip = mouth_landmarks[13]  # Top of upper lip
    bottom_lip = mouth_landmarks[19]  # Bottom of lower lip
    left_corner = mouth_landmarks[0]  # Left corner
    right_corner = mouth_landmarks[6]  # Right corner
    
    vertical_dist = euclidean_distance(top_lip, bottom_lip)
    horizontal_dist = euclidean_distance(left_corner, right_corner)
    
    mar = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
    features.append(mar)
    
    # Mouth width and height
    features.append(horizontal_dist)
    features.append(vertical_dist)
    
    # Mouth opening ratio (height/width)
    opening_ratio = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
    features.append(opening_ratio)
    
    return features


def extract_head_pose_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract head pose features that can indicate drowsiness.
    
    Args:
        landmarks: Facial landmarks
        
    Returns:
        Head pose feature vector
    """
    if len(landmarks) < 68:
        return np.zeros(6)
    
    features = []
    
    # Nose tip and chin
    nose_tip = landmarks[30]
    chin = landmarks[8]
    
    # Left and right face boundaries
    left_face = landmarks[0]
    right_face = landmarks[16]
    
    # Calculate head tilt angle
    face_width = euclidean_distance(left_face, right_face)
    face_center_x = (left_face[0] + right_face[0]) / 2
    
    # Horizontal deviation of nose from face center
    nose_deviation = abs(nose_tip[0] - face_center_x)
    normalized_deviation = nose_deviation / face_width if face_width > 0 else 0
    
    features.append(normalized_deviation)
    
    # Vertical face alignment
    face_height = euclidean_distance(landmarks[27], chin)  # Nose bridge to chin
    nose_chin_dist = euclidean_distance(nose_tip, chin)
    
    vertical_ratio = nose_chin_dist / face_height if face_height > 0 else 0
    features.append(vertical_ratio)
    
    # Eye-nose-mouth alignment
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)
    mouth_center = np.mean(landmarks[48:68], axis=0)
    
    # Calculate angles
    eye_line_angle = math.atan2(right_eye_center[1] - left_eye_center[1],
                               right_eye_center[0] - left_eye_center[0])
    
    nose_mouth_angle = math.atan2(mouth_center[1] - nose_tip[1],
                                 mouth_center[0] - nose_tip[0])
    
    features.extend([eye_line_angle, nose_mouth_angle])
    
    # Face symmetry measures
    left_distances = [euclidean_distance(landmarks[i], nose_tip) for i in range(0, 8)]
    right_distances = [euclidean_distance(landmarks[i], nose_tip) for i in range(9, 17)]
    
    symmetry_score = np.mean([abs(l - r) for l, r in zip(left_distances, right_distances)])
    features.append(symmetry_score)
    
    # Overall face compactness (can indicate head position)
    face_area = face_width * face_height
    features.append(face_area)
    
    return np.array(features, dtype=np.float32)


def extract_texture_features(eye_region: np.ndarray) -> np.ndarray:
    """
    Extract texture features from eye region for blink detection.
    
    Args:
        eye_region: Cropped eye region image
        
    Returns:
        Texture feature vector
    """
    if eye_region is None or eye_region.size == 0:
        return np.zeros(8)
    
    features = []
    
    # Convert to grayscale if needed
    if len(eye_region.shape) == 3:
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = eye_region.copy()
    
    # Ensure minimum size
    if gray.shape[0] < 10 or gray.shape[1] < 10:
        return np.zeros(8)
    
    # 1. Mean intensity
    mean_intensity = np.mean(gray)
    features.append(mean_intensity)
    
    # 2. Standard deviation
    std_intensity = np.std(gray)
    features.append(std_intensity)
    
    # 3. Intensity range
    intensity_range = np.max(gray) - np.min(gray)
    features.append(intensity_range)
    
    # 4. Gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mean_gradient = np.mean(gradient_magnitude)
    features.append(mean_gradient)
    
    # 5. Local Binary Pattern (simplified)
    try:
        lbp = calculate_lbp(gray)
        lbp_variance = np.var(lbp)
        features.append(lbp_variance)
    except:
        features.append(0.0)
    
    # 6. Contrast measure
    contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
    features.append(contrast)
    
    # 7. Energy (sum of squared intensities)
    normalized_gray = gray.astype(np.float32) / 255.0
    energy = np.sum(normalized_gray**2)
    features.append(energy)
    
    # 8. Entropy (measure of randomness)
    try:
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small value to avoid log(0)
        features.append(entropy)
    except:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)


def calculate_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    Calculate Local Binary Pattern for texture analysis.
    
    Args:
        image: Grayscale image
        radius: Radius of circular neighborhood
        n_points: Number of sample points
        
    Returns:
        LBP image
    """
    h, w = image.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center_pixel = image[i, j]
            binary_string = ""
            
            for p in range(n_points):
                angle = 2 * np.pi * p / n_points
                x = i + radius * np.cos(angle)
                y = j + radius * np.sin(angle)
                
                # Bilinear interpolation
                x1, y1 = int(x), int(y)
                x2, y2 = min(x1 + 1, h - 1), min(y1 + 1, w - 1)
                
                if x1 == x2 and y1 == y2:
                    neighbor_pixel = image[x1, y1]
                else:
                    # Simple nearest neighbor for speed
                    neighbor_pixel = image[int(round(x)), int(round(y))]
                
                binary_string += "1" if neighbor_pixel >= center_pixel else "0"
            
            lbp[i, j] = int(binary_string, 2)
    
    return lbp


def normalize_features(features: np.ndarray, feature_stats: Optional[Dict] = None) -> np.ndarray:
    """
    Normalize features for machine learning models.
    
    Args:
        features: Raw feature vector
        feature_stats: Optional statistics for normalization (mean, std)
        
    Returns:
        Normalized feature vector
    """
    if feature_stats is None:
        # Z-score normalization using feature's own statistics
        mean = np.mean(features)
        std = np.std(features)
        if std > 0:
            return (features - mean) / std
        else:
            return features
    else:
        # Use provided statistics
        mean = feature_stats.get('mean', 0)
        std = feature_stats.get('std', 1)
        if std > 0:
            return (features - mean) / std
        else:
            return features


def create_feature_vector(frame: np.ndarray, landmarks: np.ndarray, 
                         ear_history: List[float], timestamps: List[float],
                         include_texture: bool = True) -> np.ndarray:
    """
    Create comprehensive feature vector for drowsiness detection.
    
    Args:
        frame: Current video frame
        landmarks: Facial landmarks
        ear_history: Recent EAR values
        timestamps: Corresponding timestamps
        include_texture: Whether to include texture features
        
    Returns:
        Complete feature vector
    """
    features = []
    
    # Core drowsiness features
    drowsiness_features = extract_drowsiness_features(frame, landmarks, ear_history, timestamps)
    features.extend(drowsiness_features)
    
    # Head pose features
    head_pose_features = extract_head_pose_features(landmarks)
    features.extend(head_pose_features)
    
    # Texture features (if requested)
    if include_texture and landmarks is not None and len(landmarks) >= 42:
        # Extract eye regions
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Get bounding boxes
        left_bbox = get_eye_bounding_box(left_eye)
        right_bbox = get_eye_bounding_box(right_eye)
        
        # Extract eye regions
        left_eye_region = extract_eye_region(frame, left_bbox)
        right_eye_region = extract_eye_region(frame, right_bbox)
        
        # Get texture features
        left_texture = extract_texture_features(left_eye_region)
        right_texture = extract_texture_features(right_eye_region)
        
        features.extend(left_texture)
        features.extend(right_texture)
    
    return np.array(features, dtype=np.float32)


def get_eye_bounding_box(eye_landmarks: np.ndarray, padding: int = 5) -> Tuple[int, int, int, int]:
    """
    Get bounding box for eye region.
    
    Args:
        eye_landmarks: Eye landmark points
        padding: Padding around eye region
        
    Returns:
        Bounding box (x, y, width, height)
    """
    if len(eye_landmarks) == 0:
        return (0, 0, 1, 1)
    
    min_x = int(np.min(eye_landmarks[:, 0])) - padding
    max_x = int(np.max(eye_landmarks[:, 0])) + padding
    min_y = int(np.min(eye_landmarks[:, 1])) - padding
    max_y = int(np.max(eye_landmarks[:, 1])) + padding
    
    width = max_x - min_x
    height = max_y - min_y
    
    return (max(0, min_x), max(0, min_y), max(1, width), max(1, height))


def extract_eye_region(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Extract eye region from frame using bounding box.
    
    Args:
        frame: Input frame
        bbox: Bounding box (x, y, width, height)
        
    Returns:
        Cropped eye region or None if extraction fails
    """
    x, y, w, h = bbox
    
    # Ensure coordinates are within frame bounds
    frame_h, frame_w = frame.shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    x2 = max(x + 1, min(x + w, frame_w))
    y2 = max(y + 1, min(y + h, frame_h))
    
    if x2 <= x or y2 <= y:
        return None
    
    eye_region = frame[y:y2, x:x2]
    
    if eye_region.size == 0:
        return None
    
    return eye_region


