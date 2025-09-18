"""
Utility functions for eye gaze estimation.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional, List
import dlib
import logging

logger = logging.getLogger(__name__)

# Initialize face landmark predictor
try:
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    face_predictor = dlib.shape_predictor(predictor_path)
    HAS_DLIB_PREDICTOR = True
except:
    HAS_DLIB_PREDICTOR = False
    logger.warning("dlib face landmark predictor not found. Eye region extraction may be limited.")


def preprocess_eye_image(eye_image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Preprocess eye image for model input.
    
    Args:
        eye_image: Eye region image (BGR format)
        target_size: Target image size (height, width)
        
    Returns:
        Preprocessed image tensor
    """
    # Convert BGR to RGB
    eye_rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(eye_rgb)


def postprocess_gaze_vector(gaze_vector: np.ndarray) -> np.ndarray:
    """
    Post-process raw gaze vector prediction.
    
    Args:
        gaze_vector: Raw 3D gaze vector
        
    Returns:
        Normalized gaze vector
    """
    # Ensure it's a unit vector
    norm = np.linalg.norm(gaze_vector)
    if norm > 0:
        gaze_vector = gaze_vector / norm
    
    return gaze_vector


def extract_eye_regions(
    image: np.ndarray, 
    face_bbox: Tuple[int, int, int, int],
    expand_ratio: float = 0.3
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract left and right eye regions from face image.
    
    Args:
        image: Input image
        face_bbox: Face bounding box (x1, y1, x2, y2)
        expand_ratio: Ratio to expand eye regions
        
    Returns:
        Tuple of (left_eye, right_eye) images or None if extraction fails
    """
    try:
        x1, y1, x2, y2 = face_bbox
        face_region = image[y1:y2, x1:x2]
        
        if HAS_DLIB_PREDICTOR:
            return _extract_eyes_with_landmarks(face_region, expand_ratio)
        else:
            return _extract_eyes_simple(face_region, expand_ratio)
            
    except Exception as e:
        logger.error(f"Error extracting eye regions: {e}")
        return None


def _extract_eyes_with_landmarks(
    face_image: np.ndarray, 
    expand_ratio: float = 0.3
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract eyes using facial landmark detection."""
    try:
        # Convert to grayscale for landmark detection
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Create a rectangle for the entire face region
        h, w = gray.shape
        rect = dlib.rectangle(0, 0, w, h)
        
        # Get facial landmarks
        landmarks = face_predictor(gray, rect)
        
        # Eye landmark indices (68-point model)
        left_eye_indices = list(range(36, 42))   # Left eye
        right_eye_indices = list(range(42, 48))  # Right eye
        
        # Extract eye regions
        left_eye = _extract_eye_from_landmarks(face_image, landmarks, left_eye_indices, expand_ratio)
        right_eye = _extract_eye_from_landmarks(face_image, landmarks, right_eye_indices, expand_ratio)
        
        if left_eye is not None and right_eye is not None:
            return left_eye, right_eye
        
    except Exception as e:
        logger.error(f"Landmark-based eye extraction failed: {e}")
    
    # Fallback to simple extraction
    return _extract_eyes_simple(face_image, expand_ratio)


def _extract_eye_from_landmarks(
    face_image: np.ndarray,
    landmarks: dlib.full_object_detection,
    eye_indices: List[int],
    expand_ratio: float
) -> Optional[np.ndarray]:
    """Extract single eye region from landmarks."""
    try:
        # Get eye landmark points
        eye_points = []
        for i in eye_indices:
            point = landmarks.part(i)
            eye_points.append((point.x, point.y))
        
        eye_points = np.array(eye_points)
        
        # Calculate bounding box
        x_min, y_min = np.min(eye_points, axis=0)
        x_max, y_max = np.max(eye_points, axis=0)
        
        # Expand region
        width = x_max - x_min
        height = y_max - y_min
        
        expand_w = int(width * expand_ratio)
        expand_h = int(height * expand_ratio)
        
        x1 = max(0, x_min - expand_w)
        y1 = max(0, y_min - expand_h)
        x2 = min(face_image.shape[1], x_max + expand_w)
        y2 = min(face_image.shape[0], y_max + expand_h)
        
        eye_region = face_image[y1:y2, x1:x2]
        
        # Ensure minimum size
        if eye_region.shape[0] < 20 or eye_region.shape[1] < 20:
            return None
            
        return eye_region
        
    except Exception as e:
        logger.error(f"Error extracting eye from landmarks: {e}")
        return None


def _extract_eyes_simple(
    face_image: np.ndarray, 
    expand_ratio: float = 0.3
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Simple eye extraction based on face proportions."""
    try:
        h, w = face_image.shape[:2]
        
        # Approximate eye positions based on typical face proportions
        eye_y_start = int(h * 0.25)
        eye_y_end = int(h * 0.55)
        
        # Left eye (right side of image due to mirror effect)
        left_eye_x_start = int(w * 0.1)
        left_eye_x_end = int(w * 0.45)
        
        # Right eye (left side of image)
        right_eye_x_start = int(w * 0.55)
        right_eye_x_end = int(w * 0.9)
        
        left_eye = face_image[eye_y_start:eye_y_end, left_eye_x_start:left_eye_x_end]
        right_eye = face_image[eye_y_start:eye_y_end, right_eye_x_start:right_eye_x_end]
        
        # Check if regions are valid
        if left_eye.size == 0 or right_eye.size == 0:
            return None
            
        return left_eye, right_eye
        
    except Exception as e:
        logger.error(f"Simple eye extraction failed: {e}")
        return None


def calculate_gaze_accuracy(
    predicted_vectors: np.ndarray,
    ground_truth_vectors: np.ndarray
) -> float:
    """
    Calculate angular error between predicted and ground truth gaze vectors.
    
    Args:
        predicted_vectors: Predicted gaze vectors [N, 3]
        ground_truth_vectors: Ground truth gaze vectors [N, 3]
        
    Returns:
        Mean angular error in degrees
    """
    # Ensure unit vectors
    pred_norm = predicted_vectors / np.linalg.norm(predicted_vectors, axis=1, keepdims=True)
    gt_norm = ground_truth_vectors / np.linalg.norm(ground_truth_vectors, axis=1, keepdims=True)
    
    # Calculate dot product
    dot_products = np.sum(pred_norm * gt_norm, axis=1)
    
    # Clamp to valid range for arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Calculate angular error
    angular_errors = np.arccos(dot_products)
    
    # Convert to degrees
    angular_errors_deg = np.degrees(angular_errors)
    
    return np.mean(angular_errors_deg)


def visualize_gaze_vector(
    image: np.ndarray,
    gaze_vector: np.ndarray,
    eye_center: Tuple[int, int],
    scale: float = 100.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Visualize gaze vector on image.
    
    Args:
        image: Input image
        gaze_vector: 3D gaze vector
        eye_center: Eye center coordinates
        scale: Scale factor for visualization
        color: Line color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with gaze vector visualization
    """
    result_image = image.copy()
    
    # Project 3D vector to 2D
    x_offset = int(gaze_vector[0] * scale)
    y_offset = int(-gaze_vector[1] * scale)  # Negative for correct direction
    
    start_point = eye_center
    end_point = (eye_center[0] + x_offset, eye_center[1] + y_offset)
    
    # Draw gaze vector
    cv2.arrowedLine(result_image, start_point, end_point, color, thickness)
    
    # Draw eye center
    cv2.circle(result_image, eye_center, 3, color, -1)
    
    return result_image


def smooth_gaze_predictions(
    predictions: List[np.ndarray],
    window_size: int = 5,
    method: str = 'moving_average'
) -> np.ndarray:
    """
    Smooth gaze predictions over time to reduce jitter.
    
    Args:
        predictions: List of gaze vectors
        window_size: Smoothing window size
        method: Smoothing method ('moving_average', 'exponential')
        
    Returns:
        Smoothed gaze vector
    """
    if len(predictions) < window_size:
        return predictions[-1] if predictions else np.array([0, 0, 1])
    
    recent_predictions = predictions[-window_size:]
    
    if method == 'moving_average':
        # Simple moving average
        smoothed = np.mean(recent_predictions, axis=0)
    elif method == 'exponential':
        # Exponential moving average
        weights = np.exp(np.linspace(-1, 0, window_size))
        weights = weights / np.sum(weights)
        smoothed = np.average(recent_predictions, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    # Normalize to unit vector
    norm = np.linalg.norm(smoothed)
    if norm > 0:
        smoothed = smoothed / norm
    
    return smoothed


def estimate_focus_score(
    gaze_vectors: List[np.ndarray],
    target_region: Tuple[int, int, int, int],
    screen_size: Tuple[int, int],
    time_window: float = 5.0
) -> float:
    """
    Estimate focus score based on gaze consistency within a target region.
    
    Args:
        gaze_vectors: Recent gaze vectors
        target_region: Target region (x1, y1, x2, y2)
        screen_size: Screen dimensions (width, height)
        time_window: Time window for analysis in seconds
        
    Returns:
        Focus score between 0 and 1
    """
    if not gaze_vectors:
        return 0.0
    
    # Convert gaze vectors to screen coordinates (simplified)
    screen_points = []
    for gaze_vec in gaze_vectors:
        # Simple projection - in practice this needs proper calibration
        x = int(screen_size[0] * (0.5 + gaze_vec[0] * 0.5))
        y = int(screen_size[1] * (0.5 + gaze_vec[1] * 0.5))
        screen_points.append((x, y))
    
    # Count points within target region
    x1, y1, x2, y2 = target_region
    points_in_target = 0
    
    for x, y in screen_points:
        if x1 <= x <= x2 and y1 <= y <= y2:
            points_in_target += 1
    
    # Calculate focus score
    focus_score = points_in_target / len(screen_points) if screen_points else 0.0
    
    return focus_score


def calibrate_gaze_mapping(
    screen_points: List[Tuple[float, float]],
    gaze_vectors: List[np.ndarray],
    method: str = 'polynomial'
) -> Optional[np.ndarray]:
    """
    Calibrate mapping from gaze vectors to screen coordinates.
    
    Args:
        screen_points: Known screen coordinates
        gaze_vectors: Corresponding gaze vectors
        method: Calibration method ('linear', 'polynomial')
        
    Returns:
        Calibration parameters or None if calibration fails
    """
    if len(screen_points) != len(gaze_vectors) or len(screen_points) < 4:
        logger.error("Insufficient calibration data")
        return None
    
    try:
        X = np.array(gaze_vectors)
        y = np.array(screen_points)
        
        if method == 'linear':
            # Linear regression: screen_coord = A * gaze_vector + b
            X_aug = np.column_stack([X, np.ones(len(X))])
            calibration_params = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            
        elif method == 'polynomial':
            # Polynomial features
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            reg = LinearRegression()
            reg.fit(X_poly, y)
            
            calibration_params = {
                'poly_features': poly,
                'regressor': reg
            }
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        return calibration_params
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return None
