"""
Utility functions for eye gaze estimation module.

Contains preprocessing, postprocessing, and helper functions.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Union
import math


def preprocess_eye_region(eye_image: np.ndarray, target_size: Tuple[int, int] = (60, 36)) -> torch.Tensor:
    """
    Preprocess eye region for model input.
    
    Args:
        eye_image: Raw eye region image
        target_size: Target size (width, height) for resizing
        
    Returns:
        Preprocessed tensor ready for model input
    """
    if eye_image is None or eye_image.size == 0:
        # Return zero tensor if no valid eye region
        return torch.zeros(3, target_size[1], target_size[0])
    
    # Convert to RGB if needed
    if len(eye_image.shape) == 3 and eye_image.shape[2] == 3:
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
    elif len(eye_image.shape) == 2:
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_GRAY2RGB)
    
    # Resize to target size
    resized = cv2.resize(eye_image, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Apply histogram equalization to improve contrast
    normalized = apply_histogram_equalization(normalized)
    
    # Convert to tensor and change to CHW format
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)
    
    # Normalize with ImageNet statistics (commonly used)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    
    return tensor


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to improve eye region contrast.
    
    Args:
        image: Input image in range [0, 1]
        
    Returns:
        Contrast-enhanced image
    """
    # Convert to uint8 for OpenCV processing
    uint8_image = (image * 255).astype(np.uint8)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    
    if len(uint8_image.shape) == 3:
        # Process each channel separately
        enhanced = np.zeros_like(uint8_image)
        for i in range(3):
            enhanced[:, :, i] = clahe.apply(uint8_image[:, :, i])
    else:
        enhanced = clahe.apply(uint8_image)
    
    # Convert back to float
    return enhanced.astype(np.float32) / 255.0


def calculate_gaze_vector(gaze_angles: np.ndarray) -> Tuple[float, float]:
    """
    Convert gaze vector to yaw and pitch angles.
    
    Args:
        gaze_angles: Gaze vector or angles [yaw, pitch] or [x, y, z]
        
    Returns:
        Tuple of (yaw, pitch) in radians
    """
    if len(gaze_angles) == 2:
        # Already in yaw, pitch format
        return float(gaze_angles[0]), float(gaze_angles[1])
    elif len(gaze_angles) == 3:
        # Convert from 3D gaze vector to angles
        x, y, z = gaze_angles
        
        # Calculate yaw (horizontal angle)
        yaw = math.atan2(x, z)
        
        # Calculate pitch (vertical angle)
        pitch = math.asin(y / math.sqrt(x*x + y*y + z*z))
        
        return float(yaw), float(pitch)
    else:
        raise ValueError(f"Invalid gaze vector dimension: {len(gaze_angles)}")


def angles_to_gaze_vector(yaw: float, pitch: float) -> np.ndarray:
    """
    Convert yaw and pitch angles to 3D gaze vector.
    
    Args:
        yaw: Horizontal angle in radians
        pitch: Vertical angle in radians
        
    Returns:
        3D gaze vector [x, y, z]
    """
    x = math.sin(yaw) * math.cos(pitch)
    y = math.sin(pitch)
    z = math.cos(yaw) * math.cos(pitch)
    
    return np.array([x, y, z])


def angular_error(pred_yaw: float, pred_pitch: float, 
                 true_yaw: float, true_pitch: float) -> float:
    """
    Calculate angular error between predicted and true gaze directions.
    
    Args:
        pred_yaw: Predicted yaw angle in radians
        pred_pitch: Predicted pitch angle in radians
        true_yaw: True yaw angle in radians
        true_pitch: True pitch angle in radians
        
    Returns:
        Angular error in degrees
    """
    # Convert to gaze vectors
    pred_vector = angles_to_gaze_vector(pred_yaw, pred_pitch)
    true_vector = angles_to_gaze_vector(true_yaw, true_pitch)
    
    # Calculate cosine of angle between vectors
    cos_angle = np.dot(pred_vector, true_vector) / (
        np.linalg.norm(pred_vector) * np.linalg.norm(true_vector)
    )
    
    # Clamp to valid range to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate angle in radians and convert to degrees
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def smooth_gaze_predictions(gaze_history: list, window_size: int = 5, 
                          alpha: float = 0.7) -> Tuple[float, float]:
    """
    Smooth gaze predictions using exponential moving average.
    
    Args:
        gaze_history: List of recent (yaw, pitch) predictions
        window_size: Size of smoothing window
        alpha: Smoothing factor (0 = no smoothing, 1 = no history)
        
    Returns:
        Smoothed (yaw, pitch) predictions
    """
    if not gaze_history:
        return 0.0, 0.0
    
    if len(gaze_history) == 1:
        return gaze_history[0]
    
    # Use only recent predictions within window
    recent_gazes = gaze_history[-window_size:]
    
    # Apply exponential moving average
    smoothed_yaw = recent_gazes[0][0]
    smoothed_pitch = recent_gazes[0][1]
    
    for yaw, pitch in recent_gazes[1:]:
        smoothed_yaw = alpha * yaw + (1 - alpha) * smoothed_yaw
        smoothed_pitch = alpha * pitch + (1 - alpha) * smoothed_pitch
    
    return smoothed_yaw, smoothed_pitch


def detect_eye_landmarks(eye_region: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect eye landmarks within eye region.
    
    Args:
        eye_region: Cropped eye region image
        
    Returns:
        Array of landmark coordinates or None if detection fails
    """
    try:
        import dlib
        
        # Initialize dlib's face detector and landmark predictor
        # Note: This requires the shape_predictor_68_face_landmarks.dat file
        predictor_path = "models/shape_predictor_68_face_landmarks.dat"
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
        # Convert to grayscale
        if len(eye_region.shape) == 3:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_region
        
        # Detect faces (should find the eye region)
        faces = detector(gray)
        
        if len(faces) > 0:
            # Get landmarks for the first detected face
            landmarks = predictor(gray, faces[0])
            
            # Extract eye landmarks (points 36-47 for eyes)
            eye_points = []
            for i in range(36, 48):  # Eye landmark indices
                point = landmarks.part(i)
                eye_points.append([point.x, point.y])
            
            return np.array(eye_points)
    
    except ImportError:
        print("Warning: dlib not available for landmark detection")
    except Exception as e:
        print(f"Warning: Landmark detection failed: {e}")
    
    return None


def estimate_eye_center(eye_region: np.ndarray) -> Tuple[int, int]:
    """
    Estimate the center of the eye pupil/iris.
    
    Args:
        eye_region: Cropped eye region image
        
    Returns:
        Estimated (x, y) coordinates of eye center
    """
    if eye_region is None or eye_region.size == 0:
        return 0, 0
    
    # Convert to grayscale
    if len(eye_region.shape) == 3:
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = eye_region
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find the darkest region (pupil)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    
    # The minimum location should correspond to the pupil center
    pupil_x, pupil_y = min_loc
    
    # Validate the detected center is within reasonable bounds
    h, w = gray.shape
    if 0.2 * w <= pupil_x <= 0.8 * w and 0.2 * h <= pupil_y <= 0.8 * h:
        return pupil_x, pupil_y
    else:
        # Fall back to image center if detection seems unreliable
        return w // 2, h // 2


def visualize_gaze_on_image(image: np.ndarray, yaw: float, pitch: float, 
                          face_center: Tuple[int, int], 
                          arrow_length: int = 100) -> np.ndarray:
    """
    Visualize gaze direction on an image.
    
    Args:
        image: Input image
        yaw: Gaze yaw angle in radians
        pitch: Gaze pitch angle in radians
        face_center: Center point of face (x, y)
        arrow_length: Length of gaze arrow
        
    Returns:
        Image with gaze visualization
    """
    result_image = image.copy()
    
    # Calculate end point of gaze arrow
    center_x, center_y = face_center
    end_x = int(center_x + arrow_length * math.sin(yaw))
    end_y = int(center_y - arrow_length * math.sin(pitch))
    
    # Draw gaze arrow
    cv2.arrowedLine(result_image, (center_x, center_y), (end_x, end_y), 
                   (0, 0, 255), 3, tipLength=0.3)
    
    # Add angle text
    cv2.putText(result_image, f"Yaw: {math.degrees(yaw):.1f}°", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_image, f"Pitch: {math.degrees(pitch):.1f}°", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result_image


def create_gaze_heatmap(gaze_points: list, image_shape: Tuple[int, int], 
                       sigma: float = 20.0) -> np.ndarray:
    """
    Create a heatmap of gaze points.
    
    Args:
        gaze_points: List of (x, y) gaze coordinates
        image_shape: Shape of output heatmap (height, width)
        sigma: Gaussian kernel sigma for smoothing
        
    Returns:
        Heatmap as numpy array
    """
    heatmap = np.zeros(image_shape, dtype=np.float32)
    
    for x, y in gaze_points:
        # Ensure coordinates are within bounds
        x = max(0, min(image_shape[1] - 1, int(x)))
        y = max(0, min(image_shape[0] - 1, int(y)))
        
        # Add Gaussian blob at gaze point
        y_coords, x_coords = np.ogrid[:image_shape[0], :image_shape[1]]
        mask = ((x_coords - x) ** 2 + (y_coords - y) ** 2) <= (3 * sigma) ** 2
        
        gaussian = np.exp(-((x_coords - x) ** 2 + (y_coords - y) ** 2) / (2 * sigma ** 2))
        heatmap[mask] += gaussian[mask]
    
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap


