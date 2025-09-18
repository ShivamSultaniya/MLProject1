"""
Utility functions for blink and drowsiness detection.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging
from scipy import signal
from collections import deque

logger = logging.getLogger(__name__)


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) from eye landmarks.
    
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    where p1-p6 are the eye landmarks in order.
    
    Args:
        eye_landmarks: Array of 6 eye landmark points [(x, y), ...]
        
    Returns:
        EAR value
    """
    if len(eye_landmarks) != 6:
        raise ValueError("Expected 6 eye landmarks")
    
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # |p2-p6|
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # |p3-p5|
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # |p1-p4|
    
    # Compute the eye aspect ratio
    if C == 0:
        return 0.0
    
    ear = (A + B) / (2.0 * C)
    return ear


def detect_blink_sequence(
    ear_sequence: List[float], 
    threshold: float = 0.25,
    min_frames: int = 2,
    max_frames: int = 20
) -> List[Dict[str, Any]]:
    """
    Detect blinks in a sequence of EAR values.
    
    Args:
        ear_sequence: Sequence of EAR values
        threshold: EAR threshold for blink detection
        min_frames: Minimum frames for valid blink
        max_frames: Maximum frames for valid blink
        
    Returns:
        List of detected blinks with metadata
    """
    blinks = []
    
    if len(ear_sequence) < min_frames:
        return blinks
    
    # Find frames where EAR is below threshold
    below_threshold = [ear < threshold for ear in ear_sequence]
    
    # Find consecutive sequences
    in_blink = False
    blink_start = 0
    
    for i, is_below in enumerate(below_threshold):
        if is_below and not in_blink:
            # Start of potential blink
            in_blink = True
            blink_start = i
        elif not is_below and in_blink:
            # End of potential blink
            in_blink = False
            blink_length = i - blink_start
            
            # Check if it's a valid blink
            if min_frames <= blink_length <= max_frames:
                blink_sequence = ear_sequence[blink_start:i]
                
                blinks.append({
                    'start_frame': blink_start,
                    'end_frame': i - 1,
                    'duration_frames': blink_length,
                    'min_ear': min(blink_sequence),
                    'avg_ear': np.mean(blink_sequence),
                    'ear_sequence': blink_sequence
                })
    
    # Handle case where sequence ends during a blink
    if in_blink:
        blink_length = len(ear_sequence) - blink_start
        if min_frames <= blink_length <= max_frames:
            blink_sequence = ear_sequence[blink_start:]
            
            blinks.append({
                'start_frame': blink_start,
                'end_frame': len(ear_sequence) - 1,
                'duration_frames': blink_length,
                'min_ear': min(blink_sequence),
                'avg_ear': np.mean(blink_sequence),
                'ear_sequence': blink_sequence
            })
    
    return blinks


def calculate_perclos(
    ear_sequence: List[float], 
    threshold: float = 0.25,
    time_window: float = 60.0,
    fps: float = 30.0
) -> float:
    """
    Calculate PERCLOS (Percentage of Eye Closure) over a time window.
    
    Args:
        ear_sequence: Sequence of EAR values
        threshold: EAR threshold for eye closure
        time_window: Time window in seconds
        fps: Frames per second
        
    Returns:
        PERCLOS percentage (0-1)
    """
    if not ear_sequence:
        return 0.0
    
    # Calculate number of frames for time window
    window_frames = int(time_window * fps)
    
    # Use the most recent frames within the window
    recent_sequence = ear_sequence[-min(len(ear_sequence), window_frames):]
    
    # Calculate percentage of frames with eyes closed
    closed_frames = sum(1 for ear in recent_sequence if ear < threshold)
    perclos = closed_frames / len(recent_sequence)
    
    return perclos


def estimate_drowsiness_level(
    perclos: float,
    blink_rate: float,
    avg_ear: float,
    ear_variance: float
) -> Tuple[int, float]:
    """
    Estimate drowsiness level based on multiple metrics.
    
    Args:
        perclos: PERCLOS value (0-1)
        blink_rate: Blinks per minute
        avg_ear: Average EAR value
        ear_variance: EAR variance
        
    Returns:
        Tuple of (drowsiness_level, confidence_score)
        Levels: 0=Alert, 1=Slightly drowsy, 2=Moderately drowsy, 3=Very drowsy, 4=Extremely drowsy
    """
    score = 0.0
    
    # PERCLOS contribution (most important factor)
    if perclos >= 0.35:
        score += 0.4  # Very high closure
    elif perclos >= 0.25:
        score += 0.3  # High closure
    elif perclos >= 0.15:
        score += 0.2  # Moderate closure
    elif perclos >= 0.08:
        score += 0.1  # Slight closure
    
    # Blink rate contribution
    if blink_rate < 5:  # Too few blinks
        score += 0.2
    elif blink_rate > 30:  # Too many blinks
        score += 0.15
    elif blink_rate < 10:  # Below normal
        score += 0.1
    
    # Average EAR contribution
    if avg_ear < 0.2:  # Very low average EAR
        score += 0.2
    elif avg_ear < 0.25:  # Low average EAR
        score += 0.1
    
    # EAR variance contribution (instability indicator)
    if ear_variance > 0.01:  # High variability
        score += 0.1
    elif ear_variance < 0.001:  # Very stable (potentially drowsy)
        score += 0.05
    
    # Determine drowsiness level
    if score >= 0.7:
        level = 4  # Extremely drowsy
    elif score >= 0.5:
        level = 3  # Very drowsy
    elif score >= 0.3:
        level = 2  # Moderately drowsy
    elif score >= 0.15:
        level = 1  # Slightly drowsy
    else:
        level = 0  # Alert
    
    confidence = min(1.0, score)
    
    return level, confidence


def smooth_ear_sequence(
    ear_sequence: List[float],
    window_size: int = 5,
    method: str = 'moving_average'
) -> List[float]:
    """
    Smooth EAR sequence to reduce noise.
    
    Args:
        ear_sequence: Raw EAR sequence
        window_size: Smoothing window size
        method: Smoothing method ('moving_average', 'gaussian', 'median')
        
    Returns:
        Smoothed EAR sequence
    """
    if len(ear_sequence) < window_size:
        return ear_sequence.copy()
    
    smoothed = []
    
    if method == 'moving_average':
        for i in range(len(ear_sequence)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(ear_sequence), i + window_size // 2 + 1)
            window_values = ear_sequence[start_idx:end_idx]
            smoothed.append(np.mean(window_values))
    
    elif method == 'gaussian':
        # Apply Gaussian filter
        sigma = window_size / 6.0  # Standard deviation
        smoothed = signal.gaussian_filter1d(ear_sequence, sigma=sigma).tolist()
    
    elif method == 'median':
        # Median filter
        for i in range(len(ear_sequence)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(ear_sequence), i + window_size // 2 + 1)
            window_values = ear_sequence[start_idx:end_idx]
            smoothed.append(np.median(window_values))
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed


def detect_microsleep(
    ear_sequence: List[float],
    timestamps: List[float],
    ear_threshold: float = 0.25,
    min_duration: float = 0.5,  # seconds
    max_duration: float = 2.0   # seconds
) -> List[Dict[str, Any]]:
    """
    Detect microsleep episodes (brief involuntary sleep periods).
    
    Args:
        ear_sequence: Sequence of EAR values
        timestamps: Corresponding timestamps
        ear_threshold: EAR threshold for eye closure
        min_duration: Minimum duration for microsleep
        max_duration: Maximum duration for microsleep
        
    Returns:
        List of detected microsleep episodes
    """
    if len(ear_sequence) != len(timestamps):
        raise ValueError("EAR sequence and timestamps must have same length")
    
    microsleeps = []
    
    # Find consecutive periods of eye closure
    in_closure = False
    closure_start_idx = 0
    
    for i, ear in enumerate(ear_sequence):
        if ear < ear_threshold and not in_closure:
            # Start of eye closure
            in_closure = True
            closure_start_idx = i
        elif ear >= ear_threshold and in_closure:
            # End of eye closure
            in_closure = False
            
            # Check duration
            duration = timestamps[i-1] - timestamps[closure_start_idx]
            
            if min_duration <= duration <= max_duration:
                closure_ear_values = ear_sequence[closure_start_idx:i]
                
                microsleeps.append({
                    'start_time': timestamps[closure_start_idx],
                    'end_time': timestamps[i-1],
                    'duration': duration,
                    'start_frame': closure_start_idx,
                    'end_frame': i-1,
                    'min_ear': min(closure_ear_values),
                    'avg_ear': np.mean(closure_ear_values),
                    'severity': 1.0 - np.mean(closure_ear_values)  # Lower EAR = higher severity
                })
    
    return microsleeps


def analyze_blink_patterns(
    blinks: List[Dict[str, Any]],
    time_window: float = 300.0  # 5 minutes
) -> Dict[str, Any]:
    """
    Analyze blink patterns for anomalies.
    
    Args:
        blinks: List of detected blinks
        time_window: Analysis time window in seconds
        
    Returns:
        Pattern analysis results
    """
    if not blinks:
        return {
            'blink_rate': 0.0,
            'avg_duration': 0.0,
            'duration_variance': 0.0,
            'pattern_regularity': 0.0,
            'anomaly_score': 0.0
        }
    
    # Extract blink durations
    durations = [blink['duration_frames'] for blink in blinks]
    
    # Calculate statistics
    avg_duration = np.mean(durations)
    duration_variance = np.var(durations)
    
    # Calculate blink rate (assuming 30 fps)
    fps = 30.0
    total_time = time_window
    blink_rate = len(blinks) / (total_time / 60.0)  # blinks per minute
    
    # Analyze pattern regularity
    if len(blinks) > 3:
        # Calculate inter-blink intervals
        intervals = []
        for i in range(1, len(blinks)):
            interval = blinks[i]['start_frame'] - blinks[i-1]['end_frame']
            intervals.append(interval / fps)  # Convert to seconds
        
        interval_variance = np.var(intervals) if intervals else 0.0
        pattern_regularity = 1.0 / (1.0 + interval_variance)  # Higher = more regular
    else:
        pattern_regularity = 0.0
    
    # Calculate anomaly score
    anomaly_score = 0.0
    
    # Abnormal blink rate
    if blink_rate < 5 or blink_rate > 30:
        anomaly_score += 0.3
    
    # Abnormal duration variance
    if duration_variance > 10:  # High variance in blink durations
        anomaly_score += 0.2
    
    # Low pattern regularity
    if pattern_regularity < 0.3:
        anomaly_score += 0.2
    
    # Very short or very long average duration
    if avg_duration < 2 or avg_duration > 15:  # frames
        anomaly_score += 0.3
    
    anomaly_score = min(1.0, anomaly_score)
    
    return {
        'blink_rate': blink_rate,
        'avg_duration': avg_duration,
        'duration_variance': duration_variance,
        'pattern_regularity': pattern_regularity,
        'anomaly_score': anomaly_score,
        'total_blinks': len(blinks)
    }


def adaptive_threshold_calibration(
    ear_history: List[float],
    blink_labels: Optional[List[bool]] = None,
    method: str = 'statistical'
) -> float:
    """
    Adaptively calibrate EAR threshold for individual users.
    
    Args:
        ear_history: History of EAR values
        blink_labels: Optional ground truth blink labels
        method: Calibration method ('statistical', 'supervised')
        
    Returns:
        Calibrated threshold value
    """
    if len(ear_history) < 50:  # Need sufficient data
        return 0.25  # Default threshold
    
    ear_array = np.array(ear_history)
    
    if method == 'statistical':
        # Statistical approach: threshold = mean - k*std
        mean_ear = np.mean(ear_array)
        std_ear = np.std(ear_array)
        
        # Use 2 standard deviations below mean
        threshold = mean_ear - 2 * std_ear
        
        # Ensure reasonable bounds
        threshold = max(0.15, min(0.35, threshold))
    
    elif method == 'supervised' and blink_labels is not None:
        # Supervised approach using ROC analysis
        if len(blink_labels) != len(ear_history):
            raise ValueError("EAR history and blink labels must have same length")
        
        # Find threshold that maximizes F1 score
        thresholds = np.linspace(0.1, 0.4, 100)
        best_f1 = 0
        best_threshold = 0.25
        
        for thresh in thresholds:
            predictions = ear_array < thresh
            
            # Calculate F1 score
            tp = np.sum(predictions & blink_labels)
            fp = np.sum(predictions & ~blink_labels)
            fn = np.sum(~predictions & blink_labels)
            
            if tp + fp == 0 or tp + fn == 0:
                continue
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            if precision + recall == 0:
                continue
            
            f1 = 2 * precision * recall / (precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        threshold = best_threshold
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return threshold


def extract_drowsiness_features(
    ear_history: List[float],
    blink_history: List[bool],
    timestamps: List[float],
    window_size: int = 180  # 3 minutes at 1 fps
) -> np.ndarray:
    """
    Extract comprehensive features for drowsiness detection.
    
    Args:
        ear_history: History of EAR values
        blink_history: History of blink detections
        timestamps: Corresponding timestamps
        window_size: Feature extraction window size
        
    Returns:
        Feature vector for drowsiness classification
    """
    if len(ear_history) < window_size:
        # Pad with zeros if insufficient data
        return np.zeros(20)
    
    # Use most recent window
    recent_ear = ear_history[-window_size:]
    recent_blinks = blink_history[-window_size:]
    recent_times = timestamps[-window_size:]
    
    features = []
    
    # EAR-based features
    features.extend([
        np.mean(recent_ear),           # Mean EAR
        np.std(recent_ear),            # EAR standard deviation
        np.median(recent_ear),         # Median EAR
        np.percentile(recent_ear, 25), # 25th percentile
        np.percentile(recent_ear, 75), # 75th percentile
        np.min(recent_ear),            # Minimum EAR
        np.max(recent_ear),            # Maximum EAR
    ])
    
    # Blink-based features
    blink_count = sum(recent_blinks)
    total_time = recent_times[-1] - recent_times[0]
    blink_rate = blink_count / (total_time / 60.0) if total_time > 0 else 0
    
    features.extend([
        blink_rate,                    # Blinks per minute
        blink_count,                   # Total blinks in window
        np.mean(recent_blinks),        # Blink frequency
    ])
    
    # PERCLOS features
    ear_threshold = 0.25
    perclos = sum(1 for ear in recent_ear if ear < ear_threshold) / len(recent_ear)
    features.append(perclos)
    
    # Temporal features
    if len(recent_ear) > 1:
        ear_diff = np.diff(recent_ear)
        features.extend([
            np.mean(ear_diff),         # Mean EAR change
            np.std(ear_diff),          # EAR change variability
            np.sum(np.abs(ear_diff)),  # Total EAR variation
        ])
    else:
        features.extend([0, 0, 0])
    
    # Advanced statistical features
    features.extend([
        len([i for i in range(1, len(recent_ear)) if recent_ear[i] < recent_ear[i-1]]), # Decreasing trend count
        np.var(recent_ear),           # EAR variance
        sum(1 for ear in recent_ear if ear < 0.2),  # Very low EAR count
        sum(1 for ear in recent_ear if ear > 0.35), # Very high EAR count
    ])
    
    # Ensure fixed feature size
    features = features[:20]  # Take first 20 features
    while len(features) < 20:
        features.append(0.0)  # Pad if necessary
    
    return np.array(features, dtype=np.float32)


def visualize_ear_timeline(
    ear_sequence: List[float],
    blink_detections: List[bool],
    timestamps: Optional[List[float]] = None,
    threshold: float = 0.25,
    save_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Create a visualization of EAR timeline with blink detections.
    
    Args:
        ear_sequence: Sequence of EAR values
        blink_detections: Corresponding blink detections
        timestamps: Optional timestamps (uses frame numbers if None)
        threshold: EAR threshold line to display
        save_path: Optional path to save the plot
        
    Returns:
        Plot image as numpy array if save_path is None
    """
    try:
        import matplotlib.pyplot as plt
        
        if timestamps is None:
            timestamps = list(range(len(ear_sequence)))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot EAR sequence
        ax.plot(timestamps, ear_sequence, 'b-', linewidth=1, label='EAR', alpha=0.7)
        
        # Plot threshold line
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        
        # Highlight blink detections
        blink_timestamps = [timestamps[i] for i, blink in enumerate(blink_detections) if blink]
        blink_ears = [ear_sequence[i] for i, blink in enumerate(blink_detections) if blink]
        
        if blink_timestamps:
            ax.scatter(blink_timestamps, blink_ears, color='red', s=30, label='Blinks', zorder=5)
        
        # Fill area below threshold
        ax.fill_between(timestamps, 0, threshold, alpha=0.2, color='red', label='Blink zone')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Eye Aspect Ratio (EAR)')
        ax.set_title('EAR Timeline with Blink Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        ax.set_ylim(0, max(0.5, max(ear_sequence) * 1.1))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return None
        else:
            # Convert plot to numpy array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return buf
            
    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return None
    except Exception as e:
        logger.error(f"Error creating EAR visualization: {e}")
        return None
